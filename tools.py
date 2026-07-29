"""The agent's hands: run code, and make stateful HTTP requests.

The naive baseline scores zero because the model can only talk — it cannot
read the 4 MB file it was handed, run the search it needs, or click through a
login. These two tools are the minimum that changes that:

- ``run_python`` executes code in the tile's own workdir (where the files were
  downloaded) and hands back stdout/stderr. It also captures a sentinel line
  ``__ANSWER__=<value>`` so a computed answer can be submitted verbatim,
  without the model ever re-typing it (the transcription-slip the README
  warns about).
- ``http_request`` is one persistent ``httpx.AsyncClient`` per tile, so
  cookies, redirects and login state survive across calls — the whole point
  of the stateful-web category.

Both are coroutines, and neither blocks the event loop: the subprocess is
spawned with ``asyncio.create_subprocess_exec`` and awaited, so a tile
grinding through a 120-second brute force does not freeze the dozen other
tiles solving alongside it. Toolboxes own a connection pool, so ``aclose()``
when the tile is done.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import traceback

import httpx


ANSWER_SENTINEL = "__ANSWER__="

# --- schemas advertised to the model (Anthropic tool-use format) ----------

TOOL_SCHEMAS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python 3 code in this tile's working directory, where "
            "the task files are already downloaded (use relative paths, or "
            "os.listdir('.')). Returns stdout and stderr. Use this to read "
            "and parse files, decode/transform data, brute-force searches, "
            "and to COMPUTE the answer. When you have the final answer, print "
            "it on its own line as '" + ANSWER_SENTINEL + "<value>' — that "
            "exact string is captured verbatim so you never have to retype "
            "it. Packages available: numpy, pandas, requests, httpx, "
            "beautifulsoup4, lxml, plus the stdlib."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "http_request",
        "description": (
            "Make an HTTP request through a persistent session that keeps "
            "cookies and auth across calls — use it for multi-step web flows "
            "(login forms, redirects, paged endpoints). Returns status, "
            "final URL, response headers, and body text (truncated)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "GET, POST, etc.",
                    "default": "GET",
                },
                "url": {"type": "string"},
                "headers": {"type": "object"},
                "params": {"type": "object"},
                "data": {
                    "type": "object",
                    "description": "Form-encoded body fields.",
                },
                "json": {"type": "object", "description": "JSON body."},
                "allow_redirects": {"type": "boolean", "default": True},
            },
            "required": ["url"],
        },
    },
]

MAX_OUTPUT = 12_000          # keep tool results well under the token cap


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 400]
    return f"{head}\n...[truncated {len(text) - limit + 400} chars]..."


class Toolbox:
    """The tools for ONE tile. Holds that tile's workdir, HTTP session, and
    the last value a ``run_python`` call marked with the answer sentinel.
    """

    PYTHON_TIMEOUT = 120.0     # bounds a stuck search
    HTTP_TIMEOUT = 60.0

    def __init__(self, workdir: pathlib.Path) -> None:
        self._workdir = workdir
        self._client = httpx.AsyncClient(timeout=self.HTTP_TIMEOUT)
        self.computed_answer: str | None = None

    async def aclose(self) -> None:
        """Release this tile's connections. One pool per tile, and a board is
        hundreds of tiles wide — leaked pools exhaust the process's sockets."""
        await self._client.aclose()

    async def run(self, name: str, args: dict) -> str:
        """Dispatch one tool call. Never raises — errors come back as text
        the model can read and react to."""
        try:
            if name == "run_python":
                return await self._run_python(args.get("code", ""))
            if name == "http_request":
                return await self._http_request(args)
            return f"[error] unknown tool {name!r}"
        except Exception:                                      # noqa: BLE001
            return "[error]\n" + _truncate(traceback.format_exc())

    # -- run_python --------------------------------------------------------

    async def _run_python(self, code: str) -> str:
        """Run code in a subprocess rooted at the tile's workdir.

        A subprocess (not exec in-process) so a crash, an infinite import, or
        a huge allocation takes down a child, not the agent. Awaited rather
        than waited on, so the other tiles in flight keep their turns moving
        while this one computes. Timeout bounds a stuck search.
        """
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            cwd=str(self._workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.PYTHON_TIMEOUT)
        except (TimeoutError, asyncio.TimeoutError):
            await _kill(proc)
            return (f"[error] run_python timed out after "
                    f"{self.PYTHON_TIMEOUT:.0f}s and was killed. Narrow the "
                    f"search or process the file in chunks.")
        except asyncio.CancelledError:
            # The tile was dropped (claimed elsewhere). Don't leave the child
            # running — nothing will ever reap it.
            await _kill(proc)
            raise

        out = stdout.decode("utf-8", "replace")
        err = stderr.decode("utf-8", "replace")

        for line in out.splitlines():
            if line.startswith(ANSWER_SENTINEL):
                self.computed_answer = line[len(ANSWER_SENTINEL):].strip()

        parts = []
        if out:
            parts.append("stdout:\n" + _truncate(out))
        if err:
            parts.append("stderr:\n" + _truncate(err))
        if proc.returncode != 0:
            parts.append(f"(exit code {proc.returncode})")
        if self.computed_answer is not None:
            parts.append(f"(captured answer: {self.computed_answer!r})")
        return "\n".join(parts) or "(no output)"

    # -- http_request ------------------------------------------------------

    async def _http_request(self, args: dict) -> str:
        method = (args.get("method") or "GET").upper()
        url = args["url"]
        r = await self._client.request(
            method,
            url,
            headers=args.get("headers"),
            params=args.get("params"),
            data=args.get("data"),
            json=args.get("json"),
            follow_redirects=args.get("allow_redirects", True),
        )
        body = r.text or ""
        header_lines = "\n".join(f"{k}: {v}" for k, v in r.headers.items())
        return _truncate(
            f"HTTP {r.status_code} {r.reason_phrase}\n"
            f"final URL: {r.url}\n"
            f"headers:\n{header_lines}\n\n"
            f"body:\n{body}"
        )


async def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill a child and reap it, tolerating one that already exited."""
    try:
        proc.kill()
    except ProcessLookupError:
        return
    await proc.wait()
