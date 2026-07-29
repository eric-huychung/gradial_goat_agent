"""Solving a single tile — an async, tool-using agent loop.

The naive baseline handed the prompt to the model and submitted whatever came
back. It scored zero because the model cannot see the downloaded files, run
code, or open a page — so it hallucinated confident, wrong answers.

``NaiveSolver`` gives the model hands (``tools.py``) and a loop: it can read
the files, compute, and probe web endpoints, feeding each result back until it
has a real answer. It is async so many tiles can be IN FLIGHT at once —
solving has no rate limit, only submitting does — and every call it makes is
natively awaitable: model calls (AsyncAnthropic), game API calls (httpx),
web probes (httpx), and code execution (asyncio subprocess). No tile ever
blocks another one's turn. Two things protect the score:

- **Structural answer pass-through.** The model computes the answer in
  ``run_python`` and prints ``__ANSWER__=<value>``; that captured string is
  submitted verbatim. The model never re-types an exact-match token, so it
  cannot fumble a character on a tile it already solved (README: "Never let
  the model retype an exact-match answer").
- **A turn cap.** A stuck tile can't burn the whole budget.

Submission itself is NOT done here directly — every solve hands its final
answer to a shared ``SubmissionQueue`` (see ``submission.py``), which is the
one place that respects the server's one-submission-per-3-seconds limit
across every tile solving concurrently.
"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import jeopardy as jp

from config import AgentConfig
from monitor import TraceRecord
from submission import SubmissionQueue
from tools import TOOL_SCHEMAS, Toolbox

if TYPE_CHECKING:                    # avoid a hard runtime dependency
    from strategy import ClaimWatch, TileBook


class NaiveSolver:
    """A tool-use agent for one tile. (Name kept so runner.py imports it.)

    Loop: send the prompt + tool schemas, run whatever tool the model asks
    for, feed the result back, repeat until it calls ``submit_answer`` or the
    turn cap is hit. Submission is queued, not sent directly, so concurrent
    solves never violate the server's submission rate limit.
    """

    MAX_TOKENS = 4096          # the proxy's cap; use it all
    MAX_TURNS = 12             # tool round-trips before we give up on a tile

    def __init__(self, config: AgentConfig, submission_queue: SubmissionQueue,
                 client=None, book: "TileBook | None" = None,
                 claims: "ClaimWatch | None" = None) -> None:
        self._config = config
        self._queue = submission_queue
        self._client = client
        # Both optional: without them the solver just skips bookkeeping and
        # mid-solve claim checks, it doesn't fail.
        self._book = book
        self._claims = claims

    async def solve(self, task_id: str, trace: TraceRecord | None = None) -> bool:
        """Attempt one tile. True if the server scored it correct."""
        if self._book is not None:
            self._book.start(task_id)
        try:
            return await self._solve(task_id, trace)
        except Exception:
            if self._book is not None:
                self._book.record_abandon(task_id, reason="raised")
            raise

    async def _solve(self, task_id: str, trace: TraceRecord | None = None) -> bool:
        detail = await jp.task(task_id)
        workdir = jp.workdir(task_id)          # stable; see jeopardy.workdir
        names = await jp.fetch_files(task_id, detail)

        if trace is not None:
            trace.category = detail.get("category")
            trace.points = detail.get("points")

        prompt = self._build_prompt(detail, workdir, names)
        if self._config.verbose:
            jp.log(f"{task_id} ({detail.get('category')}, "
                   f"{detail.get('points')}pt) prompt:\n{prompt}\n---")

        tools = Toolbox(workdir)
        try:
            answer, turns = await self._run_loop(task_id, prompt, tools)
        finally:
            await tools.aclose()
        if trace is not None:
            trace.turns_taken = turns
            trace.answered = answer is not None
        if answer is None:
            reason = "turn cap" if turns >= self.MAX_TURNS else "claimed elsewhere"
            jp.log(f"{task_id}: no answer ({reason}, {turns} turns)")
            if self._book is not None:
                self._book.record_abandon(task_id, reason=reason)
            return False

        # Last check before spending a submission slot: the mid-loop check
        # only covers the time up to the turn that produced this answer —
        # someone else (or another worker of ours) may have solved it since.
        if self._claims is not None and await self._claims.is_taken(task_id):
            jp.log(f"{task_id}: solved elsewhere before we submitted, skipping")
            if self._book is not None:
                self._book.record_result(task_id, {"result": "already_claimed"})
            return False

        return await self._submit(task_id, answer)

    def _build_prompt(self, detail: dict, workdir: pathlib.Path,
                      names: list[str]) -> str:
        return (
            f"{detail['prompt']}\n\n"
            f"Working directory: {workdir}\n"
            f"Files already downloaded there: {names or 'none'}\n"
            f"Answer format (how the checker compares): "
            f"{detail.get('answer_format', 'exact')}\n\n"
            "You have tools: run_python (task files are in the working "
            "directory — use relative paths), http_request (stateful HTTP "
            "with cookies). Do NOT guess — read the actual data and COMPUTE "
            "the answer. When you are sure, print it in run_python as a line "
            "'__ANSWER__=<value>', then call submit_answer with "
            "use_computed=true so the exact value is submitted without you "
            "retyping it. Only submit once you have verified the answer "
            "against the data."
        )

    # -- the tool-use loop -------------------------------------------------

    def _submit_tool_schema(self) -> dict:
        return {
            "name": "submit_answer",
            "description": (
                "Submit the final answer for this tile. Prefer "
                "use_computed=true to submit the value captured from your "
                "last __ANSWER__= line verbatim (no retyping). Only pass a "
                "raw answer string if you did not compute it in code."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "use_computed": {
                        "type": "boolean",
                        "description": "Submit the last __ANSWER__= value.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "Fallback literal answer.",
                    },
                },
            },
        }

    async def _run_loop(self, task_id: str, prompt: str,
                        tools: Toolbox) -> tuple[str | None, int]:
        schemas = TOOL_SCHEMAS + [self._submit_tool_schema()]
        messages: list[dict] = [{"role": "user", "content": prompt}]

        for turn in range(self.MAX_TURNS):
            if self._claims is not None and await self._claims.is_taken(task_id):
                jp.log(f"{task_id}: claimed elsewhere mid-solve, dropping")
                return None, turn

            resp = await self._anthropic.messages.create(
                model=jp.MODEL,
                max_tokens=self.MAX_TOKENS,
                messages=messages,
                tools=schemas,
            )
            messages.append({"role": "assistant", "content": resp.content})

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                # No tool call — treat any text as a last-resort answer.
                text = "".join(b.text for b in resp.content
                               if b.type == "text").strip()
                if self._config.verbose:
                    jp.log(f"{task_id} turn {turn}: no tool, text={text!r}")
                return text or None, turn + 1

            results = []
            for tu in tool_uses:
                if tu.name == "submit_answer":
                    answer = self._resolve_answer(tu.input or {}, tools)
                    if answer is not None:
                        return answer, turn + 1
                    results.append(self._tool_result(
                        tu.id,
                        "[error] no computed answer captured yet — run "
                        "run_python and print '__ANSWER__=<value>' first."))
                    continue

                # Awaited, not thread-offloaded: the subprocess and the HTTP
                # session are natively async, so other tiles' turns keep
                # making progress meanwhile.
                out = await tools.run(tu.name, tu.input or {})
                if self._config.verbose:
                    jp.log(f"{task_id} turn {turn}: {tu.name} -> {out[:200]!r}")
                results.append(self._tool_result(tu.id, out))

            messages.append({"role": "user", "content": results})

        return None, self.MAX_TURNS

    @staticmethod
    def _resolve_answer(args: dict, tools: Toolbox) -> str | None:
        if args.get("use_computed"):
            return tools.computed_answer   # verbatim; may be None if missing
        raw = args.get("answer")
        return raw.strip() if isinstance(raw, str) and raw.strip() else None

    @staticmethod
    def _tool_result(tool_use_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_use_id,
                "content": content}

    async def _submit(self, task_id: str, answer: str) -> bool:
        # Funnels through the shared queue: many tiles finish concurrently,
        # but only one submission goes out every MIN_INTERVAL_SECONDS.
        result = await self._queue.submit(task_id, answer)
        jp.log(f"{task_id}: answered {answer[:60]!r} -> {result.get('result')}")
        if self._config.verbose:
            jp.log(f"{task_id} full submit response: {result}")
        if self._book is not None:
            self._book.record_result(task_id, result)
        if result.get("result") == "forbidden":
            jp.log("  (a scored round is live — only your HOSTED agent may "
                   "submit. Deploy with /api/agent/submit, or practise on the "
                   "practice board.)")
        return result.get("result") == "correct"

    @property
    def _anthropic(self):
        # Built on first use so a pass that picks no tiles needs no client.
        if self._client is None:
            self._client = jp.anthropic_client()
        return self._client
