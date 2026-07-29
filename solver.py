"""Solving a single tile — a tool-using agent loop, not one blind guess.

The naive baseline handed the prompt to the model and submitted whatever came
back. It scored zero because the model cannot see the downloaded files, run
code, or open a page — so it hallucinated confident, wrong answers.

``NaiveSolver`` now gives the model hands (``tools.py``) and a loop: it can
read the files, compute, and probe web endpoints, feeding each result back
until it has a real answer. Two things protect the score:

- **Structural answer pass-through.** The model computes the answer in
  ``run_python`` and prints ``__ANSWER__=<value>``; that captured string is
  submitted verbatim. The model never re-types an exact-match token, so it
  cannot fumble a character on a tile it already solved (README: "Never let
  the model retype an exact-match answer").
- **A turn cap.** A stuck tile can't burn the whole budget.
"""
from __future__ import annotations

import pathlib

import jeopardy as jp

from config import AgentConfig
from tools import TOOL_SCHEMAS, Toolbox


class NaiveSolver:
    """A tool-use agent for one tile. (Name kept so runner.py imports it.)

    Loop: send the prompt + tool schemas, run whatever tool the model asks
    for, feed the result back, repeat until it calls ``submit_answer`` or the
    turn cap is hit.
    """

    MAX_TOKENS = 4096          # the proxy's cap; use it all
    MAX_TURNS = 12             # tool round-trips before we give up on a tile

    def __init__(self, config: AgentConfig, client=None) -> None:
        self._config = config
        self._client = client

    def solve(self, task_id: str) -> bool:
        """Attempt one tile. True if the server scored it correct."""
        detail = jp.task(task_id)
        workdir = jp.workdir(task_id)          # stable; see jeopardy.workdir
        names = jp.fetch_files(task_id, detail)

        prompt = self._build_prompt(detail, workdir, names)
        if self._config.verbose:
            jp.log(f"{task_id} ({detail.get('category')}, "
                   f"{detail.get('points')}pt) prompt:\n{prompt}\n---")

        tools = Toolbox(workdir)
        answer = self._run_loop(task_id, prompt, tools)
        if answer is None:
            jp.log(f"{task_id}: no answer after {self.MAX_TURNS} turns")
            return False

        return self._submit(task_id, answer)

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

    def _run_loop(self, task_id: str, prompt: str,
                  tools: Toolbox) -> str | None:
        schemas = TOOL_SCHEMAS + [self._submit_tool_schema()]
        messages: list[dict] = [{"role": "user", "content": prompt}]

        for turn in range(self.MAX_TURNS):
            resp = self._anthropic.messages.create(
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
                return text or None

            results = []
            for tu in tool_uses:
                if tu.name == "submit_answer":
                    answer = self._resolve_answer(tu.input or {}, tools)
                    if answer is not None:
                        return answer
                    results.append(self._tool_result(
                        tu.id,
                        "[error] no computed answer captured yet — run "
                        "run_python and print '__ANSWER__=<value>' first."))
                    continue

                out = tools.run(tu.name, tu.input or {})
                if self._config.verbose:
                    jp.log(f"{task_id} turn {turn}: {tu.name} -> {out[:200]!r}")
                results.append(self._tool_result(tu.id, out))

            messages.append({"role": "user", "content": results})

        return None

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

    def _submit(self, task_id: str, answer: str) -> bool:
        result = jp.submit(task_id, answer)
        jp.log(f"{task_id}: answered {answer[:60]!r} -> {result.get('result')}")
        if self._config.verbose:
            jp.log(f"{task_id} full submit response: {result}")
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
