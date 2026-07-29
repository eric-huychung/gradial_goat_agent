"""Solving a single tile — the model call and the submission."""
from __future__ import annotations

import pathlib

import jeopardy as jp

from config import AgentConfig


class NaiveSolver:
    """One model call. No tools. No loop. The floor you are trying to beat.

    An agent replaces this: tools the model can call, a loop that feeds
    results back, and verification before anything is submitted.
    """

    MAX_TOKENS = 1000

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

        answer = self._ask_model(prompt)
        if self._config.verbose:
            jp.log(f"{task_id} model replied:\n{answer}\n---")

        return self._submit(task_id, answer)

    def _build_prompt(self, detail: dict, workdir: pathlib.Path,
                      names: list[str]) -> str:
        return (
            f"{detail['prompt']}\n\n"
            f"Files downloaded to {workdir}: {names or 'none'}\n"
            f"Answer checking: {detail.get('answer_format', 'exact')}\n\n"
            "Reply with ONLY the final answer — no working, no explanation."
        )

    def _ask_model(self, prompt: str) -> str:
        resp = self._anthropic.messages.create(
            model=jp.MODEL, max_tokens=self.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content
                       if b.type == "text").strip()

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
