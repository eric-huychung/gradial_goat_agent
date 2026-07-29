"""Orchestration — one pass over the board, tiles solved concurrently.

Solving (model calls, file parsing, HTTP probing) has no rate limit and
should run as wide as the board allows. Submitting does have a rate limit —
one call every ~3s — so every tile's solve funnels its final answer through
a single shared `SubmissionQueue` (see submission.py) instead of sleeping
between attempts itself.
"""
from __future__ import annotations

import asyncio

import jeopardy as jp

from config import AgentConfig
from monitor import MONITOR
from solver import NaiveSolver
from strategy import TierPrioritySelector
from submission import SubmissionQueue
from tile_selector import TileSelector


class AgentRunner:
    """Reads the board, picks tiles, attempts them all concurrently."""

    def __init__(self, config: AgentConfig, selector: TileSelector | None = None,
                 solver: NaiveSolver | None = None,
                 submission_queue: SubmissionQueue | None = None) -> None:
        self._config = config
        self._selector = selector or TierPrioritySelector(config)
        self._queue = submission_queue or SubmissionQueue()
        # book/claims are strategy.py's bookkeeping — a plain TileSelector
        # (no such attributes) leaves the solver to run without them.
        self._solver = solver or NaiveSolver(
            config, self._queue,
            book=getattr(self._selector, "book", None),
            claims=getattr(self._selector, "claims", None))

    async def run(self) -> int:
        """Attempt this pass's tiles. Returns how many were solved."""
        board = jp.board()
        self._log_board(board)

        tiles = self._selector.select(board)
        if not tiles:
            jp.log("nothing open — is the board live yet?")
            return 0
        jp.log(f"attempting {len(tiles)} of them, concurrently: {tiles}")

        await self._queue.start()
        try:
            results = await asyncio.gather(
                *(self._attempt(task_id) for task_id in tiles))
        finally:
            await self._queue.stop()

        solved = sum(results)
        jp.log(f"{solved}/{len(tiles)} attempted tiles solved")
        if solved == 0:
            jp.log("Exactly as expected. Now go build an agent — read the "
                   "docstring at the top of main.py.")
        return solved

    async def _attempt(self, task_id: str) -> bool:
        with MONITOR.trace(task_id) as trace:
            try:
                solved = await self._solver.solve(task_id, trace)
                trace.correct = solved
                return solved
            except jp.AuthError:
                raise                                  # fatal; fix the key
            except jp.TileUnavailable as e:
                jp.log(f"{task_id}: not available — {e}")
            except Exception as e:                     # noqa: BLE001
                jp.log(f"{task_id}: blew up — {e!r}")
        return False

    def _log_board(self, board: dict) -> None:
        every = jp.open_tiles(board)
        cells = {(t.get("category"), t.get("points")) for t in every}
        jp.log(f"phase={board.get('phase')}: {len(every)} open tiles across "
               f"{len(cells)} cells — all of them claimable in parallel")
