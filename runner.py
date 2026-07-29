"""Orchestration — one pass over the board."""
from __future__ import annotations

import time

import jeopardy as jp

from config import AgentConfig
from solver import NaiveSolver
from tile_selector import TileSelector


class AgentRunner:
    """Reads the board, picks tiles, attempts each one serially."""

    SUBMIT_COOLDOWN_SECONDS = 4

    def __init__(self, config: AgentConfig, selector: TileSelector | None = None,
                 solver: NaiveSolver | None = None) -> None:
        self._config = config
        self._selector = selector or TileSelector(config)
        self._solver = solver or NaiveSolver(config)

    def run(self) -> int:
        """Attempt this pass's tiles. Returns how many were solved."""
        board = jp.board()
        self._log_board(board)

        tiles = self._selector.select(board)
        if not tiles:
            jp.log("nothing open — is the board live yet?")
            return 0
        jp.log(f"naive baseline is attempting {len(tiles)} of them, serially: "
               f"{tiles}")

        solved = sum(self._attempt(task_id) for task_id in tiles)
        jp.log(f"naive baseline: {solved}/{len(tiles)} attempted tiles solved")
        if solved == 0:
            jp.log("Exactly as expected. Now go build an agent — read the "
                   "docstring at the top of main.py.")
        return solved

    def _attempt(self, task_id: str) -> bool:
        try:
            return self._solver.solve(task_id)
        except jp.AuthError:
            raise                                  # fatal; fix the key
        except jp.TileUnavailable as e:
            jp.log(f"{task_id}: not available — {e}")
        except Exception as e:                     # noqa: BLE001
            jp.log(f"{task_id}: blew up — {e!r}")
        finally:
            time.sleep(self.SUBMIT_COOLDOWN_SECONDS)   # submission rate limit
        return False

    def _log_board(self, board: dict) -> None:
        every = jp.open_tiles(board)
        cells = {(t.get("category"), t.get("points")) for t in every}
        jp.log(f"phase={board.get('phase')}: {len(every)} open tiles across "
               f"{len(cells)} cells — all of them claimable in parallel")
