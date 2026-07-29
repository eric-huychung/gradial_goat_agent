"""Tile selection — which tiles a pass attempts. This is the strategy."""
from __future__ import annotations

import jeopardy as jp

from config import AgentConfig


class TileSelector:
    """Picks tiles off the board. Replace me — this is most of the game.

    `jp.open_tiles()` hands back EVERY open variant, because every one of them
    is claimable right now and a real agent works them in parallel. A baseline
    with no tools and no loop must not walk that list: at one submission per
    three seconds, a few hundred tiles is the whole event spent proving the
    same thing. So: one tile per cell — a spread of different tasks rather
    than eight clones of one — capped at `max_tiles`.

    Using the full width instead of throwing it away is most of the hackathon.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def select(self, board: dict) -> list[str]:
        tiles = jp.open_tiles(board)
        if self._config.task_filter:
            return self._from_filter(tiles)
        return self._one_per_cell(tiles)

    def _from_filter(self, tiles: list[dict]) -> list[str]:
        open_now = {t["id"] for t in tiles}
        wanted = self._config.task_filter
        missing = [t for t in wanted if t not in open_now]
        if missing:
            jp.log(f"TASK_FILTER: not open right now, skipping {missing}")
        return [t for t in wanted if t in open_now]

    def _one_per_cell(self, tiles: list[dict]) -> list[str]:
        picked: list[str] = []
        seen_cells: set[tuple] = set()
        for t in tiles:
            cell = (t.get("category"), t.get("points"))
            if cell in seen_cells:
                continue                # already sampling this cell's task
            seen_cells.add(cell)
            picked.append(t["id"])
            if len(picked) >= self._config.max_tiles:
                break
        return picked
