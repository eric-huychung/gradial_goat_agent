"""Tile selection — which tiles a pass attempts. This is the strategy."""
from __future__ import annotations

from abc import ABC, abstractmethod

import jeopardy as jp

from config import AgentConfig


class TileSelector(ABC):
    """Picks tiles off the board. Subclass me — this is most of the game.

    `jp.open_tiles()` hands back EVERY open variant, because every one of them
    is claimable right now and a real agent works them in parallel. A baseline
    with no tools and no loop must not walk that list: at one submission per
    three seconds, a few hundred tiles is the whole event spent proving the
    same thing.

    `select()` is a template method: TASK_FILTER always wins (an explicit
    request to attempt exactly these tiles), otherwise it defers to
    `_choose()`, which subclasses implement with their own strategy.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def select(self, board: dict) -> list[str]:
        tiles = jp.open_tiles(board)
        if self._config.task_filter:
            return self._from_filter(tiles)
        return self._choose(tiles)

    @abstractmethod
    def _choose(self, tiles: list[dict]) -> list[str]:
        """Pick up to `self._config.max_tiles` ids from every open tile."""

    def _from_filter(self, tiles: list[dict]) -> list[str]:
        open_now = {t["id"] for t in tiles}
        wanted = self._config.task_filter
        missing = [t for t in wanted if t not in open_now]
        if missing:
            jp.log(f"TASK_FILTER: not open right now, skipping {missing}")
        return [t for t in wanted if t in open_now]


class OnePerCellSelector(TileSelector):
    """One tile per cell — a spread of different tasks rather than eight
    clones of one — capped at `max_tiles`. Board order (richest first).
    """

    def _choose(self, tiles: list[dict]) -> list[str]:
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


class LowestPointsFirstSelector(TileSelector):
    """Cheap tiles first. Low-point tiles are typically the easiest, so
    clearing them fast banks points before spending time on tiles that might
    not pay off at all. Still one tile per cell, just visited ascending by
    point value instead of the board's richest-first order.
    """

    def _choose(self, tiles: list[dict]) -> list[str]:
        cheapest_first = sorted(tiles, key=lambda t: t.get("points", 0))
        picked: list[str] = []
        seen_cells: set[tuple] = set()
        for t in cheapest_first:
            cell = (t.get("category"), t.get("points"))
            if cell in seen_cells:
                continue                # already sampling this cell's task
            seen_cells.add(cell)
            picked.append(t["id"])
            if len(picked) >= self._config.max_tiles:
                break
        return picked
