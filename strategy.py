"""Tile-picking strategy — which tiles to attempt, and in what order.

A new file on purpose: it subclasses `tile_selector.TileSelector`, so nothing
in `tile_selector.py`, `solver.py` or `runner.py` has to change. To use it,
one line in `runner.py` swaps the default selector:

    from strategy import TierPrioritySelector
    ...
    self._selector = selector or TierPrioritySelector(config)

THE IDEA
--------
Twelve teams share one pool of tiles and the first correct submission takes a
tile forever, so picking well matters as much as solving well. Three signals
drive the ranking:

1. TIER. 400s first: near-500 value, but 500s draw every team that is
   optimising for points and 100/200s draw every team that is optimising for
   volume. The 300/400 middle is where the board is quietest.
2. CONTENTION. A cell publishes `remaining`/`total`, so draining stacks are
   visible. A cell being emptied fast is one whose tiles will be gone before a
   slow solve finishes; a cell still near full is a safer place to spend a
   worker.
3. CONFIDENCE. Per-category weights, calibrated on the practice board. A
   category the agent cannot clear is worth zero points however cheap it is.

Everything degrades instead of raising: a failed observation or a bad score
falls back to board order rather than taking the pass down with it.

Calibrate on the practice board without submitting anything:

    python strategy.py            # one board report
    python strategy.py --watch    # poll, and show what is draining fastest
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import jeopardy as jp

from config import AgentConfig
from tile_selector import TileSelector

# ---- tunables — calibrate these on the practice board ---------------------

# Preference by point tier, higher is picked sooner. 400 leads, 300 backs it
# up, 500 only after those, cheap tiers last.
TIER_WEIGHTS: dict[int, float] = {400: 1.00, 300: 0.85, 500: 0.70,
                                  200: 0.50, 100: 0.40}
DEFAULT_TIER_WEIGHT = 0.60          # a tier not listed above

# How well the agent actually does per category, 0..1. All neutral until
# practice says otherwise: drop a category the agent keeps missing, raise one
# it clears in seconds. These six names are the whole board.
CATEGORY_WEIGHTS: dict[str, float] = {
    "Needle in the Haystack": 1.0,
    "The Dark Web": 1.0,
    "Ship It": 1.0,
    "Ancient Scrolls": 1.0,
    "Cryptic": 1.0,
    "Heavy Compute": 1.0,
}
DEFAULT_CATEGORY_WEIGHT = 1.0

# A cell losing this many tiles per minute counts as fully contested.
HEAT_REFERENCE_PER_MIN = 3.0
# How much a fully contested cell is docked (0 = ignore contention entirely).
HEAT_PENALTY = 0.35
# How much an almost-exhausted cell is docked versus an untouched one.
SCARCITY_PENALTY = 0.25

# Workers on variants of the same cell. Stacked tiles are independent, so >1
# is legal — but sibling variants are near-identical difficulty, so spreading
# wide is the better bet until a category is known-good.
PER_CELL_CAP = 2

# Local cooldown after a miss. The server's own cooldown is authoritative
# (`retry_in`); this is the floor used when it doesn't say.
DEFAULT_RETRY_SECONDS = 30.0


def _tier_weight(points: int) -> float:
    return TIER_WEIGHTS.get(points, DEFAULT_TIER_WEIGHT)


def _category_weight(category: str | None) -> float:
    return CATEGORY_WEIGHTS.get(category or "", DEFAULT_CATEGORY_WEIGHT)


# ---- contention tracking --------------------------------------------------

@dataclass
class CellHistory:
    """How one cell's stack has drained since we first saw it."""

    first_remaining: int
    remaining: int
    total: int
    first_seen: float
    last_seen: float

    @property
    def availability(self) -> float:
        """1.0 = stack untouched, 0.0 = stack empty."""
        if self.total <= 0:
            return 1.0
        return max(0.0, min(1.0, self.remaining / self.total))

    @property
    def drain_per_min(self) -> float:
        """Tiles claimed per minute since first observed — anyone's claims."""
        elapsed = self.last_seen - self.first_seen
        if elapsed < 1.0:
            return 0.0
        drained = max(0, self.first_remaining - self.remaining)
        return drained / elapsed * 60.0

    @property
    def heat(self) -> float:
        """Contention, 0..1, from the drain rate."""
        if HEAT_REFERENCE_PER_MIN <= 0:
            return 0.0
        return min(1.0, self.drain_per_min / HEAT_REFERENCE_PER_MIN)


class BoardWatcher:
    """Remembers `remaining` per cell across board reads.

    One read tells you a stack is half gone; two tell you whether it is being
    emptied right now. Every poll the runner already makes feeds this for
    free, so no extra requests are needed.
    """

    def __init__(self) -> None:
        self._cells: dict[tuple, CellHistory] = {}

    @staticmethod
    def cell_key(tile: dict) -> tuple:
        return (tile.get("category"), tile.get("points"))

    def observe(self, board: dict) -> None:
        now = time.monotonic()
        seen: dict[tuple, tuple[int, int]] = {}
        for tile in jp.open_tiles(board):
            key = self.cell_key(tile)
            if key in seen:
                continue                    # one entry per cell, not per tile
            remaining = int(tile.get("remaining") or 0)
            total = int(tile.get("total") or remaining or 0)
            seen[key] = (remaining, total)

        for key, (remaining, total) in seen.items():
            history = self._cells.get(key)
            if history is None:
                self._cells[key] = CellHistory(
                    first_remaining=remaining, remaining=remaining,
                    total=total, first_seen=now, last_seen=now)
            else:
                history.remaining = remaining
                history.total = max(history.total, total)
                history.last_seen = now

    def history(self, tile: dict) -> CellHistory | None:
        return self._cells.get(self.cell_key(tile))

    def hottest(self, limit: int = 5) -> list[tuple[tuple, CellHistory]]:
        ranked = sorted(self._cells.items(),
                        key=lambda kv: -kv[1].drain_per_min)
        return ranked[:limit]


# ---- per-tile bookkeeping -------------------------------------------------

@dataclass
class TileRecord:
    attempts: int = 0
    misses: int = 0
    retry_at: float = 0.0
    done: bool = False


@dataclass
class TileBook:
    """What we've already tried, so a miss is a delay and not an abandon.

    A wrong answer earns a cooldown that expires; bookkeeping that quietly
    drops the tile donates it to another team. Nothing here is required —
    unreported results just mean the selector ranks on board data alone.
    """

    records: dict[str, TileRecord] = field(default_factory=dict)
    in_progress: set[str] = field(default_factory=set)

    def _record(self, task_id: str) -> TileRecord:
        return self.records.setdefault(task_id, TileRecord())

    def start(self, task_id: str) -> None:
        self.in_progress.add(task_id)
        self._record(task_id).attempts += 1

    def finish(self, task_id: str) -> None:
        self.in_progress.discard(task_id)

    def record_result(self, task_id: str, result: dict | str) -> None:
        """Feed a `jp.submit` response (or bare result string) back in."""
        if isinstance(result, str):
            result, retry_in = {"result": result}, None
        else:
            retry_in = result.get("retry_in")
        outcome = result.get("result")
        record = self._record(task_id)
        self.in_progress.discard(task_id)

        if outcome in ("correct", "already_claimed", "voided"):
            record.done = True
            return
        if outcome == "incorrect":
            record.misses += 1
        # `is not None`, not truthiness: a stated retry_in of 0 means "go
        # again now", and treating that as missing invents a 30s stall.
        wait = (float(retry_in) if retry_in is not None
                else DEFAULT_RETRY_SECONDS)
        record.retry_at = time.monotonic() + wait

    def is_available(self, task_id: str) -> bool:
        record = self.records.get(task_id)
        if record is None:
            return task_id not in self.in_progress
        if record.done or task_id in self.in_progress:
            return False
        return time.monotonic() >= record.retry_at

    def miss_count(self, task_id: str) -> int:
        record = self.records.get(task_id)
        return record.misses if record else 0


# ---- the selector ---------------------------------------------------------

@dataclass
class ScoredTile:
    task_id: str
    category: str | None
    points: int
    score: float
    heat: float
    availability: float


class TierPrioritySelector(TileSelector):
    """Ranks open tiles by tier, contention, and per-category confidence.

    Order is 400 -> 300 -> 500 -> 200 -> 100 (see `TIER_WEIGHTS`), pulled
    around by how fast each cell is draining and how well the agent does in
    that category. `TASK_FILTER` still wins outright — the base class handles
    that before `_choose` is ever called.
    """

    def __init__(self, config: AgentConfig,
                 watcher: BoardWatcher | None = None,
                 book: TileBook | None = None) -> None:
        super().__init__(config)
        self.watcher = watcher or BoardWatcher()
        self.book = book or TileBook()

    def select(self, board: dict) -> list[str]:
        try:
            self.watcher.observe(board)
        except Exception as e:                              # noqa: BLE001
            jp.log(f"strategy: contention read failed, ignoring — {e!r}")
        return super().select(board)

    def _choose(self, tiles: list[dict]) -> list[str]:
        try:
            picked = [s.task_id for s in self.rank(tiles)]
        except Exception as e:                              # noqa: BLE001
            # Board order is a worse plan than the ranking, and a far better
            # one than an empty pass.
            jp.log(f"strategy: ranking failed, using board order — {e!r}")
            return [t["id"] for t in tiles[:self._config.max_tiles]]

        if self._config.verbose:
            self._log_ranking(tiles)
        return picked[:self._config.max_tiles]

    def rank(self, tiles: list[dict]) -> list[ScoredTile]:
        """Every pickable tile, best first, respecting `PER_CELL_CAP`."""
        scored = [self.score(t) for t in tiles
                  if self.book.is_available(t.get("id", ""))]
        # Stable: equal scores keep the server's own ordering, whose leading
        # variants are the ones it pre-generates and serves fastest.
        scored.sort(key=lambda s: -s.score)

        per_cell: dict[tuple, int] = {}
        out: list[ScoredTile] = []
        for tile in scored:
            key = (tile.category, tile.points)
            if per_cell.get(key, 0) >= PER_CELL_CAP:
                continue
            per_cell[key] = per_cell.get(key, 0) + 1
            out.append(tile)
        return out

    def score(self, tile: dict) -> ScoredTile:
        task_id = tile.get("id", "")
        category = tile.get("category")
        points = int(tile.get("points") or 0)

        history = self.watcher.history(tile)
        heat = history.heat if history else 0.0
        availability = history.availability if history else 1.0

        score = _tier_weight(points) * _category_weight(category)
        score *= 1.0 - HEAT_PENALTY * heat
        score *= 1.0 - SCARCITY_PENALTY * (1.0 - availability)
        # Each miss halves the tile's pull without ever removing it: the
        # cooldown already delays it, and a tile we drop is a tile we gift.
        score *= 0.5 ** self.book.miss_count(task_id)

        return ScoredTile(task_id=task_id, category=category, points=points,
                          score=score, heat=heat, availability=availability)

    def _log_ranking(self, tiles: list[dict], limit: int = 8) -> None:
        for tile in self.rank(tiles)[:limit]:
            jp.log(f"  {tile.task_id:>8} {tile.points:>4}pt "
                   f"score={tile.score:.3f} heat={tile.heat:.2f} "
                   f"avail={tile.availability:.2f} {tile.category}")


# ---- calibration CLI (read-only; submits nothing) ------------------------

def _report(selector: TierPrioritySelector) -> None:
    board = jp.board()
    tiles = jp.open_tiles(board)
    selector.watcher.observe(board)

    by_tier: dict[int, int] = {}
    for tile in tiles:
        points = int(tile.get("points") or 0)
        by_tier[points] = by_tier.get(points, 0) + 1

    jp.log(f"phase={board.get('phase')}  open tiles={len(tiles)}")
    jp.log("  open per tier: "
           + ", ".join(f"{p}={by_tier[p]}" for p in sorted(by_tier)))

    jp.log("  top picks:")
    for tile in selector.rank(tiles)[:10]:
        jp.log(f"    {tile.task_id:>8} {tile.points:>4}pt "
               f"score={tile.score:.3f} heat={tile.heat:.2f} "
               f"avail={tile.availability:.2f} {tile.category}")

    hottest = [(k, h) for k, h in selector.watcher.hottest(5)
               if h.drain_per_min > 0]
    if hottest:
        jp.log("  draining fastest (other teams are here):")
        for (category, points), history in hottest:
            jp.log(f"    {category} {points}pt "
                   f"{history.drain_per_min:.1f} tiles/min "
                   f"{history.remaining}/{history.total} left")


def main() -> None:
    config = AgentConfig.from_env()
    selector = TierPrioritySelector(config)
    watch = "--watch" in sys.argv

    while True:
        try:
            _report(selector)
        except jp.AuthError:
            raise
        except Exception as e:                              # noqa: BLE001
            jp.log(f"scout: board read failed, retrying — {e!r}")
        if not watch:
            return
        jp.log("---")
        time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except jp.AuthError as e:
        raise SystemExit(f"[auth] {e}")
