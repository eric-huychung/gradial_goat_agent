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

HOOKS THE SOLVER SHOULD CALL
----------------------------
Ranking works on board data alone, so none of this is required — but without
it the selector cannot tell a solved tile from an unattempted one:

    selector.book.start(task_id)                 # before solving
    selector.book.record_result(task_id, resp)   # the jp.submit response
    selector.book.record_abandon(task_id)        # gave up without submitting
    await selector.claims.is_taken(task_id)      # between tool turns -> stop

`record_abandon` is the one that is easy to miss: a solver that hits its turn
cap returns without submitting, so no server response exists to report, and an
unreported tile keeps full priority and is re-picked immediately, forever.

Calibrate on the practice board without submitting anything:

    python strategy.py            # one board report
    python strategy.py --watch    # poll, and show what is draining fastest
"""
from __future__ import annotations

import asyncio
import random
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
# is legal — but sibling variants are near-identical difficulty, and our solve
# is slow relative to other teams' agents, so a second worker on the same
# cell is a second worker likely to arrive after it is already gone. Spread
# wide across cells instead of doubling up.
PER_CELL_CAP = 1

# Local cooldown after a miss. The server's own cooldown is authoritative
# (`retry_in`); this is the floor used when it doesn't say.
DEFAULT_RETRY_SECONDS = 30.0

# Assumed solve time before we have real data of our own, seconds. This
# agent is slower than the field, so guess pessimistic rather than instant —
# `TileBook.average_solve_seconds` replaces this with a measured value the
# moment a handful of tiles have actually finished.
DEFAULT_SOLVE_SECONDS = 90.0

# How hard "will this cell still have stock once our slow solve finishes"
# pulls the score down. 0 ignores our own speed entirely (the old behaviour);
# 1 lets a cell projected to fully drain before we'd finish score at zero,
# however good its tier/category weight looks right now.
SURVIVAL_WEIGHT = 0.8

# A tile the solver gave up on (turn cap, timeout) is parked this long. Long
# enough that workers move on to tiles that pay sooner, short enough that it
# comes back inside the round — an abandoned tile is a donated tile.
ABANDON_RETRY_SECONDS = 120.0
# Score multiplier per abandon. Below 1 so repeatedly-slow tiles sink, never 0
# so they stay reachable once the quick tiles run out.
ABANDON_PENALTY = 0.6

# How stale a claim check may be before `ClaimWatch` re-reads the board. The
# board read is shared by every worker, so this is a per-agent cost, not a
# per-tile one.
CLAIM_REFRESH_SECONDS = 5.0

# `GreedyBreadthSelector` only. Shuffling cell and variant order means two
# passes (or two agents on the same board) don't queue up behind the same
# tiles in the same order, which is where first-solve-wins races are lost.
GREEDY_SHUFFLE = True


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

    def predicted_survival(self, solve_seconds: float) -> float:
        """Fraction of the stack expected to still be open `solve_seconds` from now.

        `heat`/`availability` describe the cell as of the last board read;
        neither says whether it will still exist once OUR solve — slower than
        the field's — actually finishes. Project the observed drain rate
        forward by our own solve time instead of reading it as of right now.
        """
        if self.total <= 0:
            return 1.0
        projected_drained = self.drain_per_min * (solve_seconds / 60.0)
        predicted_remaining = self.remaining - projected_drained
        return max(0.0, min(1.0, predicted_remaining / self.total))


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


# ---- mid-solve claim checking ---------------------------------------------

class ClaimWatch:
    """Answers "is this tile still worth finishing?" during a solve.

    Selection-time filtering is not enough: tiles are first-correct-wins, so a
    tile can be taken by another team while we are ten tool-turns into it, and
    every turn after that is spent buying nothing. A solver that checks this
    between turns can drop the tile and free the worker.

    `is_open`/`is_taken` are coroutines because a stale check refetches the
    board — doing that behind a synchronous-looking call would stall every
    other tile on the loop. The board read is cached for
    `CLAIM_REFRESH_SECONDS` and guarded by a lock, so the dozens of tiles
    polling this per tool-turn share ONE request every few seconds rather than
    each firing their own.

    Fails OPEN: if the board cannot be read, tiles are reported still open. A
    network blip must not throw away work that is nearly finished.
    """

    def __init__(self, refresh_seconds: float = CLAIM_REFRESH_SECONDS) -> None:
        self._refresh_seconds = refresh_seconds
        self._open_ids: set[str] | None = None
        self._checked_at = 0.0
        self._lock = asyncio.Lock()

    def observe(self, board: dict) -> None:
        """Reuse a board someone else already fetched."""
        try:
            self._open_ids = {t["id"] for t in jp.open_tiles(board)}
            self._checked_at = time.monotonic()
        except Exception as e:                                  # noqa: BLE001
            jp.log(f"claimwatch: board unreadable, assuming all open — {e!r}")

    def _is_fresh(self) -> bool:
        return (self._open_ids is not None
                and time.monotonic() - self._checked_at < self._refresh_seconds)

    async def _refresh_if_stale(self) -> None:
        if self._is_fresh():
            return
        async with self._lock:
            # Re-checked inside the lock: while we queued for it, whoever held
            # it did the fetch, and repeating it is the stampede this avoids.
            if self._is_fresh():
                return
            try:
                self.observe(await jp.board())
            except jp.AuthError:
                raise                               # fatal; not our call
            except Exception as e:                              # noqa: BLE001
                jp.log(f"claimwatch: board fetch failed, assuming all open — {e!r}")
                self._checked_at = time.monotonic()  # don't hammer a dead server

    async def is_open(self, task_id: str) -> bool:
        await self._refresh_if_stale()
        if not self._open_ids:
            # Empty, not just None. A malformed board parses to zero open
            # tiles without raising, and reading that as "everything is taken"
            # would abandon every solve in flight. A board that really is
            # empty costs one `already_claimed` per submit, which is free.
            return True
        return task_id in self._open_ids

    async def is_taken(self, task_id: str) -> bool:
        return not await self.is_open(task_id)


# ---- per-tile bookkeeping -------------------------------------------------

@dataclass
class TileRecord:
    attempts: int = 0
    misses: int = 0
    abandons: int = 0
    slowest_seconds: float = 0.0
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
    started_at: dict[str, float] = field(default_factory=dict)

    def _record(self, task_id: str) -> TileRecord:
        return self.records.setdefault(task_id, TileRecord())

    def start(self, task_id: str) -> None:
        self.in_progress.add(task_id)
        self.started_at[task_id] = time.monotonic()
        self._record(task_id).attempts += 1

    def finish(self, task_id: str) -> None:
        self.in_progress.discard(task_id)
        self.started_at.pop(task_id, None)

    def elapsed(self, task_id: str) -> float:
        """Seconds the current attempt has been running, 0 if not running."""
        started = self.started_at.get(task_id)
        return 0.0 if started is None else time.monotonic() - started

    def record_abandon(self, task_id: str, reason: str = "gave up") -> None:
        """The solver stopped without submitting — turn cap, timeout, crash.

        This is the case `record_result` never sees: no submission means no
        server response, so without this the tile keeps its full score and is
        re-picked on the very next pass, at the same cost, forever. Park it
        instead, and let it come back when the cheap work is gone.
        """
        record = self._record(task_id)
        record.abandons += 1
        record.slowest_seconds = max(record.slowest_seconds,
                                     self.elapsed(task_id))
        record.retry_at = time.monotonic() + ABANDON_RETRY_SECONDS
        self.finish(task_id)
        jp.log(f"{task_id}: parked for {ABANDON_RETRY_SECONDS:.0f}s "
               f"({reason}, {record.abandons} so far)")

    def record_result(self, task_id: str, result: dict | str) -> None:
        """Feed a `jp.submit` response (or bare result string) back in."""
        if isinstance(result, str):
            result, retry_in = {"result": result}, None
        else:
            retry_in = result.get("retry_in")
        outcome = result.get("result")
        record = self._record(task_id)
        record.slowest_seconds = max(record.slowest_seconds,
                                     self.elapsed(task_id))
        self.finish(task_id)

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

    def average_solve_seconds(self, default: float = DEFAULT_SOLVE_SECONDS) -> float:
        """Mean wall time of our own finished solves — how slow "slow" is.

        Self-calibrating: falls back to `default` until enough tiles have
        actually finished to measure, then reflects this agent's real pace
        instead of a guess.
        """
        finished = [r.slowest_seconds for r in self.records.values()
                   if r.done and r.slowest_seconds > 0]
        if not finished:
            return default
        return sum(finished) / len(finished)

    def miss_count(self, task_id: str) -> int:
        record = self.records.get(task_id)
        return record.misses if record else 0

    def abandon_count(self, task_id: str) -> int:
        record = self.records.get(task_id)
        return record.abandons if record else 0


# ---- the selector ---------------------------------------------------------

@dataclass
class ScoredTile:
    task_id: str
    category: str | None
    points: int
    score: float
    heat: float
    availability: float
    survival: float


class TierPrioritySelector(TileSelector):
    """Ranks open tiles by tier, contention, and per-category confidence.

    Order is 400 -> 300 -> 500 -> 200 -> 100 (see `TIER_WEIGHTS`), pulled
    around by how fast each cell is draining and how well the agent does in
    that category. `TASK_FILTER` still wins outright — the base class handles
    that before `_choose` is ever called.
    """

    def __init__(self, config: AgentConfig,
                 watcher: BoardWatcher | None = None,
                 book: TileBook | None = None,
                 claims: ClaimWatch | None = None) -> None:
        super().__init__(config)
        self.watcher = watcher or BoardWatcher()
        self.book = book or TileBook()
        # Shared with the solver so mid-solve claim checks reuse these reads.
        self.claims = claims or ClaimWatch()

    def select(self, board: dict) -> list[str]:
        try:
            self.watcher.observe(board)
            self.claims.observe(board)
        except Exception as e:                              # noqa: BLE001
            jp.log(f"strategy: contention read failed, ignoring — {e!r}")
        return super().select(board)

    def _choose(self, tiles: list[dict]) -> list[str]:
        try:
            ranked = self.rank(tiles)
        except Exception as e:                              # noqa: BLE001
            # Board order is a worse plan than the ranking, and a far better
            # one than an empty pass. `.get` because this path exists for
            # malformed tiles, and must not fail on one too.
            jp.log(f"strategy: ranking failed, using board order — {e!r}")
            ids = [t.get("id") for t in tiles if t.get("id")]
            return ids[:self._config.max_tiles]

        if self._config.verbose:
            self._log_ranking(ranked)
        return [s.task_id for s in ranked[:self._config.max_tiles]]

    def rank(self, tiles: list[dict]) -> list[ScoredTile]:
        """Every pickable tile, best first, respecting `PER_CELL_CAP`."""
        # A tile with no usable id would be scored, picked, and submitted as an
        # empty task_id, so drop it here rather than at the server.
        scored = [self.score(t) for t in tiles
                  if t.get("id") and self.book.is_available(t["id"])]
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
        solve_seconds = self.book.average_solve_seconds()
        survival = (history.predicted_survival(solve_seconds)
                   if history else 1.0)

        score = _tier_weight(points) * _category_weight(category)
        score *= 1.0 - HEAT_PENALTY * heat
        score *= 1.0 - SCARCITY_PENALTY * (1.0 - availability)
        # Compensates for a slower-than-the-field solve: a cell projected to
        # drain out before we'd finish is worth approaching zero however good
        # its tier/category weight looks right now.
        score *= (1.0 - SURVIVAL_WEIGHT) + SURVIVAL_WEIGHT * survival
        # Each miss halves the tile's pull without ever removing it: the
        # cooldown already delays it, and a tile we drop is a tile we gift.
        score *= 0.5 ** self.book.miss_count(task_id)
        # Same idea for tiles the solver has already run out of turns on: they
        # sink below fresh work, then resurface once that work is gone.
        score *= ABANDON_PENALTY ** self.book.abandon_count(task_id)

        return ScoredTile(task_id=task_id, category=category, points=points,
                          score=score, heat=heat, availability=availability,
                          survival=survival)

    @staticmethod
    def _log_ranking(ranked: list[ScoredTile], limit: int = 8) -> None:
        for tile in ranked[:limit]:
            jp.log(f"  {tile.task_id:>8} {tile.points:>4}pt "
                   f"score={tile.score:.3f} heat={tile.heat:.2f} "
                   f"avail={tile.availability:.2f} surv={tile.survival:.2f} "
                   f"{tile.category}")


class GreedyBreadthSelector(TileSelector):
    """Naive and greedy: touch as many different cells as possible, fast.

    The opposite bet to `TierPrioritySelector`. No tier preference, no
    contention modelling, no category confidence — every open tile is equally
    worth trying, so the only question is coverage. Tiles are dealt
    round-robin across cells: one from every cell first, then a second from
    every cell, and so on until `max_tiles` runs out. A board of 30 cells and
    a `max_tiles` of 30 therefore touches all 30 cells rather than emptying
    six of them.

    Cell and variant order are shuffled per pass (`GREEDY_SHUFFLE`), so a
    pass that gets cut short by `max_tiles` doesn't cut short at the same
    place every time, and two agents working the same board don't contend for
    an identical, identically-ordered queue.

    Useful as a control when the ranked selector is mis-calibrated: if
    `CATEGORY_WEIGHTS` or the survival projection are steering the agent away
    from tiles it could actually have taken, this scores better by simply not
    having an opinion. Swap it in from `runner.py`:

        from strategy import GreedyBreadthSelector
        self._selector = selector or GreedyBreadthSelector(config)

    Keeps `book`/`claims` so the solver's bookkeeping and mid-solve claim
    checks wire up exactly as they do for the ranked selector.
    """

    def __init__(self, config: AgentConfig,
                 book: TileBook | None = None,
                 claims: ClaimWatch | None = None,
                 shuffle: bool = GREEDY_SHUFFLE) -> None:
        super().__init__(config)
        self.book = book or TileBook()
        self.claims = claims or ClaimWatch()
        self.shuffle = shuffle

    def select(self, board: dict) -> list[str]:
        try:
            self.claims.observe(board)
        except Exception as e:                              # noqa: BLE001
            jp.log(f"greedy: claim read failed, ignoring — {e!r}")
        return super().select(board)

    def _choose(self, tiles: list[dict]) -> list[str]:
        by_cell: dict[tuple, list[str]] = {}
        for tile in tiles:
            task_id = tile.get("id")
            if not task_id or not self.book.is_available(task_id):
                continue
            key = (tile.get("category"), tile.get("points"))
            by_cell.setdefault(key, []).append(task_id)

        cells = list(by_cell.values())
        if self.shuffle:
            random.shuffle(cells)
            for ids in cells:
                random.shuffle(ids)

        picked: list[str] = []
        # Deal one tile per cell per round, so coverage is widest-first and a
        # deep cell never crowds out a cell we haven't touched at all.
        for depth in range(max((len(v) for v in cells), default=0)):
            for ids in cells:
                if depth < len(ids):
                    picked.append(ids[depth])
                    if len(picked) >= self._config.max_tiles:
                        if self._config.verbose:
                            jp.log(f"greedy: {len(picked)} tiles across "
                                   f"{len(cells)} cells")
                        return picked
        if self._config.verbose:
            jp.log(f"greedy: {len(picked)} tiles across {len(cells)} cells")
        return picked


# ---- calibration CLI (read-only; submits nothing) ------------------------

async def _report(selector: TierPrioritySelector) -> None:
    board = await jp.board()
    tiles = jp.open_tiles(board)
    selector.watcher.observe(board)

    by_tier: dict[int, int] = {}
    for tile in tiles:
        points = int(tile.get("points") or 0)
        by_tier[points] = by_tier.get(points, 0) + 1

    jp.log(f"phase={board.get('phase')}  open tiles={len(tiles)}")
    jp.log("  open per tier: "
           + ", ".join(f"{p}={by_tier[p]}" for p in sorted(by_tier)))
    jp.log(f"  assumed solve time: {selector.book.average_solve_seconds():.0f}s "
           f"(measured once tiles finish, else DEFAULT_SOLVE_SECONDS)")

    jp.log("  top picks:")
    for tile in selector.rank(tiles)[:10]:
        jp.log(f"    {tile.task_id:>8} {tile.points:>4}pt "
               f"score={tile.score:.3f} heat={tile.heat:.2f} "
               f"avail={tile.availability:.2f} surv={tile.survival:.2f} "
               f"{tile.category}")

    hottest = [(k, h) for k, h in selector.watcher.hottest(5)
               if h.drain_per_min > 0]
    if hottest:
        jp.log("  draining fastest (other teams are here):")
        for (category, points), history in hottest:
            jp.log(f"    {category} {points}pt "
                   f"{history.drain_per_min:.1f} tiles/min "
                   f"{history.remaining}/{history.total} left")


async def main() -> None:
    config = AgentConfig.from_env()
    selector = TierPrioritySelector(config)
    watch = "--watch" in sys.argv

    try:
        while True:
            try:
                await _report(selector)
            except jp.AuthError:
                raise
            except Exception as e:                          # noqa: BLE001
                jp.log(f"scout: board read failed, retrying — {e!r}")
            if not watch:
                return
            jp.log("---")
            await asyncio.sleep(10)
    finally:
        await jp.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except jp.AuthError as e:
        raise SystemExit(f"[auth] {e}")
