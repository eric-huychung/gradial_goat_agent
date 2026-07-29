"""Performance tracking — trace each question/solve unit, dump stats on exit.

``PerformanceMonitor.trace()`` is a context manager around one tile attempt.
It times the block and yields a ``TraceRecord`` for the caller to fill in
(turns taken, whether an answer was produced, whether it was correct). All
records are written to JSON automatically when the process exits, so a run
that crashes or is killed still leaves its stats behind.
"""
from __future__ import annotations

import atexit
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TraceRecord:
    """Stats for one question/solve unit."""

    task_id: str
    category: str | None = None
    points: int | None = None
    turns_taken: int = 0
    answered: bool = False
    correct: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0


class PerformanceMonitor:
    """Traces question/solve units and dumps stats to JSON on exit."""

    def __init__(self, output_path: str | Path = "stats.json") -> None:
        self._output_path = Path(output_path)
        self._records: list[TraceRecord] = []
        atexit.register(self.dump)

    @contextmanager
    def trace(self, task_id: str, category: str | None = None,
              points: int | None = None):
        """Time one question/solve unit.

        Yields a ``TraceRecord`` — set ``turns_taken``, ``answered`` and
        ``correct`` on it as the solve progresses. Recorded even if the block
        raises (with the exception captured in ``error``).
        """
        record = TraceRecord(task_id=task_id, category=category, points=points)
        start = time.monotonic()
        try:
            yield record
        except Exception as e:                          # noqa: BLE001
            record.error = repr(e)
            raise
        finally:
            record.elapsed_seconds = time.monotonic() - start
            self._records.append(record)

    def summary(self) -> dict:
        n = len(self._records)
        answered = sum(1 for r in self._records if r.answered)
        correct = sum(1 for r in self._records if r.correct)
        total_time = sum(r.elapsed_seconds for r in self._records)
        return {
            "tiles_attempted": n,
            "tiles_answered": answered,
            "tiles_correct": correct,
            "total_seconds": total_time,
            "avg_seconds": (total_time / n) if n else 0.0,
        }

    def dump(self, path: str | Path | None = None) -> None:
        """Write all recorded stats to JSON. Registered to run at exit."""
        out = Path(path) if path is not None else self._output_path
        payload = {
            "summary": self.summary(),
            "tiles": [asdict(r) for r in self._records],
        }
        out.write_text(json.dumps(payload, indent=2))


MONITOR = PerformanceMonitor()
