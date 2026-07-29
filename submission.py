"""Serializes /api/submit calls behind the server's rate limit.

Solving now happens concurrently — many tiles' model calls and tool loops
in flight at once — but the game only allows one submission every 3 seconds
per team. `SubmissionQueue` is the single choke point every solver funnels
through: each `submit()` call queues its answer and awaits its own result,
while one worker drains the queue at a fixed minimum interval, however many
tiles finished solving at the same instant.
"""
from __future__ import annotations

import asyncio
import time

import jeopardy as jp


class SubmissionQueue:
    """One worker, one queue, spaced-out /api/submit calls.

    A small margin over the server's stated 3s limit absorbs clock/network
    jitter — landing exactly on 3.000s risks an occasional `rate_limited`
    that would just cost a retry (and another 3s wait) anyway.
    """

    MIN_INTERVAL_SECONDS = 3.1
    MAX_RATE_LIMIT_RETRIES = 5     # server's global limit, not just ours

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, str, asyncio.Future]] = asyncio.Queue()
        self._last_submit = 0.0
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        """Launch the background worker. Call once before submitting."""
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Drain any queued submissions, then stop the worker."""
        await self._queue.join()
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

    async def submit(self, task_id: str, answer: str) -> dict:
        """Queue an answer and wait for its own submit result.

        Safe to call from many solvers at once — they all resolve in the
        order the queue serves them, each spaced `MIN_INTERVAL_SECONDS`
        apart, but each caller only awaits its own tile's outcome.
        """
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((task_id, answer, fut))
        return await fut

    async def _run(self) -> None:
        while True:
            task_id, answer, fut = await self._queue.get()
            try:
                result = await self._submit_with_retries(task_id, answer)
                fut.set_result(result)
            except Exception as e:                          # noqa: BLE001
                fut.set_exception(e)
            finally:
                self._queue.task_done()

    async def _submit_with_retries(self, task_id: str, answer: str) -> dict:
        """One submit, retried on `rate_limited`.

        Our own pacing keeps us under MIN_INTERVAL_SECONDS, but the limit is
        per-team, not per-process — another agent instance on the same key
        (or a restarted run) can still trip it. `rate_limited` responses
        carry `retry_in`; honor it rather than donating the tile as a loss.
        """
        for attempt in range(self.MAX_RATE_LIMIT_RETRIES + 1):
            wait = self.MIN_INTERVAL_SECONDS - (time.monotonic() - self._last_submit)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_submit = time.monotonic()

            result = await jp.submit(task_id, answer)
            if result.get("result") != "rate_limited":
                return result
            if attempt == self.MAX_RATE_LIMIT_RETRIES:
                return result
            retry_in = float(result.get("retry_in") or self.MIN_INTERVAL_SECONDS)
            jp.log(f"{task_id}: rate_limited, retrying in {retry_in}s "
                   f"(attempt {attempt + 1}/{self.MAX_RATE_LIMIT_RETRIES})")
            await asyncio.sleep(retry_in)
        return result
