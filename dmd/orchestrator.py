"""Bounded-concurrency synthesis job scheduler with priority and staleness cancellation."""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Awaitable, Callable

from .types import Card, Job

type _Entry = tuple[tuple[int, float, int], Job, Callable[[], Awaitable[Card]], asyncio.Future]


class JobPool:
    """Bounded-concurrency scheduler for synthesis jobs with priority and staleness cancellation."""

    def __init__(
        self,
        max_concurrent: int,
        job_timeout_s: float,
        stale_after_s: float,
        on_card: Callable[[Card], Awaitable[None]] | None = None,
        on_drop: Callable[[Job, str], Awaitable[None]] | None = None,
    ) -> None:
        """Store configuration; no workers are started until the first submit."""
        self._max_concurrent = max_concurrent
        self._job_timeout_s = job_timeout_s
        self._stale_after_s = stale_after_s
        self._on_card = on_card
        self._on_drop = on_drop
        self._queue: list[_Entry] = []
        self._seq = 0
        self._workers: list[asyncio.Task[None]] = []
        self._workers_started = False
        self._closed = False
        self._queued_count = 0
        self._running_count = 0
        self._completed = 0
        self._dropped_stale = 0
        self._dropped_timeout = 0
        self._dropped_error = 0
        self._by_kind: dict[str, int] = {}
        self._cond = asyncio.Condition()

    async def submit(self, job: Job, work: Callable[[], Awaitable[Card]]) -> Card | None:
        """Queue a job and return its Card once it runs, or None on drop/cancel."""
        loop = asyncio.get_running_loop()
        async with self._cond:
            if self._closed:
                return None
            future: asyncio.Future = loop.create_future()
            self._seq += 1
            entry: _Entry = (
                (-int(job.priority.value), job.t_created, self._seq),
                job,
                work,
                future,
            )
            heapq.heappush(self._queue, entry)
            self._queued_count += 1
            if not self._workers_started:
                self._workers_started = True
                for _ in range(self._max_concurrent):
                    self._workers.append(loop.create_task(self._worker_loop()))
            self._cond.notify_all()
        return await future

    async def cancel_kind(self, kind: str) -> int:
        """Drop all queued (not running) jobs of the given kind; returns count cancelled."""
        async with self._cond:
            kept: list[_Entry] = []
            removed = 0
            for entry in self._queue:
                _, j, _w, fut = entry
                if j.kind == kind:
                    removed += 1
                    if not fut.done():
                        fut.set_result(None)
                else:
                    kept.append(entry)
            self._queue = kept
            heapq.heapify(self._queue)
            self._queued_count -= removed
            self._cond.notify_all()
        return removed

    def stats(self) -> dict:
        """Snapshot of pool counters (best-effort, may race mid-update under live load)."""
        return {
            "queued": self._queued_count,
            "running": self._running_count,
            "completed": self._completed,
            "dropped_stale": self._dropped_stale,
            "dropped_timeout": self._dropped_timeout,
            "dropped_error": self._dropped_error,
            "by_kind": dict(self._by_kind),
        }

    async def drain(self) -> None:
        """Wait until the queue is empty and no jobs are running."""
        async with self._cond:
            while self._queue or self._running_count > 0:
                await self._cond.wait()

    async def close(self) -> None:
        """Cancel every queued job, stop workers, and wait for them to finish."""
        async with self._cond:
            self._closed = True
            for entry in self._queue:
                _, _j, _w, fut = entry
                if not fut.done():
                    fut.set_result(None)
            self._queue.clear()
            self._queued_count = 0
            self._cond.notify_all()
        for w in self._workers:
            try:
                await w
            except Exception:
                pass
        self._workers.clear()
        self._workers_started = False

    async def _worker_loop(self) -> None:
        while True:
            async with self._cond:
                while not self._queue and not self._closed:
                    await self._cond.wait()
                if self._closed and not self._queue:
                    return
                entry = heapq.heappop(self._queue)
                _, job, work, future = entry
                self._queued_count -= 1
                self._running_count += 1
            await self._run_job(job, work, future)

    async def _run_job(
        self,
        job: Job,
        work: Callable[[], Awaitable[Card]],
        future: asyncio.Future,
    ) -> None:
        age = time.monotonic() - job.t_created
        threshold = min(job.context_window_s, self._stale_after_s)
        if age > threshold:
            async with self._cond:
                self._running_count -= 1
                self._cond.notify_all()
            self._dropped_stale += 1
            if not future.done():
                future.set_result(None)
            if self._on_drop is not None:
                try:
                    await self._on_drop(job, "stale")
                except Exception:
                    pass
            return

        try:
            card = await asyncio.wait_for(work(), self._job_timeout_s)
        except asyncio.TimeoutError:
            async with self._cond:
                self._running_count -= 1
                self._cond.notify_all()
            self._dropped_timeout += 1
            if not future.done():
                future.set_result(None)
            if self._on_drop is not None:
                try:
                    await self._on_drop(job, "timeout")
                except Exception:
                    pass
            return
        except asyncio.CancelledError:
            async with self._cond:
                self._running_count -= 1
                self._cond.notify_all()
            if not future.done():
                future.set_result(None)
            raise
        except Exception as e:
            async with self._cond:
                self._running_count -= 1
                self._cond.notify_all()
            self._dropped_error += 1
            if not future.done():
                future.set_result(None)
            if self._on_drop is not None:
                try:
                    await self._on_drop(job, f"error:{type(e).__name__}")
                except Exception:
                    pass
            return

        async with self._cond:
            self._running_count -= 1
            self._cond.notify_all()
        self._completed += 1
        self._by_kind[job.kind] = self._by_kind.get(job.kind, 0) + 1
        if self._on_card is not None and card is not None:
            try:
                await self._on_card(card)
            except Exception:
                pass
        if not future.done():
            future.set_result(card)
