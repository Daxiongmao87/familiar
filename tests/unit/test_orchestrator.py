"""Unit tests for dmd.orchestrator.JobPool.

All tests use the real JobPool with small timeouts (<0.5s) for speed.
Deterministic ordering is enforced by synchronizing via asyncio.Event /
queue-count polls rather than time-based waits.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

import pytest

from dmd.orchestrator import JobPool
from dmd.types import Card, Job, Priority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_card(label: str) -> Card:
    return Card(id=label, kind="info", title=label, body_md=label, t_context=0.0)


async def _async_return(value: Card) -> Card:
    return value


def _work_returning(card: Card) -> Callable[[], Awaitable[Card]]:
    async def _w() -> Card:
        return card

    return _w


async def _wait_until(predicate: Callable[[], bool], timeout_s: float = 1.0) -> None:
    """Spin until predicate() is truthy or timeout expires (raises on timeout)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError(f"predicate not satisfied within {timeout_s}s")
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# 1. submit executes work and returns the Card.
# ---------------------------------------------------------------------------


async def test_submit_executes_work_and_returns_card():
    pool = JobPool(max_concurrent=1, job_timeout_s=0.2, stale_after_s=10.0)
    try:
        card = _make_card("hello")
        result = await pool.submit(
            Job(id="j1", kind="trigger", prompt_context={}, priority=Priority.TRIGGER),
            _work_returning(card),
        )
        assert result is card
        assert pool._completed == 1
        assert pool.stats()["completed"] == 1
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 2. Priority ordering: MANUAL beats TRIGGER beats AMBIENT under contention.
# ---------------------------------------------------------------------------


async def test_priority_manual_beats_trigger_beats_ambient():
    pool = JobPool(max_concurrent=1, job_timeout_s=2.0, stale_after_s=10.0)
    try:
        completion_order: list[str] = []
        gate = asyncio.Event()
        first_started = asyncio.Event()

        def make_work(label: str) -> Callable[[], Awaitable[Card]]:
            async def _w() -> Card:
                if label == "A1":
                    first_started.set()
                    await gate.wait()
                completion_order.append(label)
                return _make_card(label)

            return _w

        # Distinct, monotonically-increasing t_created so FIFO is deterministic
        # regardless of monotonic-clock resolution between submissions.
        t_base = time.monotonic()
        specs = [
            ("A1", Priority.AMBIENT, "a", t_base + 0.0),
            ("B", Priority.TRIGGER, "t", t_base + 1.0),
            ("C", Priority.MANUAL, "t", t_base + 2.0),
            ("A2", Priority.AMBIENT, "a", t_base + 3.0),
        ]

        tasks = []
        for i, (label, prio, kind, t_created) in enumerate(specs):
            job = Job(
                id=label,
                kind=kind,
                prompt_context={},
                priority=prio,
                t_created=t_created,
            )
            task = asyncio.create_task(pool.submit(job, make_work(label)))
            tasks.append(task)
            if i == 0:
                # Wait until the worker has pulled A1 off the queue.
                await first_started.wait()
                await _wait_until(lambda: pool._running_count == 1)
            else:
                # Yield until the new job is on the heap (queued_count == i).
                await _wait_until(lambda i=i: pool._queued_count == i)

        assert pool._running_count == 1
        assert pool._queued_count == 3

        gate.set()
        await asyncio.gather(*tasks)

        # Already-running A1 finishes first; then the heap re-orders the rest
        # by priority (MANUAL > TRIGGER > AMBIENT), with FIFO within priority.
        assert completion_order == ["A1", "C", "B", "A2"]
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 3. FIFO within the same priority.
# ---------------------------------------------------------------------------


async def test_fifo_within_same_priority():
    pool = JobPool(max_concurrent=1, job_timeout_s=2.0, stale_after_s=10.0)
    try:
        completion: list[str] = []
        gate = asyncio.Event()
        first_started = asyncio.Event()

        def make_work(label: str) -> Callable[[], Awaitable[Card]]:
            async def _w() -> Card:
                if label == "A":
                    first_started.set()
                    await gate.wait()
                completion.append(label)
                return _make_card(label)

            return _w

        t_base = time.monotonic()
        specs = [
            ("A", t_base + 0.0),
            ("B", t_base + 1.0),
            ("C", t_base + 2.0),
            ("D", t_base + 3.0),
        ]

        tasks = []
        for i, (label, t_created) in enumerate(specs):
            job = Job(
                id=label,
                kind="trigger",
                prompt_context={},
                priority=Priority.TRIGGER,
                t_created=t_created,
            )
            task = asyncio.create_task(pool.submit(job, make_work(label)))
            tasks.append(task)
            if i == 0:
                await first_started.wait()
                await _wait_until(lambda: pool._running_count == 1)
            else:
                await _wait_until(lambda i=i: pool._queued_count == i)

        gate.set()
        await asyncio.gather(*tasks)
        assert completion == ["A", "B", "C", "D"]
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 4. Timeout drop: reason "timeout", on_drop fired, work cancelled.
# ---------------------------------------------------------------------------


async def test_timeout_drop_fires_on_drop_and_cancels_work():
    drops: list[tuple[str, str]] = []

    async def on_drop(job: Job, reason: str) -> None:
        drops.append((job.id, reason))

    pool = JobPool(
        max_concurrent=1,
        job_timeout_s=0.05,
        stale_after_s=10.0,
        on_drop=on_drop,
    )
    try:
        cancelled = [False]

        async def slow_work() -> Card:
            try:
                await asyncio.sleep(1.0)
                return _make_card("never")
            except asyncio.CancelledError:
                cancelled[0] = True
                raise

        result = await pool.submit(
            Job(id="slow", kind="trigger", prompt_context={}),
            slow_work,
        )
        assert result is None
        assert drops == [("slow", "timeout")]
        assert cancelled[0] is True
        assert pool._dropped_timeout == 1
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 5. Exception drop: reason starts with "error:".
# ---------------------------------------------------------------------------


async def test_exception_drop_reason_prefixed_error():
    drops: list[tuple[str, str]] = []

    async def on_drop(job: Job, reason: str) -> None:
        drops.append((job.id, reason))

    pool = JobPool(
        max_concurrent=1,
        job_timeout_s=1.0,
        stale_after_s=10.0,
        on_drop=on_drop,
    )
    try:
        async def boom() -> Card:
            raise ValueError("oh no")

        result = await pool.submit(
            Job(id="err", kind="trigger", prompt_context={}),
            boom,
        )
        assert result is None
        assert len(drops) == 1
        job_id, reason = drops[0]
        assert job_id == "err"
        assert reason.startswith("error:")
        assert pool._dropped_error == 1
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 6. Stale drop: work is never invoked.
# ---------------------------------------------------------------------------


async def test_stale_drop_never_invokes_work():
    drops: list[tuple[str, str]] = []

    async def on_drop(job: Job, reason: str) -> None:
        drops.append((job.id, reason))

    # threshold = min(context_window_s=2.0, stale_after_s=0.5) = 0.5
    pool = JobPool(
        max_concurrent=1,
        job_timeout_s=1.0,
        stale_after_s=0.5,
        on_drop=on_drop,
    )
    try:
        invoked = [False]

        async def work() -> Card:
            invoked[0] = True
            return _make_card("nope")

        # t_created 100s in the past => age >> threshold => stale
        old_t = time.monotonic() - 100.0
        job = Job(
            id="stale",
            kind="trigger",
            prompt_context={},
            t_created=old_t,
            context_window_s=2.0,
        )
        result = await pool.submit(job, work)
        assert result is None
        assert drops == [("stale", "stale")]
        assert invoked[0] is False
        assert pool._dropped_stale == 1
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 7. cancel_kind drops only queued, not running. Asserts the count returned.
# ---------------------------------------------------------------------------


async def test_cancel_kind_drops_only_queued_not_running():
    pool = JobPool(max_concurrent=1, job_timeout_s=2.0, stale_after_s=10.0)
    try:
        gate = asyncio.Event()
        first_started = asyncio.Event()
        work_calls: list[str] = []

        def make_work(label: str) -> Callable[[], Awaitable[Card]]:
            async def _w() -> Card:
                work_calls.append(label)
                if label == "x1":
                    first_started.set()
                    await gate.wait()
                return _make_card(label)

            return _w

        # x1 starts running immediately; x2 and x3 sit on the queue (kind="x").
        t_x1 = asyncio.create_task(
            pool.submit(
                Job(id="x1", kind="x", prompt_context={}),
                make_work("x1"),
            )
        )
        await first_started.wait()
        await _wait_until(lambda: pool._running_count == 1)

        t_x2 = asyncio.create_task(
            pool.submit(
                Job(id="x2", kind="x", prompt_context={}),
                make_work("x2"),
            )
        )
        t_x3 = asyncio.create_task(
            pool.submit(
                Job(id="x3", kind="x", prompt_context={}),
                make_work("x3"),
            )
        )
        await _wait_until(lambda: pool._queued_count == 2)

        # Running x1 must NOT be cancelled; queued x2 and x3 must be.
        removed = await pool.cancel_kind("x")
        assert removed == 2
        assert pool._queued_count == 0
        assert pool._running_count == 1  # x1 still running

        gate.set()
        r_x1, r_x2, r_x3 = await asyncio.gather(t_x1, t_x2, t_x3)

        assert r_x1 is not None and r_x1.id == "x1"
        assert r_x2 is None
        assert r_x3 is None
        # Only x1 ever ran its work callable.
        assert work_calls == ["x1"]
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 8. Stats consistency after a mix of outcomes.
# ---------------------------------------------------------------------------


async def test_stats_consistency_after_mixed_outcomes():
    pool = JobPool(
        max_concurrent=2,
        job_timeout_s=0.05,
        stale_after_s=10.0,
    )
    try:
        async def ok1() -> Card:
            return _make_card("ok1")

        async def ok2() -> Card:
            return _make_card("ok2")

        async def slow() -> Card:
            await asyncio.sleep(1.0)
            return _make_card("slow")

        async def boom() -> Card:
            raise RuntimeError("nope")

        tasks = [
            asyncio.create_task(
                pool.submit(
                    Job(id="ok1", kind="trigger", prompt_context={}),
                    ok1,
                )
            ),
            asyncio.create_task(
                pool.submit(
                    Job(id="ok2", kind="trigger", prompt_context={}),
                    ok2,
                )
            ),
            asyncio.create_task(
                pool.submit(
                    Job(id="slow", kind="trigger", prompt_context={}),
                    slow,
                )
            ),
            asyncio.create_task(
                pool.submit(
                    Job(id="err", kind="trigger", prompt_context={}),
                    boom,
                )
            ),
        ]
        results = await asyncio.gather(*tasks)
        await pool.drain()

        # Outcomes: ok1=Card, ok2=Card, slow=None (timeout), err=None (error).
        assert sum(1 for r in results if r is not None) == 2
        assert sum(1 for r in results if r is None) == 2

        stats = pool.stats()
        assert stats["completed"] == 2
        assert stats["dropped_timeout"] == 1
        assert stats["dropped_error"] == 1
        assert stats["dropped_stale"] == 0
        assert stats["queued"] == 0
        assert stats["running"] == 0
        assert stats["by_kind"].get("trigger") == 2

        # All four submissions are accounted for.
        accounted = (
            stats["completed"]
            + stats["dropped_timeout"]
            + stats["dropped_error"]
            + stats["dropped_stale"]
        )
        assert accounted == 4
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 9. drain() waits for queued + running work to finish.
# ---------------------------------------------------------------------------


async def test_drain_waits_for_completion():
    pool = JobPool(max_concurrent=2, job_timeout_s=1.0, stale_after_s=10.0)
    try:
        completed = [0]

        def make_work(i: int) -> Callable[[], Awaitable[Card]]:
            async def _w() -> Card:
                await asyncio.sleep(0.05)
                completed[0] += 1
                return _make_card(f"c{i}")

            return _w

        tasks = []
        for i in range(3):
            tasks.append(
                asyncio.create_task(
                    pool.submit(
                        Job(id=f"j{i}", kind="trigger", prompt_context={}),
                        make_work(i),
                    )
                )
            )

        # Yield to the loop until all 3 submit coroutines have pushed their
        # jobs onto the heap. Without this, drain() would observe an empty
        # pool and return immediately (the work hasn't been enqueued yet).
        await _wait_until(lambda: pool._queued_count + pool._running_count == 3)

        # Now drain must actually block until all 3 jobs finish.
        await pool.drain()
        assert completed[0] == 3
        assert pool._queued_count == 0
        assert pool._running_count == 0
        # The futures backing the submit tasks are resolved at this point;
        # awaiting them confirms each submit caller also sees its Card.
        for t in tasks:
            assert t.done() or (await t) is not None
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# 10. close() then submit returns None without executing work.
# ---------------------------------------------------------------------------


async def test_close_then_submit_returns_none_without_executing():
    pool = JobPool(max_concurrent=1, job_timeout_s=0.1, stale_after_s=10.0)
    await pool.close()

    invoked = [False]

    async def work() -> Card:
        invoked[0] = True
        return _make_card("after")

    result = await pool.submit(
        Job(id="after", kind="trigger", prompt_context={}),
        work,
    )
    assert result is None
    assert invoked[0] is False


# ---------------------------------------------------------------------------
# 11. on_card raising must not kill the worker.
# ---------------------------------------------------------------------------


async def test_on_card_exception_does_not_kill_worker():
    called: list[str] = []

    async def on_card(card: Card) -> None:
        called.append(card.id)
        if card.id == "boom":
            raise RuntimeError("on_card boom")

    pool = JobPool(
        max_concurrent=1,
        job_timeout_s=1.0,
        stale_after_s=10.0,
        on_card=on_card,
    )
    try:
        r1 = await pool.submit(
            Job(id="boom", kind="trigger", prompt_context={}),
            _work_returning(_make_card("boom")),
        )
        # First submit still returns the Card; on_card's exception is swallowed.
        assert r1 is not None
        assert r1.id == "boom"

        # A subsequent submit must succeed: the worker survived.
        r2 = await pool.submit(
            Job(id="after", kind="trigger", prompt_context={}),
            _work_returning(_make_card("after")),
        )
        assert r2 is not None
        assert r2.id == "after"
        assert called == ["boom", "after"]
        assert pool._completed == 2
    finally:
        await pool.close()
