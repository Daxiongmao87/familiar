"""STT-health monitor (SPEC §12/§15 degraded-mode UX).

Regression suite for the 2026-09-05 live defect: the monitor called the
Gateway's coroutine function synchronously, unpacked the coroutine object
instead of its result, and the probe task died at startup — the UI banner
silently reported "STT unavailable" while whisperx was answering 200.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from dmd.config import load_config_dict
from dmd.gateway import Gateway
from dmd.stt_health import SttHealthMonitor


class _Bus:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish_sync(self, event: dict) -> None:
        self.events.append(event)


class _GwStub:
    def __init__(self, *results: tuple[bool, str]) -> None:
        self._results = list(results)
        self.calls = 0

    async def stt_health(self) -> tuple[bool, str]:
        self.calls += 1
        r = self._results[min(self.calls - 1, len(self._results) - 1)]
        if isinstance(r, Exception):
            raise r
        return r


async def test_probe_awaits_the_gateway_coroutine_and_reports_healthy() -> None:
    bus = _Bus()
    gw = _GwStub((True, "http 200"))
    mon = SttHealthMonitor(gw, bus=bus)
    await mon._probe_once()
    assert mon.snapshot() == {"healthy": True, "detail": "http 200"}
    assert bus.events == [{"type": "stt_health", "healthy": True, "detail": "http 200"}]


async def test_snapshot_is_unknown_before_first_probe() -> None:
    """Not-yet-probed must not be reported as dead (false banner)."""
    mon = SttHealthMonitor(_GwStub((True, "ok")))
    assert mon.snapshot()["healthy"] is None


async def test_unreachable_then_recovery_emits_transitions() -> None:
    bus = _Bus()
    gw = _GwStub((False, "ConnectError: refused"), (True, "http 200"))
    mon = SttHealthMonitor(gw, bus=bus)
    await mon._probe_once()
    await mon._probe_once()
    assert [e["healthy"] for e in bus.events] == [False, True]
    assert mon.snapshot()["healthy"] is True


async def test_probe_exception_is_captured_not_raised() -> None:
    gw = _GwStub(RuntimeError("endpoint exploded"))
    mon = SttHealthMonitor(gw)
    await mon._probe_once()  # must not raise
    snap = mon.snapshot()
    assert snap["healthy"] is False and "RuntimeError" in snap["detail"]


async def test_run_loop_survives_a_failing_probe() -> None:
    """The old bug killed the task on the first probe; the loop must not die."""
    import asyncio

    class _ExplodingGw:
        def __init__(self) -> None:
            self.calls = 0

        async def stt_health(self) -> tuple[bool, str]:
            self.calls += 1
            raise ValueError("boom")

    gw = _ExplodingGw()
    mon = SttHealthMonitor(gw, interval_s=0.01)
    await mon.start()
    await asyncio.sleep(0.35)  # interval clamps to 0.1 s floor -> ~3+ probes
    assert gw.calls >= 3, "probe loop died on the first failing probe"
    await mon.stop()
    assert mon.snapshot()["healthy"] is False


async def test_contract_with_the_real_gateway() -> None:
    """Drive one probe against the real Gateway + MockTransport: the monitor
    and the gateway agree on the (bool, str) coroutine contract."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    cfg = load_config_dict(
        {
            "models": {
                "synthesis": {"base_url": "http://test", "model_id": "m"},
                "stt": {"base_url": "http://test", "dialect": "whisperx"},
            }
        }
    )
    gw = Gateway(cfg)
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    mon = SttHealthMonitor(gw)
    await mon._probe_once()
    assert mon.snapshot()["healthy"] is True
    await gw.aclose()
