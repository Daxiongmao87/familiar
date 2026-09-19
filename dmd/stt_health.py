"""Background STT-endpoint health monitor.

STT transcription is the live path's only audio input. When the endpoint is
dead the UI must make that obvious within seconds, not silently produce no
transcript. This module probes the configured STT endpoint on a cadence,
caches the last health result, publishes state transitions to the event bus
so the live UI can surface a degraded banner, and logs a connect failure once
(with retry backoff) instead of staying silent.

Run as a background task; ``snapshot()`` is the read-only view the
`/api/status` endpoint exposes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .gateway import Gateway
from .server import EventBus


class SttHealthMonitor:
    """Periodically probes the STT endpoint and tracks its health."""

    def __init__(
        self,
        gateway: Gateway | None,
        bus: EventBus | None = None,
        interval_s: float = 5.0,
        backoff_s: float = 60.0,
    ) -> None:
        self._gw = gateway
        self._bus = bus
        self._interval = max(0.1, float(interval_s))
        self._backoff = max(1.0, float(backoff_s))
        self._healthy: bool | None = None  # None = not yet probed
        self._detail = ""
        self._last_log: float | None = None  # monotonic; rate-limit logs
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def run(self) -> None:
        """Probe immediately, then on cadence until stopped."""
        import logging

        while not self._stop.is_set():
            try:
                await self._probe_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the probe loop must never die
                logging.getLogger(__name__).warning(
                    "stt-health probe failed", exc_info=True
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    async def _probe_once(self) -> None:
        healthy, detail = await self._probe()
        prev = self._healthy
        self._healthy = healthy
        self._detail = detail
        if self._bus is not None and prev != healthy:
            self._bus.publish_sync(
                {"type": "stt_health", "healthy": bool(healthy), "detail": detail}
            )
        if not healthy:
            # Log the failure once, then back off until _backoff_s elapses so a
            # persistently-dead endpoint doesn't spam the logs every probe.
            # (Order matters: the first failure has _last_log None.)
            if self._last_log is None or (
                time.monotonic() - self._last_log
            ) >= self._backoff:
                import logging

                logging.getLogger(__name__).warning(
                    "STT endpoint unreachable: %s", detail
                )
                self._last_log = time.monotonic()

    async def _probe(self) -> tuple[bool, str]:
        if self._gw is None:
            return False, "gateway not initialized"
        try:
            # Gateway.stt_health is a coroutine function — it must be awaited
            # (the sync call unpacked the coroutine object itself and killed
            # this task at startup: 2026-09-05 live defect).
            return await self._gw.stt_health()
        except Exception as exc:  # noqa: BLE001 - already reported via status
            return False, f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> dict[str, Any]:
        """Read-only health view exposed by `/api/status`.

        ``healthy`` is None until the first probe lands (unknown, not dead),
        False only on a real probe failure.
        """
        return {"healthy": self._healthy, "detail": self._detail}

    async def start(self) -> None:
        """Start the background probe loop."""
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """Stop the background probe loop."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
