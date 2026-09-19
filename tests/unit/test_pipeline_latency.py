"""SPEC §14 latency: intake must never block on STT (owner-verified defect,
2026-09-05 audit).

``consume_source``'s chunk loop only feeds the VAD and enqueues utterances;
transcription runs on the per-user worker queue. These tests prove:

  * the feed loop drains a paced source promptly even when every STT call is
    slow (regression: the inline ``await _dispatch_utterance`` stalled intake
    for the full transcription duration per utterance);
  * a slow speaker never delays another speaker's utterances;
  * per-user ordering survives the queue;
  * the timestamped ``stt_latency`` measurement the session log emits carries
    real enqueue/queue-wait/STT durations.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from dmd.config import load_config_dict
from dmd.orchestrator import JobPool
from dmd.pipeline import SessionEngine
from dmd.sources.base import AudioSource

SR = 16000
SPEECH = b"\x40\x0f\x40\x0f" * (SR // 10)  # 100 ms of int16 amplitude 3904
SILENCE = b"\x00\x00" * (SR // 10)


class TimedSource(AudioSource):
    """Emits a scripted (user, pcm) sequence with pacing; records pop times."""

    def __init__(self, script: list[tuple[str, bytes]], pace_s: float = 0.01) -> None:
        super().__init__()
        self._script = list(script)
        self._pace_s = pace_s
        self.pop_times: list[float] = []

    async def run(self) -> None:
        t0 = time.monotonic()
        for user_id, pcm in self._script:
            self.emit(
                self._chunk(user_id, pcm, time.monotonic())
            )
            await asyncio.sleep(self._pace_s)
        self.emit(None)

    @staticmethod
    def _chunk(user_id: str, pcm: bytes, t: float) -> Any:
        from dmd.types import PcmChunk

        return PcmChunk(user_id=user_id, samples=pcm, sample_rate=SR, t_mono=t)

    async def __anext__(self):
        chunk = await super().__anext__()
        self.pop_times.append(time.monotonic())
        return chunk


class SlowSttGateway:
    """STT endpoint stub whose transcribe sleeps a scripted duration per call."""

    def __init__(self, sleeps: list[float]) -> None:
        self._sleeps = list(sleeps)
        self._texts = [f"text {i}" for i in range(len(sleeps))]
        self.call_times: list[float] = []

    async def transcribe(self, audio_bytes: bytes, **kw: Any) -> str:
        i = len(self.call_times)
        self.call_times.append(time.monotonic())
        await asyncio.sleep(self._sleeps[i])
        return self._texts[i]


class NoPool:
    async def submit(self, job: Any, work: Any) -> None:
        return None

    async def drain(self) -> None:
        return None


def _engine(tmp_path: Any, gw: Any, on_event: Any) -> SessionEngine:
    cfg = load_config_dict(
        {
            "project": {"path": str(tmp_path)},
            "models": {
                "synthesis": {"base_url": "http://fake.invalid", "model_id": "f"},
                "stt": {"base_url": "http://fake.invalid"},
            },
            "stt_pipeline": {"silence_ms": 60, "min_utterance_ms": 40},
        }
    )
    return SessionEngine(
        cfg=cfg,
        store=None,
        gw=gw,
        entries=[],
        embedder=None,
        pool=NoPool(),
        on_event=on_event,
        project_path=str(tmp_path),
    )


async def _drain(engine: SessionEngine, source: TimedSource) -> None:
    """Run the source producer alongside consume_source (as the server does)."""
    producer = asyncio.create_task(source.run())
    try:
        await engine.consume_source(source)
    finally:
        await producer


async def test_intake_does_not_block_on_stt(tmp_path: Any) -> None:
    """Five paced chunks (~0.05 s of feed) with 0.4 s STT per utterance: intake
    finishes long before the transcriptions, and total wall time is not the
    sum of STT durations serialized into the feed loop."""
    events: list[dict[str, Any]] = []
    gw = SlowSttGateway([0.4, 0.4])
    engine = _engine(tmp_path, gw, events.append)
    source = TimedSource(
        [
            ("dm", SPEECH),
            ("dm", SPEECH),
            ("dm", SILENCE),
            ("alice", SPEECH),
            ("alice", SPEECH),
            ("alice", SILENCE),
        ]
    )

    t0 = time.monotonic()
    await _drain(engine, source)
    total = time.monotonic() - t0

    # Intake read every chunk (minus the final None) well before either 0.4 s
    # transcription finished: with the inline await the feed loop alone would
    # have burned >= 0.8 s of STT time serially.
    feed_span = source.pop_times[-1] - source.pop_times[0]
    assert feed_span < 0.3, f"intake feed blocked for {feed_span:.2f}s"

    # Both transcripts landed; two stt_latency measurements exist with sane
    # per-stage timestamps.
    transcripts = [e for e in events if e["type"] == "transcript"]
    assert [t["user_id"] for t in transcripts] == ["dm", "alice"]
    latencies = [e for e in events if e["type"] == "stt_latency"]
    assert len(latencies) == 2
    for ev in latencies:
        assert ev["stt_ms"] >= 400  # real measured sleep
        assert ev["queue_wait_ms"] >= 0

    # Total wall ≈ max(0.4, 0.4) + feed, NOT 0.8 + feed (workers are per-user
    # and run concurrently).
    assert total < 0.75, f"total {total:.2f}s indicates serialized STT"

    stats = engine.intake_stats()
    assert stats["max_work_ms"] < 50, f"chunk handler blocked: {stats}"
    assert stats["chunks"] == 6  # 6 data chunks; the None sentinel exits the loop


async def test_slow_speaker_never_delays_another_speaker(tmp_path: Any) -> None:
    """User A's 0.6 s STT must not hold user B's fast utterance behind it."""
    events: list[dict[str, Any]] = []
    gw = SlowSttGateway([0.6, 0.05])  # A first, then B
    engine = _engine(tmp_path, gw, events.append)
    source = TimedSource(
        [
            ("a", SPEECH),
            ("a", SILENCE),
            ("b", SPEECH),
            ("b", SILENCE),
        ],
        pace_s=0.005,
    )

    await _drain(engine, source)

    transcripts = [e for e in events if e["type"] == "transcript"]
    assert len(transcripts) == 2
    # Completion order is the wall-clock of the stt_latency events; the
    # transcript event's own t is the speech-end time, not the publish time.
    latencies = [e for e in events if e["type"] == "stt_latency"]
    order = [ev["user_id"] for ev in sorted(latencies, key=lambda e: e["t"])]
    assert order == ["b", "a"], (
        "fast speaker B was held behind slow speaker A's transcription"
    )


async def test_per_user_ordering_survives_the_queue(tmp_path: Any) -> None:
    """Two back-to-back utterances from one user publish in speech order."""
    events: list[dict[str, Any]] = []
    gw = SlowSttGateway([0.05, 0.01])  # first is slower; order must still hold
    engine = _engine(tmp_path, gw, events.append)
    source = TimedSource(
        [
            ("solo", SPEECH),
            ("solo", SILENCE),
            ("solo", SPEECH),
            ("solo", SILENCE),
        ],
        pace_s=0.005,
    )

    await _drain(engine, source)
    texts = [e["text"] for e in events if e["type"] == "transcript"]
    assert texts == ["text 0", "text 1"]


async def test_pool_drain_still_runs(tmp_path: Any) -> None:
    """Regression: consume_source must drain the job pool even though STT moved
    off the intake loop."""
    drained: list[bool] = []

    class _DrainPool(NoPool):
        async def drain(self) -> None:
            drained.append(True)

    events: list[dict[str, Any]] = []
    gw = SlowSttGateway([0.01])
    cfg_engine = _engine(tmp_path, gw, events.append)
    cfg_engine.pool = _DrainPool()
    source = TimedSource([("dm", SPEECH), ("dm", SILENCE)], pace_s=0.005)
    await _drain(cfg_engine, source)
    assert drained == [True]
