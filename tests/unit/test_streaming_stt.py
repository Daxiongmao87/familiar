"""Streaming STT wiring: finals dispatch through the shared tail, partials
publish as ephemeral events, and batch twins are deduped (2026-09-05).

Requires the SimulStreaming server on 127.0.0.1:43007 (see
~/streaming-stt) — skipped automatically when it is not listening.
"""

from __future__ import annotations

import asyncio
import json
import wave

import pytest

from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.streaming_stt import probe_server
from dmd.types import PcmChunk

SR = 16000
WAV = "/tmp/opencode/live_utt_16k.wav"


class FakeGw:
    """Batch stub: never transcribes (dedup must drop the twin anyway)."""

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, audio_bytes: bytes, **kw) -> str:
        self.calls += 1
        await asyncio.sleep(0.05)
        return "I search the goblin corpse for loot"


class NoPool:
    async def submit(self, job, work) -> None:
        return None

    async def drain(self) -> None:
        return None


def _engine(tmp_path) -> SessionEngine:
    cfg = load_config_dict(
        {
            "project": {"path": str(tmp_path)},
            "models": {
                "synthesis": {"base_url": "http://fake.invalid", "model_id": "f"},
                "stt": {
                    "base_url": "http://127.0.0.1:8123",
                    "dialect": "streaming",
                    "stream_host": "127.0.0.1",
                    "stream_port": 43007,
                },
            },
            "stt_pipeline": {"silence_ms": 500, "min_utterance_ms": 40},
        }
    )
    return SessionEngine(
        cfg=cfg,
        store=None,
        gw=FakeGw(),
        entries=[],
        embedder=None,
        pool=NoPool(),
        on_event=lambda e: events.append(e),
        project_path=str(tmp_path),
    )


events: list[dict] = []


class _DetectStub:
    """detect_trigger is imported into pipeline; fast-lane classification of
    real whisper text is irrelevant here — treat everything as a trigger."""


@pytest.mark.asyncio
async def test_streaming_final_dispatches_once(tmp_path, monkeypatch):
    if not await probe_server("127.0.0.1", 43007):
        pytest.skip("SimulStreaming server not listening on 43007")
    events.clear()

    async def _never_trigger(gw, text):
        return False, "none"

    monkeypatch.setattr("dmd.pipeline.detect_trigger", _never_trigger)
    engine = _engine(tmp_path)
    with wave.open(WAV, "rb") as w:
        frames = w.readframes(w.getnframes())
    step = SR // 5 * 2  # 200 ms chunks at real pace
    for off in range(0, len(frames), step):
        await engine._stt_adapter.feed("u1", frames[off : off + step])
        await asyncio.sleep(0.2)
    # trailing silence so the server VAD endpoints
    for _ in range(15):
        await engine._stt_adapter.feed("u1", b"\x00" * step)
        await asyncio.sleep(0.2)
    # wait for final to arrive
    for _ in range(50):
        await asyncio.sleep(0.1)
        if any(e["type"] == "transcript" for e in events):
            break
    finals = [e for e in events if e["type"] == "transcript"]
    partials = [e for e in events if e["type"] == "transcript_partial"]
    await engine.aclose()
    assert finals, f"no streaming final; partials={len(partials)}"
    assert "goblin" in finals[0]["text"].lower()
    assert partials, "expected mid-speech partials"
    # exactly ONE transcript for the utterance (no batch twin fired)
    assert len(finals) == 1


@pytest.mark.asyncio
async def test_batch_twin_is_deduped(tmp_path, monkeypatch):
    """A late batch result covering the same window must not re-publish."""
    events.clear()

    async def _never_trigger(gw, text):
        return False, "none"

    monkeypatch.setattr("dmd.pipeline.detect_trigger", _never_trigger)
    engine = _engine(tmp_path)
    import time as _t

    now = _t.monotonic()
    from dmd.attribution import AttributedSegment

    # streaming final for window [now-4, now-1]
    await engine._on_stream_final("u1", "i search the corpse", now - 4, now - 1)
    # batch twin for overlapping window arrives late
    twin = AttributedSegment(
        user_id="u1", text="i search the corpse", t_start=now - 4, t_end=now - 1
    )
    await engine._publish_and_dispatch([twin])
    # a genuinely different later utterance must NOT be dropped
    other = AttributedSegment(
        user_id="u1", text="roll a d20 for me", t_start=now + 30, t_end=now + 32
    )
    await engine._publish_and_dispatch([other])
    texts = [e["text"] for e in events if e["type"] == "transcript"]
    await engine.aclose()
    assert texts.count("i search the corpse") == 1
    assert "roll a d20 for me" in texts
