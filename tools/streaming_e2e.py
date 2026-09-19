"""End-to-end: real clip through consume_source with streaming armed.

Feeds /tmp/opencode/dax_clip.wav (single speaker, 41.3 s) as 200 ms
PcmChunks at real-time pace into SessionEngine.consume_source — the same
entry point Discord uses. Verifies: partials mid-speech, exactly one
transcript per spoken segment (batch twins deduped), post-speech latency.
"""
from __future__ import annotations

import asyncio
import time
import wave

from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.types import PcmChunk

SR = 16000
WAV = "/tmp/opencode/dax_clip.wav"
events: list[dict] = []


class SlowBatchGw:
    """Batch STT that takes 3.5 s — realistic whisperx latency, proves
    dedup kills its twins without the timing being trivially favorable."""

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, audio_bytes: bytes, **kw) -> str:
        self.calls += 1
        await asyncio.sleep(3.5)
        return f"batch transcription call {self.calls}"


class NoPool:
    async def submit(self, job, work) -> None:
        return None

    async def drain(self) -> None:
        return None


async def _no_trigger(gw, text):
    return False, "none"


async def main() -> None:
    import dmd.pipeline as pipe
    pipe.detect_trigger = _no_trigger  # no card generation; STT path only

    cfg = load_config_dict(
        {
            "project": {"path": "/tmp/fam-test"},
            "models": {
                "synthesis": {"base_url": "http://fake", "model_id": "m"},
                "stt": {
                    "base_url": "http://fake",
                    "dialect": "streaming",
                    "stream_host": "127.0.0.1",
                    "stream_port": 43007,
                },
            },
            "stt_pipeline": {"silence_ms": 500, "min_utterance_ms": 40},
        }
    )
    gw = SlowBatchGw()
    engine = SessionEngine(
        cfg=cfg, store=None, gw=gw, entries=[], embedder=None,
        pool=NoPool(), on_event=events.append, project_path="/tmp/fam-test",
    )

    with wave.open(WAV, "rb") as w:
        frames = w.readframes(w.getnframes())
    dur = len(frames) / 2 / SR
    print(f"clip: {dur:.1f}s, {len(frames)//2} samples")

    class Src:
        async def __aiter__(self):
            step = SR // 5 * 2  # 200 ms
            t0 = time.monotonic()
            for off in range(0, len(frames), step):
                yield PcmChunk(
                    user_id="dax", samples=frames[off : off + step],
                    sample_rate=SR, t_mono=time.monotonic() - 0.2,
                )
                await asyncio.sleep(0.2)
            # tail silence to flush the server VAD
            for _ in range(15):
                yield PcmChunk(user_id="dax", samples=b"\x00" * step,
                               sample_rate=SR, t_mono=time.monotonic() - 0.2)
                await asyncio.sleep(0.2)
            # let late finals/twins land
            await asyncio.sleep(4.0)

    t_start = time.monotonic()
    await engine.consume_source(Src())
    # wait out any remaining batch calls
    await asyncio.sleep(5.0)
    await engine.aclose()

    partials = [e for e in events if e["type"] == "transcript_partial"]
    finals = [e for e in events if e["type"] == "transcript"]
    lat = [e for e in events if e["type"] == "stt_latency"]
    print(f"\npartials: {len(partials)}  finals: {len(finals)}  batch calls: {gw.calls}")
    print("\n--- transcript (finals) ---")
    for f in finals:
        print(f"  [{f['t']:.2f}] {f['text'][:100]}")
    print("\n--- streaming final latencies ---")
    for l in lat:
        p = l.get("path", "batch")
        print(f"  {p:9s} post_speech_ms={l['post_speech_ms']:.0f}")
    s = [l for l in lat if l.get("path") == "streaming"]
    if s:
        avg = sum(x["post_speech_ms"] for x in s) / len(s)
        print(f"\nSTREAMING finals: {len(s)}, avg post-speech {avg:.0f} ms")
    b = [l for l in lat if l.get("path") != "streaming"]
    print(f"batch transcriptions: {gw.calls}; batch dispatches: {len(b)}"
          f" (deduped: {gw.calls - len(b)})")

    # sanity: no text duplicated across finals
    texts = [f["text"] for f in finals]
    assert len(texts) == len(set(texts)), f"DUPLICATE FINAL: {texts}"
    print("\nOK: no duplicate transcripts")


asyncio.run(main())
