"""SPEC §14 latency: intake must never block on STT (owner-verified defect,
2026-09-05 audit).

``consume_source``'s chunk loop only feeds the streaming adapter; the
server's VAD endpoints segments and finals arrive asynchronously. These
tests drive a fake in-process SimulStreaming-protocol TCP server and prove:

  * the feed loop drains a paced source promptly even when the server is
    slow to answer (regression: inline STT stalled intake per utterance);
  * a slow speaker never delays another speaker's utterances;
  * per-user finals publish in speech order;
  * a dead server counts audio unfed without stalling or raising
    (streaming is the only path — there is no fallback);
  * the timestamped ``stt_latency`` measurement carries real durations.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from dmd.config import load_config_dict
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
        for user_id, pcm in self._script:
            self.emit(self._chunk(user_id, pcm, time.monotonic()))
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


class ScriptedServer:
    """Fake SimulStreaming server: per-connection (min_bytes, delay, lines).

    ``scripts[i]`` drives connection ``i`` (later connections reuse the
    last script): once ``min_bytes`` of PCM arrive, it waits ``delay_s``,
    then sends each line dict as newline JSON. After answering it
    half-closes (write EOF) so the client's drain path terminates
    immediately, while still reading client audio until full EOF.
    """

    def __init__(self, scripts: list[tuple[int, float, list[dict]]]) -> None:
        self._scripts = scripts
        self.connections = 0

    async def _answer(self, writer: asyncio.StreamWriter, lines: list[dict]) -> None:
        for line in lines:
            writer.write((json.dumps(line) + "\n").encode())
        await writer.drain()
        try:
            writer.write_eof()
        except (OSError, RuntimeError, NotImplementedError):
            pass

    async def handler(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        idx = self.connections
        self.connections += 1
        min_bytes, delay_s, lines = self._scripts[min(idx, len(self._scripts) - 1)]
        received = 0
        fired = False
        while True:
            data = await reader.read(65536)
            if not data:
                break
            received += len(data)
            if not fired and received >= min_bytes:
                fired = True
                await asyncio.sleep(delay_s)
                await self._answer(writer, lines)
        writer.close()


async def _start(
    scripts: list[tuple[int, float, list[dict]]],
) -> tuple[ScriptedServer, asyncio.Server, int]:
    fake = ScriptedServer(scripts)
    server = await asyncio.start_server(fake.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return fake, server, port


class NoPool:
    async def submit(self, job: Any, work: Any) -> None:
        return None

    async def drain(self) -> None:
        return None


def _final(text: str) -> dict:
    return {"text": text, "start": 0.0, "end": 0.0, "is_final": True}


def _engine(port: int, tmp_path: Any, on_event: Any) -> SessionEngine:
    cfg = load_config_dict(
        {
            "project": {"path": str(tmp_path)},
            "models": {
                "synthesis": {"base_url": "http://fake.invalid", "model_id": "f"},
                "stt": {"stream_host": "127.0.0.1", "stream_port": port},
            },
        }
    )
    return SessionEngine(
        cfg=cfg,
        store=None,
        gw=object(),  # STT never touches the gateway (streaming-only)
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
    """Five paced chunks with a 0.4 s-slow server: intake finishes long
    before the finals land."""
    events: list[dict[str, Any]] = []
    fake, server, port = await _start(
        [
            (10240, 0.4, [_final("text dm")]),
            (10240, 0.4, [_final("text alice")]),
        ]
    )
    try:
        engine = _engine(port, tmp_path, events.append)
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

        # Intake read every chunk (minus the final None) well before either
        # 0.4 s server answer finished.
        feed_span = source.pop_times[-1] - source.pop_times[0]
        assert feed_span < 0.3, f"intake feed blocked for {feed_span:.2f}s"

        # Both finals landed (mid-speech partials are pinned in
        # test_streaming_stt, where the pump is alive to see them — here
        # the slow answers land in the end-of-source drain by design).
        transcripts = [e for e in events if e["type"] == "transcript"]
        assert [(t["user_id"], t["text"]) for t in transcripts] == [
            ("dm", "text dm"),
            ("alice", "text alice"),
        ]

        # Two stt_latency measurements with real measured server delay.
        latencies = [e for e in events if e["type"] == "stt_latency"]
        assert len(latencies) == 2
        for ev in latencies:
            assert ev["path"] == "streaming"
            assert ev["stt_ms"] >= 300  # real measured 0.4 s server think
            assert ev["queue_wait_ms"] == 0.0  # no queue exists anymore

        # Total wall ≈ server think + feed, NOT serialized per utterance.
        assert total < 1.2, f"total {total:.2f}s indicates serialized STT"

        stats = engine.intake_stats()
        assert stats["max_work_ms"] < 50, f"chunk handler blocked: {stats}"
        assert stats["chunks"] == 6
        assert stats["unfed"] == 0
        await engine.aclose()
    finally:
        server.close()
        await server.wait_closed()


async def test_slow_speaker_never_delays_another_speaker(tmp_path: Any) -> None:
    """User A's 0.6 s server answer must not hold user B's fast final."""
    events: list[dict[str, Any]] = []
    fake, server, port = await _start(
        [
            (6400, 0.6, [_final("slow line")]),  # first connection: a
            (6400, 0.05, [_final("fast line")]),  # second connection: b
        ]
    )
    try:
        engine = _engine(port, tmp_path, events.append)
        # Two speech chunks per user trip a mid-intake client flush, so
        # both server answers are in flight while intake continues.
        source = TimedSource(
            [
                ("a", SPEECH),
                ("a", SPEECH),
                ("a", SILENCE),
                ("b", SPEECH),
                ("b", SPEECH),
                ("b", SILENCE),
            ],
            pace_s=0.005,
        )

        await _drain(engine, source)

        transcripts = [e for e in events if e["type"] == "transcript"]
        assert len(transcripts) == 2
        # Completion order is the wall-clock of the stt_latency events.
        latencies = [e for e in events if e["type"] == "stt_latency"]
        order = [ev["user_id"] for ev in sorted(latencies, key=lambda e: e["t"])]
        assert order == ["b", "a"], (
            "fast speaker B was held behind slow speaker A's transcription"
        )
        await engine.aclose()
    finally:
        server.close()
        await server.wait_closed()


async def test_per_user_finals_publish_in_speech_order(tmp_path: Any) -> None:
    """Two finals on one connection publish in the order the server sent."""
    events: list[dict[str, Any]] = []
    received: list[int] = []

    async def _two_finals(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        total = 0
        sent = 0
        eofed = False
        while True:
            data = await reader.read(65536)
            if not data:
                break
            total += len(data)
            # one final per 10240-byte client flush
            while sent < 2 and total >= 10240 * (sent + 1):
                writer.write((json.dumps(_final(f"text {sent}")) + "\n").encode())
                await writer.drain()
                sent += 1
            if sent == 2 and not eofed:
                eofed = True
                try:
                    writer.write_eof()
                except (OSError, RuntimeError, NotImplementedError):
                    pass
        received.append(total)
        writer.close()

    server = await asyncio.start_server(_two_finals, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        engine = _engine(port, tmp_path, events.append)
        source = TimedSource(
            [
                ("solo", SPEECH),
                ("solo", SPEECH),
                ("solo", SPEECH),
                ("solo", SPEECH),
            ],
            pace_s=0.005,
        )

        await _drain(engine, source)
        texts = [e["text"] for e in events if e["type"] == "transcript"]
        assert texts == ["text 0", "text 1"]
        await engine.aclose()
    finally:
        server.close()
        await server.wait_closed()


async def test_dead_server_counts_unfed_without_stall(tmp_path: Any) -> None:
    """Nothing listening: audio is counted unfed, intake stays fast, and no
    transcript or exception escapes (the health monitor owns the banner)."""
    events: list[dict[str, Any]] = []
    engine = _engine(1, tmp_path, events.append)  # port 1: refused
    source = TimedSource(
        [("dm", SPEECH), ("dm", SPEECH), ("alice", SPEECH)], pace_s=0.005
    )

    t0 = time.monotonic()
    await _drain(engine, source)
    total = time.monotonic() - t0

    assert total < 1.0, f"dead server stalled intake for {total:.2f}s"
    assert not [e for e in events if e["type"] == "transcript"]
    stats = engine.intake_stats()
    assert stats["chunks"] == 3
    assert stats["unfed"] == 3
    await engine.aclose()


async def test_pool_drain_still_runs(tmp_path: Any) -> None:
    """Regression: consume_source must drain the job pool at end of source."""
    drained: list[bool] = []

    class _DrainPool(NoPool):
        async def drain(self) -> None:
            drained.append(True)

    events: list[dict[str, Any]] = []
    fake, server, port = await _start([(1, 0.0, [_final("hi")])])
    try:
        engine = _engine(port, tmp_path, events.append)
        engine.pool = _DrainPool()
        source = TimedSource([("dm", SPEECH), ("dm", SILENCE)], pace_s=0.005)
        await _drain(engine, source)
        assert drained == [True]
        await engine.aclose()
    finally:
        server.close()
        await server.wait_closed()
