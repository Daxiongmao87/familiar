"""Streaming STT: adapter protocol, reconnect backoff, and final dispatch.

All tests drive a fake in-process TCP server speaking the SimulStreaming
protocol (raw s16le PCM in, newline-JSON partials/finals out) — no live
server required.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.streaming_stt import StreamingSttAdapter, probe_server

SR = 16000


def _final(text: str) -> dict:
    return {"text": text, "start": 0.0, "end": 0.5, "is_final": True}


def _partial(text: str) -> dict:
    return {"text": text, "start": 0.0, "end": 0.2, "is_final": False}


class NoPool:
    async def submit(self, job: Any, work: Any) -> None:
        return None

    async def drain(self) -> None:
        return None


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
        gw=object(),
        entries=[],
        embedder=None,
        pool=NoPool(),
        on_event=on_event,
        project_path=str(tmp_path),
    )


async def test_streaming_final_dispatches_through_engine(tmp_path: Any) -> None:
    """Partial publishes an ephemeral event; the final dispatches exactly
    one transcript through attribution, publication, and OpenJEV."""
    events: list[dict] = []

    async def _server(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        total = 0
        fired = False
        while True:
            data = await reader.read(65536)
            if not data:
                break
            total += len(data)
            if not fired and total >= 10240:
                fired = True
                for line in [_partial("i search"), _final(" the goblin corpse")]:
                    writer.write((json.dumps(line) + "\n").encode())
                await writer.drain()
                try:
                    writer.write_eof()
                except (OSError, RuntimeError, NotImplementedError):
                    pass
        writer.close()

    server = await asyncio.start_server(_server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        engine = _engine(port, tmp_path, events.append)
        chunk = b"\x40\x0f" * 3200  # 6400 bytes; two chunks trip a flush
        await engine._stt_adapter.feed("u1", chunk)
        await engine._stt_adapter.feed("u1", chunk)
        for _ in range(100):
            await asyncio.sleep(0.05)
            if any(e["type"] == "transcript" for e in events):
                break
        finals = [e for e in events if e["type"] == "transcript"]
        partials = [e for e in events if e["type"] == "transcript_partial"]
        await engine.aclose()
        assert partials, "expected a mid-speech partial"
        assert len(finals) == 1, f"expected exactly one final, got {len(finals)}"
        # partial + final increments accumulate into the whole utterance
        assert finals[0]["text"] == "i search the goblin corpse"
        assert finals[0]["user_id"] == "u1"
    finally:
        server.close()
        await server.wait_closed()


async def test_failed_session_reopens_with_backoff() -> None:
    """A dead server costs one connect attempt per RETRY_S per user — not
    one per chunk — and a recovered server is picked up automatically."""
    adapter = StreamingSttAdapter(host="127.0.0.1", port=1)  # refused
    adapter.RETRY_S = 0.05
    assert await adapter.feed("u", b"\x00" * 100) is False
    # Immediate retry is throttled (no new attempt, still False).
    assert await adapter.feed("u", b"\x00" * 100) is False

    # A server appears mid-outage: the throttled feed still refuses, but
    # the next feed after the window reopens and holds the audio.
    attempts = 0

    async def _count(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        nonlocal attempts
        attempts += 1
        while await reader.read(65536):
            pass
        writer.close()

    server = await asyncio.start_server(_count, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        adapter.port = port
        assert await adapter.feed("u", b"\x00" * 100) is False
        assert attempts == 0, "throttled feed attempted a connect"
        await asyncio.sleep(0.08)
        assert await adapter.feed("u", b"\x00" * 100) is True
        assert attempts == 1
        await adapter.close_all()
    finally:
        server.close()
        await server.wait_closed()


async def test_feed_never_raises_and_close_is_idempotent() -> None:
    adapter = StreamingSttAdapter(host="127.0.0.1", port=1)
    assert await adapter.feed("u", b"\x00" * 100) is False
    assert adapter.has_session("u") is False
    await adapter.close_user("u")  # unknown user: no-op
    await adapter.close_all()
    await adapter.close_all()


async def test_probe_server_true_and_false() -> None:
    async def _noop(reader: object, writer: object) -> None:
        # Must close: wait_closed() below blocks until every accepted
        # connection is done.
        w = writer  # type: ignore[union-attr]
        w.close()
        try:
            await w.wait_closed()
        except OSError:
            pass

    server = await asyncio.start_server(_noop, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        assert await probe_server("127.0.0.1", port) is True
    finally:
        server.close()
        await server.wait_closed()
    assert await probe_server("127.0.0.1", port) is False
