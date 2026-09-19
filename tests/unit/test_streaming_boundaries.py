"""Streaming boundaries must preserve already emitted words."""

import asyncio
import json

from dmd.streaming_stt import StreamingSttAdapter


async def test_empty_final_commits_partial_before_connection_closes():
    """A VAD final with no new words still commits the ongoing utterance."""
    partials, finals = [], []
    ready = asyncio.Event()

    async def partial(uid, text):
        partials.append(text)

    async def final(uid, text, start, end):
        finals.append(text)
        ready.set()

    async def serve(reader, writer):
        await reader.read(10240)
        for text, done in [("Roll initiative", False), ("", True)]:
            writer.write((json.dumps(dict(text=text, is_final=done, start=0, end=1)) + "\n").encode())
        await writer.drain()
        await reader.read()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    adapter = StreamingSttAdapter(port=server.sockets[0].getsockname()[1],
                                  on_partial=partial, on_final=final, drain_timeout_s=0.01)
    try:
        await adapter.feed("speaker", bytes(10240))
        await asyncio.wait_for(ready.wait(), 1)
        assert partials == ["Roll initiative"]
        assert finals == ["Roll initiative"]
    finally:
        await adapter.close_all()
        server.close()
        await server.wait_closed()


async def test_remote_eof_releases_session_and_next_audio_reconnects():
    """A server restart must not leave a dead stream marked healthy."""
    connections = []

    async def serve(reader, writer):
        connections.append(writer)
        await reader.read(10240)
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    adapter = StreamingSttAdapter(port=server.sockets[0].getsockname()[1])
    try:
        for expected in (1, 2):
            await adapter.feed('speaker', bytes(10240))
            for _ in range(100):
                if not adapter.has_session('speaker'):
                    break
                await asyncio.sleep(0.01)
            assert not adapter.has_session('speaker')
            assert len(connections) == expected
    finally:
        await adapter.close_all()
        server.close()
        await server.wait_closed()
