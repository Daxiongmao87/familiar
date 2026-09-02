"""AudioSource protocol: every audio producer emits PcmChunks per speaker.

Implementations:
  - sources/replay.py     deterministic file playback (tests, regression, E2E)
  - sources/discord_src.py live Discord voice via py-cord (DAVE E2EE)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from dmd.types import PcmChunk


class AudioSource(ABC):
    """Pushes decoded mono int16 PCM chunks onto an internal queue."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[PcmChunk | None] = asyncio.Queue()
        self._running = False

    @abstractmethod
    async def run(self) -> None:
        """Produce frames until stop() is awaited. Must push None on termination."""

    def emit(self, chunk: PcmChunk | None) -> None:
        self._queue.put_nowait(chunk)

    async def start(self) -> AsyncIterator[PcmChunk]:
        self._running = True
        return self

    async def stop(self) -> None:
        self._running = False
        self._queue.put_nowait(None)

    def __aiter__(self) -> AudioSource:
        return self

    async def __anext__(self) -> PcmChunk:
        chunk = await self._queue.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk
