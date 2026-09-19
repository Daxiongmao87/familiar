"""Browser-captured audio source: receives PCM pushed from the frontend over WebSocket."""

from __future__ import annotations

import asyncio
import time

from dmd.sources.base import AudioSource
from dmd.types import PcmChunk


class BrowserAudioSource(AudioSource):
    """Single mixed stream from the browser's getDisplayMedia + getUserMedia.

    The frontend mixes system audio + mic into one mono 16kHz PCM stream and
    pushes chunks via the /ws/audio WebSocket. Attribution comes from the
    companion SpeakingTracker, not from per-user audio separation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._closed = False

    def push_chunk(self, pcm: bytes, t_mono: float | None = None) -> None:
        if self._closed or not pcm:
            return
        self.emit(
            PcmChunk(
                user_id="browser_mixed",
                samples=pcm,
                sample_rate=16000,
                t_mono=t_mono if t_mono is not None else time.monotonic(),
            )
        )

    async def run(self) -> None:
        self._running = True
        await self._stop_event.wait() if hasattr(self, "_stop_event") else asyncio.Event().wait()

    def close(self) -> None:
        self._closed = True
        self.emit(None)
