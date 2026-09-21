"""Headless session replay shared by deterministic and live-STT tests.

Speaking events and transcript references are independent fixture inputs.
Times are seconds relative to the recording start, including silence.
Audio mode requires real-time pacing and mono 16-bit 16 kHz WAV input.
"""

from __future__ import annotations

import asyncio
import time
import wave
from dataclasses import dataclass
from typing import Callable, Awaitable, Any

from dmd.sources.base import AudioSource
from dmd.types import PcmChunk


@dataclass(frozen=True)
class ReplayEvent:
    """One Discord event or independently recorded STT completion."""

    at: float
    kind: str
    payload: dict[str, Any]


class SessionReplay(AudioSource):
    """Replay both inputs on one clock without desktop capture or Discord."""

    def __init__(self, events: list[ReplayEvent],
                 discord_sink: Callable[[dict], Awaitable[None]],
                 final_sink: Callable[..., Awaitable[None]], *,
                 wav_path: str | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        super().__init__()
        for event in events:
            if event.at < 0 or event.kind not in {"discord", "final"}:
                raise ValueError("invalid replay event")
            if event.kind == "final":
                p = event.payload
                if not 0 <= p["start"] <= p["end"] <= event.at:
                    raise ValueError("final must arrive after its audio window")
        if wav_path and any(e.kind == "final" for e in events):
            raise ValueError("audio mode must use real STT, not injected finals")
        self.events = events
        self.discord_sink = discord_sink
        self.final_sink = final_sink
        self.wav_path = wav_path
        self.clock, self.sleep = clock, sleep
        self.epoch = 0.0

    async def run(self) -> None:
        """Pace inputs together; always terminate the AudioSource queue."""
        self.epoch = self.clock()
        self._running = True
        timeline = [(e.at, e.kind, e.payload) for e in self.events]
        try:
            if self.wav_path:
                with wave.open(self.wav_path, "rb") as audio:
                    if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                        raise ValueError("expected mono PCM16 16000 Hz WAV")
                    offset = 0
                    while pcm := audio.readframes(3200):
                        timeline.append((offset / 16000, "pcm", pcm))
                        offset += len(pcm) // 2
            # Stable ordering lets fixtures specify simultaneous stop/start events.
            for at, kind, payload in sorted(timeline, key=lambda item: item[0]):
                if not self._running:
                    break
                delay = self.epoch + at - self.clock()
                if delay > 0:
                    await self.sleep(delay)
                if kind == "discord":
                    await self.discord_sink(payload)
                elif kind == "final":
                    await self.final_sink("browser_mixed", payload["text"],
                                          self.epoch + payload["start"],
                                          self.epoch + payload["end"])
                else:
                    self.emit(PcmChunk(user_id="browser_mixed", samples=payload,
                                       sample_rate=16000, t_mono=self.epoch + at))
        finally:
            self._running = False
            self.emit(None)


def word_error_rate(reference: str, actual: str) -> float:
    """Word edit distance divided by reference length; may exceed one."""
    import re

    expected = re.findall(r"\w+", reference.lower())
    observed = re.findall(r"\w+", actual.lower())
    row = list(range(len(observed) + 1))
    for i, word in enumerate(expected, 1):
        nxt = [i]
        for j, other in enumerate(observed, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = nxt
    return row[-1] / max(len(expected), 1)
