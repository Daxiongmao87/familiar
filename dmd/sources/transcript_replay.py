"""Deterministic timed-transcript replay.

Feeds timestamped transcript entries (fixture JSON: ``t``/``text``/``type``/
``user_id``) to an async utterance sink, pacing wall-clock sleeps so each
entry lands at its recorded offset — mimicking live play for trigger,
retrieval, and publishing observation.

This is intentionally not an ``AudioSource``: it injects post-STT text,
bypassing VAD and transcription the way a scripted STT does in tests.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..types import Utterance


@dataclass(slots=True)
class TranscriptEvent:
    """One timed transcript entry from a replay fixture."""

    t: float
    user_id: str
    text: str


def load_transcript_events(path: str) -> list[TranscriptEvent]:
    """Load and validate a transcript fixture; returns entries in time order."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"transcript fixture must be a JSON array: {path}")
    events: list[TranscriptEvent] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"entry {i} is not an object: {path}")
        if item.get("type", "transcript") != "transcript":
            continue
        try:
            t = float(item["t"])
            user_id = str(item["user_id"])
            text = str(item["text"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"entry {i} missing t/user_id/text: {path}") from exc
        if not user_id or not text.strip():
            raise ValueError(f"entry {i} has empty user_id/text: {path}")
        events.append(TranscriptEvent(t=t, user_id=user_id, text=text))
    events.sort(key=lambda e: e.t)
    return events


class TranscriptReplayer:
    """Pace transcript events into an utterance sink at recorded offsets.

    With ``realtime=True`` the replayer sleeps between entries so wall-clock
    gaps match the fixture; with ``realtime=False`` entries dispatch
    back-to-back. ``sleep``/``clock`` are injectable for deterministic tests.
    """

    def __init__(
        self,
        events: list[TranscriptEvent],
        sink: Callable[[Utterance], Awaitable[Any]],
        realtime: bool = True,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._events = sorted(events, key=lambda e: e.t)
        self._sink = sink
        self._realtime = bool(realtime)
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or time.monotonic

    @property
    def events(self) -> list[TranscriptEvent]:
        """The loaded entries in dispatch order."""
        return list(self._events)

    async def run(self) -> int:
        """Dispatch every entry; returns the count delivered."""
        if not self._events:
            return 0
        t0 = self._events[0].t
        wall_base = self._clock()
        delivered = 0
        for i, ev in enumerate(self._events):
            if self._realtime:
                delay = (ev.t - t0) - (self._clock() - wall_base)
                if delay > 0:
                    await self._sleep(delay)
            nxt = self._events[i + 1].t if i + 1 < len(self._events) else ev.t + 2.0
            await self._sink(
                Utterance(
                    user_id=ev.user_id,
                    text=ev.text,
                    t_start=wall_base + (ev.t - t0),
                    t_end=wall_base + max(nxt - t0, ev.t - t0 + 0.1),
                )
            )
            delivered += 1
        return delivered
