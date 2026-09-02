"""Speaking-state tracker: who is talking right now, via Discord gateway.

Familiar joins the guild's voice channels; the gateway delivers
member_speaking_state_update events even while DAVE audio decrypt is
broken, so this tracker works in the DAVE-broken world.
"""

from __future__ import annotations

import time
from typing import Any


class SpeakingTracker:
    """Tracks speaking windows per user_id from gateway events.

    Callers push events via on_speaking(user_id, speaking: bool, t: float).
    Query with active_at(t) or active_during(t0, t1).
    """

    def __init__(self) -> None:
        self._active: dict[str, float] = {}
        self._history: list[tuple[str, float, float | None]] = []

    def on_speaking(self, user_id: str, speaking: bool, t: float | None = None) -> None:
        now = t if t is not None else time.monotonic()
        uid = str(user_id)
        if speaking:
            if uid not in self._active:
                self._active[uid] = now
        else:
            start = self._active.pop(uid, None)
            if start is not None:
                self._history.append((uid, start, now))

    def active_at(self, t: float) -> list[str]:
        out = [uid for uid, start in self._active.items() if start <= t]
        for uid, s, e in self._history:
            if s <= t <= (e if e is not None else float("inf")):
                if uid not in out:
                    out.append(uid)
        return sorted(out)

    def active_during(self, t0: float, t1: float) -> list[str]:
        candidates: dict[str, float] = {}
        for uid, start in self._active.items():
            if start <= t1:
                candidates[uid] = t1 - max(start, t0)
        for uid, s, e in self._history:
            end = e if e is not None else t1
            if s <= t1 and end >= t0:
                overlap = min(end, t1) - max(s, t0)
                candidates[uid] = candidates.get(uid, 0) + overlap
        return sorted(candidates, key=lambda u: candidates[u], reverse=True)

    def snapshot(self) -> dict[str, Any]:
        return {"active": dict(self._active), "history_len": len(self._history)}
