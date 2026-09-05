"""Transcript monitor: proactive, cadence-based reasoning over the session.

A v2 layer that runs in the background: every ``monitor_cadence_s`` it reads the
rolling transcript + current scene context and asks the fast model whether the
DM should be proactively alerted to something (a consequence, a forgotten
thread, a player in danger, or a card that has just been resolved). The
monitor only PRODUCES a verdict; the engine decides how to act on it (fire an
ephemeral scene note, fire a card job, or auto-mark a card done). This keeps
the monitor decoupled from engine internals and easy to test.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

_MONITOR_SYSTEM = (
    "You are a proactive DM monitor for a live tabletop session. Read the recent "
    "transcript and the current scene context, then decide whether the DM should be "
    "proactively alerted to something: a consequence the players have triggered, a "
    "forgotten thread, a player in danger, or a card that has now been resolved in the "
    "conversation. Respond with JSON only.\n"
    "Options:\n"
    "- action='surface', tier='ephemeral', text='<short 1-2 sentence scene note>' for a proactive 'you might want to know' note.\n"
    "- action='surface', tier='card', reason='<what to look up and produce as a card>' for a durable artifact worth a card.\n"
    "- action='card_done', card_id='<id>' when the transcript clearly shows a card's content is resolved.\n"
    "- action='none' when nothing needs attention.\n"
    "Default to action='none' unless something is clearly actionable. Do not invent "
    "events that are not in the transcript.\n"
    "Additionally, on EVERY tick, predict what the next beats will need (advisory "
    "only — this never mutates anything):\n"
    "- situation='<one-line summary of the current situation>';\n"
    "- predicted_entities=['<canonical entity names likely to matter in the next 1-2 "
    "beats>'] (characters, places, items, factions grounded in the transcript/scene; "
    "at most 6; do not invent entities that are not present or clearly foreshadowed);\n"
    "- likely_next_events=['<short phrase>'] (e.g. 'loot the corpse', 'death save', "
    "'search the chapel altar')."
)

_MONITOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["surface", "card_done", "none"]},
        "tier": {"type": "string", "enum": ["ephemeral", "card"]},
        "text": {"type": "string"},
        "card_id": {"type": "string"},
        "reason": {"type": "string"},
        "situation": {"type": "string"},
        "predicted_entities": {"type": "array", "items": {"type": "string"}},
        "likely_next_events": {"type": "array", "items": {"type": "string"}},
    },
    # The prediction fields are REQUIRED, not optional: verified live on the
    # configured fast role (ling-3.0-tiny) that a tiny model skips optional
    # fields entirely (returns only action), while schema-required fields are
    # enforced by the endpoint and always emitted — without this the staging
    # prefetch would never fire. Defaults to empty arrays when nothing is
    # predicted; the judge only ever forwards non-empty lists to on_predict.
    "required": [
        "action",
        "situation",
        "predicted_entities",
        "likely_next_events",
    ],
}

_MIN_TRANSCRIPT_CHARS = 40

# Defensive ceiling on how many predicted entities a single verdict may ask to
# prefetch (the engine applies its own stricter StagingConfig.max_predicted).
_MAX_PREDICTED = 12


def _predicted_entities(result: dict[str, Any]) -> list[str]:
    raw = result.get("predicted_entities")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for ent in raw:
        if not isinstance(ent, str):
            continue
        e = ent.strip()
        if e and e not in out:
            out.append(e)
    return out[:_MAX_PREDICTED]


def _parse(result: object) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    action = result.get("action")
    if action not in ("surface", "card_done", "none"):
        return None
    return result


class TranscriptMonitor:
    """Cadence-driven proactive monitor over the rolling transcript."""

    def __init__(
        self,
        gw: Any,
        cfg: Any,
        get_transcript: Callable[[], str],
        get_scene: Callable[[], str],
        on_action: Callable[[dict[str, Any]], Awaitable[None]],
        cadence_s: float | None = None,
        on_predict: Callable[[list[str]], Awaitable[None]] | None = None,
    ) -> None:
        self.gw = gw
        self.cfg = cfg
        self.get_transcript = get_transcript
        self.get_scene = get_scene
        self.on_action = on_action
        self.on_predict = on_predict
        self.cadence = cadence_s if cadence_s is not None else getattr(cfg, "monitor_cadence_s", 30.0)
        self._task: asyncio.Task | None = None
        self._stopping = False
        self.ticks = 0

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.get_running_loop().create_task(self._run())

    def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(self.cadence)
            except asyncio.CancelledError:
                return
            if self._stopping:
                return
            try:
                await self.tick_once()
            except asyncio.CancelledError:
                return
            except Exception:
                # A monitor tick must never take the session down.
                continue

    async def tick_once(self) -> None:
        """One monitor pass: read context, judge, and act (via on_action).

        A verdict's ``predicted_entities`` are forwarded to ``on_predict`` on
        EVERY tick that carries them (including ``action='none'``) — the
        prediction is the staging trigger, independent of whether an alert is
        warranted — so the engine can prefetch ahead of the next real turn.
        """
        transcript = self.get_transcript() or ""
        if len(transcript.strip()) < _MIN_TRANSCRIPT_CHARS:
            return
        scene = self.get_scene() or ""
        verdict = await self._judge(transcript, scene)
        self.ticks += 1
        if not verdict:
            return
        predicted = _predicted_entities(verdict)
        if predicted and self.on_predict is not None:
            try:
                await self.on_predict(predicted)
            except Exception:
                # A prefetch must never take the monitor (or session) down.
                pass
        if verdict.get("action") not in (None, "none"):
            await self.on_action(verdict)

    async def _judge(self, transcript: str, scene: str) -> dict[str, Any] | None:
        user = (
            f"RECENT TRANSCRIPT:\n{transcript[-4000:]}\n\n"
            f"CURRENT SCENE CONTEXT:\n{scene[-1500:] or '(none)'}"
        )
        try:
            result = await self.gw.chat(
                "fast",
                [{"role": "system", "content": _MONITOR_SYSTEM}, {"role": "user", "content": user}],
                json_schema=_MONITOR_SCHEMA,
                temperature=0,
                max_tokens=1024,
            )
        except Exception:
            return None
        return _parse(result)
