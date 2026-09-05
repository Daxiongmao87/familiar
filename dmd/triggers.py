"""Trigger detection: regex rule fallback plus optional fast-lane LLM classifier."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gateway import Gateway

TRIGGER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(i|we|lets|let's)\s+(search|loot|examine|inspect)\b", re.IGNORECASE),
    re.compile(r"\b(search(es|ing)?|examin(e|es|ing)|inspect(s|ing)?)\s+the\s+\w+", re.IGNORECASE),
    re.compile(r"\bwhat.s?\s+(in|on)\s+(the\s+)?(body|corpse|desk|chest|bag)\b", re.IGNORECASE),
    re.compile(r"\bloot\s+the\s+\w+", re.IGNORECASE),
]

_TRIGGER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "is_trigger": {"type": "boolean"},
        "kind": {"type": "string", "enum": ["loot", "lore", "rules", "other"]},
    },
    "required": ["is_trigger", "kind"],
    "additionalProperties": False,
}

_CLASSIFIER_SYSTEM = (
    "You are a strict intent classifier for a tabletop RPG session. "
    "Read the player's transcript and return JSON with two fields: "
    "is_trigger (true if the line demands a DM artifact: a skill check, a rules ruling, "
    "or a lore briefing) and kind (loot for search/loot/examine intents, lore for world or "
    "entity questions, rules for rules questions, other for everything else). "
    "Default to is_trigger=false and kind=other unless the line clearly asks."
)


def detect_trigger_rule(text: str) -> bool:
    """Return True if any rule pattern matches the given text."""
    if not text:
        return False
    for pat in TRIGGER_PATTERNS:
        if pat.search(text):
            return True
    return False


def _has_fast_role(gw: Gateway | None) -> bool:
    if gw is None:
        return False
    cfg = getattr(gw, "_cfg", None) or getattr(gw, "cfg", None)
    if cfg is None:
        return False
    models = getattr(cfg, "models", None)
    if models is None:
        return False
    return getattr(models, "fast", None) is not None


async def detect_trigger(gw: Gateway | None, text: str) -> tuple[bool, str]:
    """Classify transcript intent. Returns (is_trigger, kind).

    When ``gw`` exposes a configured fast role, the fast lane LLM is asked via
    ``gw.chat('fast', ...)`` with the classifier JSON schema. Any failure (bad
    config, HTTP error, malformed JSON, unexpected shape) falls back to the
    regex rules. Without a fast role, the regex rules are used directly.
    """
    if not text:
        return False, "other"

    if _has_fast_role(gw):
        try:
            messages = [
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": text},
            ]
            result = await gw.chat(  # type: ignore[union-attr]
                "fast",
                messages,
                json_schema=_TRIGGER_SCHEMA,
                temperature=0,
                max_tokens=1024,
            )
            parsed = _parse_classifier_result(result)
            if parsed is not None:
                return parsed
        except Exception:
            pass

    if detect_trigger_rule(text):
        return True, "loot"
    return False, "other"


def _parse_classifier_result(result: object) -> tuple[bool, str] | None:
    if not isinstance(result, dict):
        return None
    is_trigger = result.get("is_trigger")
    kind = result.get("kind")
    if not isinstance(is_trigger, bool):
        return None
    if not isinstance(kind, str):
        return None
    if kind not in ("loot", "lore", "rules", "other"):
        return None
    return is_trigger, kind
