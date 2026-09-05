"""Trigger detection: deterministic regex fast path plus a hard-bounded
optional fast-lane LLM tie-breaker.

Latency rationale (2026-09-05 Priority-1 measurement): the configured fast
endpoint (ling-3.0-tiny) is a *reasoning-first* model — a bare classification
request spends ~180 hidden reasoning tokens on every call and takes 17-24s
before it emits the JSON (measured: tools/latency_probe baseline, and a direct
timed POST to the fast role). It also misclassified an unambiguous
"we loot ... body" as is_trigger=false. That made the fast-lane classifier the
single largest term in the transcript->answer budget (turn_latency detect_ms
23854ms). The fast lane must be deterministic and instant, so the keyword
regex runs FIRST and short-circuits; the LLM is only a best-effort tie-breaker
for prose phrasings the regex misses, and it is bounded by a hard per-request
timeout so a slow reasoning call can never push an answer past its budget.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gateway import Gateway

# Hard ceiling on the fast-lane LLM classification. The 5s voice->transcript
# and 15s transcript->answer budgets are measured from speech-stop; a classify
# step that can run to 24s is disqualifying. 3s leaves the answer its window
# and, combined with the small max_tokens below, never lets one utterance
# dominate the critical path. On timeout/empty/misparse we fall back to the
# deterministic regex verdict.
LANE_CLASSIFY_TIMEOUT_S = 3.0
# The classifier's real answer is ~15 tokens; the tiny endpoint pads with
# reasoning. A tight cap bounds the worst case and lets an abandoned call
# release its llama.cpp slot quickly (the classifier never reaches 1024).
LANE_CLASSIFY_MAX_TOKENS = 96

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

    Precedence is deterministic-first for latency (see module docstring):

    1. The keyword regex short-circuits instantly and reliably for the
       search/loot/examine intents that dominate the fast lane — no LLM call,
       no network, no reasoning-token tax.
    2. Only when the regex is silent do we ask the fast-lane LLM (prose rules
       or lore questions the patterns can't see), under a hard
       ``LANE_CLASSIFY_TIMEOUT_S`` ceiling with a small ``max_tokens``. Any
       failure (bad config, HTTP error, timeout, malformed JSON, unexpected
       shape) falls back to the regex verdict.

    Without a fast role, only the regex fast path applies.
    """
    if not text:
        return False, "other"

    rule_hit = detect_trigger_rule(text)
    if rule_hit:
        return True, "loot"

    if _has_fast_role(gw):
        try:
            messages = [
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": text},
            ]
            result = await asyncio.wait_for(  # type: ignore[union-attr]
                gw.chat(
                    "fast",
                    messages,
                    json_schema=_TRIGGER_SCHEMA,
                    temperature=0,
                    max_tokens=LANE_CLASSIFY_MAX_TOKENS,
                ),
                timeout=LANE_CLASSIFY_TIMEOUT_S,
            )
            parsed = _parse_classifier_result(result)
            if parsed is not None:
                return parsed
        except Exception:
            pass

    # Regex already ruled this out (rule_hit is False here); the LLM either
    # agreed or was unavailable — either way the utterance is not a trigger.
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
