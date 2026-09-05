"""Unit tests for dmd.triggers.

Precedence is deterministic-first for latency (see dmd/triggers.py):

- detect_trigger_rule matches / non-matches for the spec examples.
- a rule match short-circuits WITHOUT consulting the (slow) fast lane.
- detect_trigger rule fallback when no fast lane is available.
- the fast-lane LLM is only reached when the regex is silent.
- the fast-lane call is hard-bounded: a slow/erroring LLM can never block the
  lane past its timeout, and always carries a capped max_tokens (slot-leak).
- kind passthrough / malformed-result fallback.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from dmd import triggers as triggers_mod
from dmd.triggers import detect_trigger, detect_trigger_rule

# ---------------------------------------------------------------------------
# detect_trigger_rule: positive matches
# ---------------------------------------------------------------------------

MATCHES_PASSING = [
    "I search the body",
    "we loot the chest",
    "what's in the corpse",
    "let's examine the altar",
]


@pytest.mark.parametrize("text", MATCHES_PASSING)
def test_detect_trigger_rule_matches(text: str) -> None:
    assert detect_trigger_rule(text) is True


# ---------------------------------------------------------------------------
# detect_trigger_rule: non-matches (casual chatter)
# ---------------------------------------------------------------------------

NON_MATCHES = [
    "hello everyone",
    "my turn",
    "roll initiative",
]


@pytest.mark.parametrize("text", NON_MATCHES)
def test_detect_trigger_rule_does_not_match(text: str) -> None:
    assert detect_trigger_rule(text) is False


# ---------------------------------------------------------------------------
# detect_trigger with gw=None: pure rule fallback.
# ---------------------------------------------------------------------------


async def test_detect_trigger_no_gateway_uses_rule_fallback_positive() -> None:
    is_trigger, kind = await detect_trigger(None, "I search the body")
    assert is_trigger is True
    assert kind == "loot"


async def test_detect_trigger_no_gateway_uses_rule_fallback_negative() -> None:
    is_trigger, kind = await detect_trigger(None, "hello everyone")
    assert is_trigger is False
    assert kind == "other"


# ---------------------------------------------------------------------------
# Fake gateway for fast-lane tests.
# ---------------------------------------------------------------------------


class _FakeModels:
    def __init__(self, fast: Any) -> None:
        self.fast = fast


class _FakeCfg:
    def __init__(self, fast: Any) -> None:
        self.models = _FakeModels(fast)


class _FakeGw:
    """Minimal gateway stub: exposes _cfg with a fast model and a chat() method."""

    def __init__(
        self,
        chat_return: Any = None,
        chat_raises: bool = False,
        fast_present: bool = True,
        chat_sleep: float = 0.0,
    ) -> None:
        # `_has_fast_role` looks for `gw._cfg.models.fast` truthy.
        self._cfg = _FakeCfg(fast="stub-fast-model" if fast_present else None)
        self._chat_return = chat_return
        self._chat_raises = chat_raises
        self._chat_sleep = chat_sleep
        self.calls: list[dict] = []

    async def chat(
        self,
        role: str,
        messages: list[dict],
        json_schema: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "json_schema": json_schema,
                "max_tokens": max_tokens,
            }
        )
        if self._chat_sleep:
            await asyncio.sleep(self._chat_sleep)
        if self._chat_raises:
            raise RuntimeError("stub gw.chat failure")
        return self._chat_return


# ---------------------------------------------------------------------------
# Latency fast path: a rule match short-circuits the slow LLM entirely.
# ---------------------------------------------------------------------------


async def test_rule_match_short_circuits_without_consulting_llm() -> None:
    """Regression (2026-09-05 Priority-1): the fast-lane LLM took 17-24s on the
    tiny reasoning endpoint and misfired on loot lines. A keyword match must
    return instantly WITHOUT any network call."""
    gw = _FakeGw(chat_return={"is_trigger": False, "kind": "other"})
    t0 = time.monotonic()
    is_trigger, kind = await detect_trigger(gw, "I search the body")
    assert is_trigger is True
    assert kind == "loot"
    # The slow LLM was never reached.
    assert gw.calls == []
    assert time.monotonic() - t0 < 0.05


# ---------------------------------------------------------------------------
# detect_trigger: rule non-match lets the fast lane classify.
# ---------------------------------------------------------------------------


async def test_detect_trigger_fast_lane_classifies_when_rule_silent() -> None:
    # "hello everyone" is a rule non-match, but the fast lane says trigger.
    gw = _FakeGw(chat_return={"is_trigger": True, "kind": "lore"})
    is_trigger, kind = await detect_trigger(gw, "hello everyone")
    assert is_trigger is True
    assert kind == "lore"
    assert len(gw.calls) == 1
    assert gw.calls[0]["role"] == "fast"


# ---------------------------------------------------------------------------
# detect_trigger: a slow LLM is hard-bounded (never blocks the lane).
# ---------------------------------------------------------------------------


async def test_slow_llm_is_hard_bounded_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fast-lane call that outlives the timeout must not hold the utterance:
    the classifier returns the regex verdict (not-a-trigger) within budget."""
    monkeypatch.setattr(triggers_mod, "LANE_CLASSIFY_TIMEOUT_S", 0.2)
    gw = _FakeGw(chat_return={"is_trigger": True, "kind": "loot"}, chat_sleep=5.0)
    t0 = time.monotonic()
    is_trigger, kind = await detect_trigger(gw, "tell me a story about the keep")
    elapsed = time.monotonic() - t0
    assert is_trigger is False
    assert kind == "other"
    # bounded well under the sleep it was waiting on
    assert elapsed < 1.0, f"classifier not bounded: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# detect_trigger: gw.chat raising falls back to the rule verdict.
# ---------------------------------------------------------------------------


async def test_detect_trigger_falls_back_when_gw_chat_raises() -> None:
    # Rule MATCHES => fast path returns the trigger without touching the LLM.
    gw = _FakeGw(chat_raises=True)
    is_trigger, kind = await detect_trigger(gw, "I search the body")
    assert is_trigger is True
    assert kind == "loot"
    assert gw.calls == []


async def test_detect_trigger_falls_back_when_gw_chat_raises_negative() -> None:
    gw = _FakeGw(chat_raises=True)
    # Rule does NOT match => LLM raises => fallback is (False, "other").
    is_trigger, kind = await detect_trigger(gw, "hello everyone")
    assert is_trigger is False
    assert kind == "other"


# ---------------------------------------------------------------------------
# detect_trigger: kind passthrough from fast lane.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    ["loot", "lore", "rules", "other"],
)
async def test_detect_trigger_kind_passthrough(kind: str) -> None:
    gw = _FakeGw(chat_return={"is_trigger": True, "kind": kind})
    is_trigger, out_kind = await detect_trigger(gw, "a prose question with no keyword")
    assert is_trigger is True
    assert out_kind == kind


# ---------------------------------------------------------------------------
# detect_trigger: malformed fast-lane result also falls back to rules.
# ---------------------------------------------------------------------------


async def test_detect_trigger_malformed_fast_lane_falls_back_to_rule() -> None:
    # Rule silent here, so the LLM is consulted; a bad kind falls back to
    # the regex verdict (not-a-trigger).
    gw = _FakeGw(chat_return={"is_trigger": True, "kind": "bogus"})
    is_trigger, kind = await detect_trigger(gw, "the wind picks up outside")
    assert is_trigger is False
    assert kind == "other"


# ---------------------------------------------------------------------------
# Slot-leak guard (2026-09-05 incident): the classifier call must bound
# generation — a missing max_tokens ran llama.cpp with n_predict=-1. The cap
# is now small (the classifier never reaches the old 1024).
# ---------------------------------------------------------------------------


async def test_detect_trigger_fast_lane_passes_capped_max_tokens() -> None:
    gw = _FakeGw(chat_return={"is_trigger": False, "kind": "other"})
    await detect_trigger(gw, "hello everyone")
    assert gw.calls, "fast lane was not used"
    assert gw.calls[0]["max_tokens"] == triggers_mod.LANE_CLASSIFY_MAX_TOKENS
    assert gw.calls[0]["max_tokens"] is not None
    assert gw.calls[0]["max_tokens"] < 1024
