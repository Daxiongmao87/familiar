"""Unit tests for dmd.triggers.

Covers:
- detect_trigger_rule matches / non-matches for the spec examples.
- detect_trigger rule fallback when no fast lane is available.
- detect_trigger fast-lane precedence (used even when it contradicts rules).
- detect_trigger defensive fallback when gw.chat raises.
- detect_trigger kind passthrough from the fast lane.
"""

from __future__ import annotations

from typing import Any

import pytest

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
    ) -> None:
        # `_has_fast_role` looks for `gw._cfg.models.fast` truthy.
        self._cfg = _FakeCfg(fast="stub-fast-model" if fast_present else None)
        self._chat_return = chat_return
        self._chat_raises = chat_raises
        self.calls: list[dict] = []

    async def chat(
        self,
        role: str,
        messages: list[dict],
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> Any:
        self.calls.append(
            {"role": role, "messages": messages, "json_schema": json_schema}
        )
        if self._chat_raises:
            raise RuntimeError("stub gw.chat failure")
        return self._chat_return


# ---------------------------------------------------------------------------
# detect_trigger: fast-lane overrides the rule verdict.
# ---------------------------------------------------------------------------


async def test_detect_trigger_fast_lane_overrides_rules() -> None:
    # "hello everyone" is a non-match by rule, but the fast lane says trigger.
    gw = _FakeGw(chat_return={"is_trigger": True, "kind": "loot"})
    is_trigger, kind = await detect_trigger(gw, "hello everyone")
    assert is_trigger is True
    assert kind == "loot"
    # And the fast lane was actually consulted.
    assert len(gw.calls) == 1
    assert gw.calls[0]["role"] == "fast"


# ---------------------------------------------------------------------------
# detect_trigger: gw.chat raising falls back to the rule verdict.
# ---------------------------------------------------------------------------


async def test_detect_trigger_falls_back_when_gw_chat_raises() -> None:
    gw = _FakeGw(chat_raises=True)
    # Rule matches => still reported as a trigger via fallback.
    is_trigger, kind = await detect_trigger(gw, "I search the body")
    assert is_trigger is True
    assert kind == "loot"


async def test_detect_trigger_falls_back_when_gw_chat_raises_negative() -> None:
    gw = _FakeGw(chat_raises=True)
    # Rule does NOT match => fallback is (False, "other").
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
    is_trigger, out_kind = await detect_trigger(gw, "anything at all")
    assert is_trigger is True
    assert out_kind == kind


# ---------------------------------------------------------------------------
# detect_trigger: malformed fast-lane result also falls back to rules.
# ---------------------------------------------------------------------------


async def test_detect_trigger_malformed_fast_lane_falls_back_to_rule() -> None:
    # Shape that fails _parse_classifier_result (kind not in allowed set).
    gw = _FakeGw(chat_return={"is_trigger": True, "kind": "bogus"})
    is_trigger, kind = await detect_trigger(gw, "I search the body")
    assert is_trigger is True
    assert kind == "loot"
