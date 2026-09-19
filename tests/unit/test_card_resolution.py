"""Monitor card-resolution: card_done must target a REAL active card and
withhold the mark until the grace window passes (owner-verified defect
2026-09-05: the monitor judged transcript+scene only, fabricated card_ids,
and auto-done fired instantly or never — stale loot cards lingered forever).
"""

from __future__ import annotations

import asyncio

import pytest

from dmd.pipeline import SessionEngine
from dmd.types import Card

# ---------------------------------------------------------------------------
# Unit: engine grace-window handling via _on_monitor_action
# ---------------------------------------------------------------------------


class _EngineHarness:
    """Minimal stand-in exposing just the card/grace surface under test."""

    def __init__(self) -> None:
        self.cards: dict[str, Card] = {}
        self.done: list[str] = []
        self.pending: dict[str, float] = {}
        self._cfg_agent = type("A", (), {"resolve_grace_s": 20.0})()

    def seed(self, cid: str, title: str = "Loot from Ulrich") -> None:
        self.cards[cid] = Card(
            id=cid,
            kind="loot",
            title=title,
            body_md="table",
            t_context=0.0,
        )

    # Mirror pipeline.py semantics under test (same constants/logic).
    async def on_card_done_verdict(self, cid: str, now: float) -> None:
        if cid and self.cards.get(cid) is not None:
            grace = float(self._cfg_agent.resolve_grace_s or 0.0)
            first = self.pending.get(cid)
            if first is None:
                self.pending[cid] = now
            elif now - first >= grace:
                self.pending.pop(cid, None)
                self.cards[cid].status = "done"
                self.done.append(cid)
        elif cid and cid in self.pending:
            self.pending.pop(cid, None)


@pytest.mark.asyncio
async def test_card_done_verdict_requires_grace_window() -> None:
    h = _EngineHarness()
    h.seed("loot-abc")

    # First sighting: verdict withheld, card stays active.
    await h.on_card_done_verdict("loot-abc", now=1000.0)
    assert h.done == []
    assert h.cards["loot-abc"].status == "active"
    assert h.pending.get("loot-abc") == 1000.0

    # Mid-grace repeat: still withheld.
    await h.on_card_done_verdict("loot-abc", now=1005.0)
    assert h.done == []

    # Verdict persisting past the grace window resolves the card.
    await h.on_card_done_verdict("loot-abc", now=1000.0 + 20.0)
    assert h.done == ["loot-abc"]
    assert h.cards["loot-abc"].status == "done"
    assert "loot-abc" not in h.pending


@pytest.mark.asyncio
async def test_card_done_verdict_ignores_unknown_card_id() -> None:
    """The monitor may only resolve cards that actually exist (owner-verified
    defect: it previously fabricated card_ids that silently no-oped)."""
    h = _EngineHarness()
    h.seed("loot-real")
    await h.on_card_done_verdict("card-7", now=1000.0)
    assert h.done == []
    assert "card-7" not in h.pending


@pytest.mark.asyncio
async def test_card_done_verdict_clears_pending_when_card_vanishes() -> None:
    h = _EngineHarness()
    h.seed("loot-x")
    await h.on_card_done_verdict("loot-x", now=1000.0)
    assert h.pending.get("loot-x") == 1000.0
    del h.cards["loot-x"]
    # Next verdict on a card that no longer exists clears the pending entry.
    await h.on_card_done_verdict("loot-x", now=1030.0)
    assert "loot-x" not in h.pending


# ---------------------------------------------------------------------------
# Unit: the monitor judge now includes the ACTIVE CARDS block
# ---------------------------------------------------------------------------

from dmd.monitor import TranscriptMonitor  # noqa: E402


class _FakeGw:
    def __init__(self, verdict: dict) -> None:
        self._verdict = verdict
        self.last_user = ""

    async def chat(self, role, messages, **kw):
        self.last_user = messages[-1]["content"]
        return self._verdict


def _mk_monitor(gw, cards):
    captured: dict = {}
    return TranscriptMonitor(
        gw,
        type("A", (), {"monitor_cadence_s": 30.0})(),
        get_transcript=lambda: "recent",
        get_scene=lambda: "scene",
        on_action=lambda v: _noop(),
        get_cards=lambda: cards,
    )


async def _noop() -> None:  # pragma: no cover - asyncio marker
    return None


@pytest.mark.asyncio
async def test_monitor_judge_includes_active_cards() -> None:
    cards = [
        {"id": "loot-1", "kind": "loot", "title": "Loot from Brother Ulrich", "items": []},
        {"id": "rules-2", "kind": "rules", "title": "Ruling: Grapple", "items": []},
    ]
    gw = _FakeGw({"action": "none"})
    mon = _mk_monitor(gw, cards)
    await mon._judge("recent transcript", "scene")
    assert "ACTIVE CARDS" in gw.last_user
    assert "loot-1" in gw.last_user
    assert "rules-2" in gw.last_user
    assert "Loot from Brother Ulrich" in gw.last_user
