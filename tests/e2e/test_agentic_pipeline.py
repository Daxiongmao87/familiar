"""Full-pipeline agentic E2E: replay a scripted Discord session through the
real stack and assert the agentic outputs are grounded for the right triggers.

This exercises the FULL path, not just the UI: every line goes through
``SessionEngine.handle_utterance`` -> ``detect_trigger`` (the fast-lane
binary verdict) -> ``WorkerAgent``. The guarantees under test:

  * a loot trigger produces a grounded CARD (skill/DC table);
  * a lore trigger produces a grounded CARD (briefing) — the legacy path
    defaults every trigger to the card tier (recall bias; no taxonomy);
  * a non-trigger line produces nothing;
  * the fast lane was actually invoked (the classifier ran).

The deterministic mock backend stands in for the models; its cards
reference real campaign content (Vex'ahlia / the Ashforge), so "grounded"
here means "campaign-specific, not a generic stub". Web pre-grounding is
stubbed to empty results: this test pins trigger->card behavior, and live
search snippets would make the mock's content routing nondeterministic.
"""

from __future__ import annotations

from typing import Any

import pytest

from dmd.agent import AgentResult
from dmd.orchestrator import JobPool
from dmd.pipeline import SessionEngine
from dmd.types import Card, Utterance

# Scripted Discord session in order (the rolling transcript grows line by line).
SCRIPTED_SESSION: list[tuple[str, str]] = [
    ("kael", "I search the goblin corpse"),        # trigger -> CARD
    ("kael", "rough luck, nothing on him"),        # non-trigger -> nothing
    ("bryn", "what's the history of this place?"), # trigger -> CARD
]


@pytest.mark.e2e
async def test_agentic_session_grounds_outputs_for_right_triggers(stack: Any) -> None:
    cards: list[Card] = []
    events: list[dict[str, Any]] = []

    async def on_card(card: Card) -> None:
        cards.append(card)

    pool = JobPool(max_concurrent=2, job_timeout_s=15.0, stale_after_s=120.0, on_card=on_card)
    engine = SessionEngine(
        cfg=stack.cfg,
        store=stack.store,
        gw=stack.gw,
        entries=stack.entries,
        embedder=stack.embedder,
        pool=pool,
        on_event=events.append,
        project_path=str(stack.campaign_path),
    )

    async def _no_web_results(args: dict) -> dict:
        return {"results": [], "note": "stubbed empty for hermetic routing"}

    engine._agent._tool_web_search = _no_web_results  # type: ignore[method-assign]

    # Replay the session. handle_utterance only awaits queueing, so drain
    # before close: close() cancels still-queued jobs.
    t = 0.0
    for user_id, text in SCRIPTED_SESSION:
        await engine.handle_utterance(Utterance(user_id=user_id, text=text, t_start=t, t_end=t + 1.0))
        t += 1.0

    await pool.drain()
    await pool.close()

    # 1. Both triggers produced grounded cards; the non-trigger produced nothing.
    assert len(cards) == 2, f"expected exactly 2 cards (2 triggers), got {len(cards)}: {cards}"
    loot_card = cards[0]
    body = (loot_card.body_md or "").lower()
    assert "investigation" in body or "dc" in body, f"card lacks a skill/DC: {body!r}"
    assert any(w in body for w in ("vex'ahlia", "forge", "gp")), f"card not campaign-grounded: {body!r}"

    # 2. The lore trigger produced a grounded briefing card (no taxonomy left
    #    to route it to the ephemeral tier).
    lore_body = (cards[1].body_md or "").lower()
    assert any(w in lore_body for w in ("forge", "temple", "vex'ahlia")), (
        f"lore card not campaign-grounded: {lore_body!r}"
    )

    # 3. No scene notes: both triggers went to the card tier.
    scene_notes = [e for e in events if e.get("type") == "scene_context"]
    assert scene_notes == [], f"expected no scene notes, got {scene_notes}"

    # 4. The fast lane was actually used: the classifier was invoked on the fast role.
    fast_classifier_calls = [
        r
        for r in stack.state.requests
        if r["path"] == "/v1/chat/completions"
        and "fast" in str(r["body"].get("model", ""))
        and "response_format" in r["body"]
    ]
    assert fast_classifier_calls, "the fast-lane classifier was never invoked"


@pytest.mark.e2e
async def test_empty_agent_output_produces_no_scene_context(stack: Any) -> None:
    """Regression: an ephemeral-tier agent that produces no text (e.g. times out
    under GPU load) must NOT publish an empty scene-context event.

    Before the fix, ``_generate_card`` called ``_emit_scene("", ...)`` on an
    empty agent result, leaking a no-op scene note. The agent's run is patched
    to return empty text and the card generator is invoked directly on the
    ephemeral tier; zero scene_context events must result.
    """
    events: list[dict[str, Any]] = []

    pool = JobPool(max_concurrent=2, job_timeout_s=15.0, stale_after_s=120.0)
    engine = SessionEngine(
        cfg=stack.cfg,
        store=stack.store,
        gw=stack.gw,
        entries=stack.entries,
        embedder=stack.embedder,
        pool=pool,
        on_event=events.append,
        project_path=str(stack.campaign_path),
    )

    # Force the agent to produce nothing (simulated timeout / empty output).
    async def _run_empty(task, tier, trigger_portion="", transcript=""):
        return AgentResult(tier=tier, text="", error="simulated empty output")

    original_run = engine._agent.run
    engine._agent.run = _run_empty
    try:
        await engine._generate_card(
            {
                "utterance": "what's the history of this place?",
                "entities": [],
                "tier": "ephemeral",
                "recent": [],
            }
        )
    finally:
        engine._agent.run = original_run

    await pool.close()

    scene_notes = [e for e in events if e.get("type") == "scene_context"]
    assert scene_notes == [], (
        f"empty agent output must not emit a scene note, got {scene_notes}"
    )
