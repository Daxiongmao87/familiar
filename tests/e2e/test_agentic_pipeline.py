"""Full-pipeline agentic E2E: replay a scripted Discord session through the
real stack and assert the agentic outputs are grounded for the right triggers.

This exercises the FULL path, not just the UI: every line goes through
``SessionEngine.handle_utterance`` -> ``detect_trigger`` (the fast-lane
classifier) -> tier routing -> ``WorkerAgent``. The guarantees under test:

  * a loot trigger (fast lane kind="loot") produces a grounded CARD;
  * a lore trigger (fast lane kind="lore") produces a grounded scene-context
    (the ephemeral tier), not a card;
  * a non-trigger line produces neither;
  * the fast lane was actually invoked (the classifier ran).

The deterministic mock backend stands in for the models; its loot card and
scene note reference real campaign content (Vex'ahlia / the Ashforge), so
"grounded" here means "campaign-specific, not a generic stub".
"""

from __future__ import annotations

from typing import Any

import pytest

from dmd.orchestrator import JobPool
from dmd.pipeline import SessionEngine
from dmd.types import Card, Utterance

# Scripted Discord session in order (the rolling transcript grows line by line).
SCRIPTED_SESSION: list[tuple[str, str]] = [
    ("kael", "I search the goblin corpse"),        # loot trigger  -> CARD
    ("kael", "rough luck, nothing on him"),        # non-trigger   -> nothing
    ("bryn", "what's the history of this place?"), # lore trigger  -> scene context
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

    # Replay the session. handle_utterance awaits each agent job, so after this
    # loop every produced card / scene note has been published.
    t = 0.0
    for user_id, text in SCRIPTED_SESSION:
        await engine.handle_utterance(Utterance(user_id=user_id, text=text, t_start=t, t_end=t + 1.0))
        t += 1.0

    await pool.close()

    # 1. The loot trigger produced a grounded card (skill/DC table, campaign flavor).
    assert len(cards) == 1, f"expected exactly 1 card (loot trigger), got {len(cards)}: {cards}"
    loot_card = cards[0]
    body = (loot_card.body_md or "").lower()
    assert "investigation" in body or "dc" in body, f"card lacks a skill/DC: {body!r}"
    assert any(w in body for w in ("vex'ahlia", "forge", "gp")), f"card not campaign-grounded: {body!r}"

    # 2. The lore trigger produced a grounded scene context (ephemeral tier, no card).
    scene_notes = [e for e in events if e.get("type") == "scene_context"]
    assert len(scene_notes) == 1, f"expected exactly 1 scene context (lore trigger), got {len(scene_notes)}: {events}"
    scene_text = scene_notes[0].get("text", "").lower()
    assert any(w in scene_text for w in ("forge", "temple", "vex'ahlia")), (
        f"scene context not campaign-grounded: {scene_text!r}"
    )

    # 3. No card was produced for the lore trigger, and no scene note for the loot.
    #    (Guaranteed by the exact counts above: 1 card + 1 scene note for 3 lines.)

    # 4. The fast lane was actually used: the classifier was invoked on the fast role.
    fast_classifier_calls = [
        r
        for r in stack.state.requests
        if r["path"] == "/v1/chat/completions"
        and "fast" in str(r["body"].get("model", ""))
        and "response_format" in r["body"]
    ]
    assert fast_classifier_calls, "the fast-lane classifier was never invoked"
