"""Rolling speaker-attribution review: new context reassesses prior blocks.

Every new finalized transcript block must provide new context for JEV to
reassess prior recent blocks — including blocks that looked timing-certain
at first. Score distributions and their history are retained so confidence
changes are observable; later dialogue can change earlier identities.

These tests prove the scheduling/audit contract with scripted scorer
responses. Scripted responses prove scheduling ONLY, never model accuracy.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from typing import Any

from dmd.config import ConfigError, load_config_dict
from dmd.pipeline import SessionEngine
from dmd.speaking_tracker import SpeakingTracker


def _cfg(review: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
    """Engine config with optional attribution-review overrides."""
    raw: dict[str, Any] = {
        "models": {
            "synthesis": {"base_url": "http://unused", "model_id": "fixture"},
            "stt": {},
        }
    }
    if review:
        raw["attribution_review"] = review
    return load_config_dict(raw)


def _engine(published: list[dict], tracker: SpeakingTracker,
            review: dict[str, Any] | None = None) -> SessionEngine:
    engine = SessionEngine(_cfg(review), None, None, [], None, None,
                           published.append, speaking_tracker=tracker)
    engine.set_ooc(True)  # isolate transcription/attribution from generation
    return engine


async def _settle(engine: SessionEngine, timeout_s: float = 5.0) -> None:
    """Wait until scheduled attribution reviews finish (bounded)."""
    drain = getattr(engine, "drain_attribution_reviews", None)
    if callable(drain):
        await drain(timeout_s)
    else:  # pre-fix engine: fire-and-forget tasks get one grace window
        await asyncio.sleep(min(timeout_s, 0.5))


def _transcripts(published: list[dict]) -> list[dict]:
    return [e for e in published if e["type"] == "transcript"]


def _revisions(published: list[dict]) -> list[dict]:
    return [e for e in published if e["type"] == "transcript_revision"]


class _ScriptedRank:
    """Recorded, scripted JEV scorer: scheduling oracle, not a model."""

    def __init__(self, script: Any) -> None:
        self._script = script
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, row_id: str, state: str, question: str,
                       options: list[dict[str, str]]) -> dict[str, float]:
        call = {"row_id": row_id, "state": state, "question": question,
                "options": [dict(o) for o in options]}
        self.calls.append(call)
        step = self._script
        if isinstance(step, list):
            step = step[min(len(self.calls) - 1, len(step) - 1)]
        if isinstance(step, Exception):
            raise step
        if callable(step):
            result = step(call)
            if inspect.isawaitable(result):
                result = await result
            return dict(result)
        return dict(step)

    def for_block(self, block_id: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if block_id in c["row_id"]]


def _dist(call: dict[str, Any], winner: str, score: float) -> dict[str, float]:
    """A complete valid distribution over the offered options."""
    ids = [o["id"] for o in call["options"]]
    assert winner in ids, f"{winner} not offered: {ids}"
    rest = (1.0 - score) / max(len(ids) - 1, 1)
    return {i: (score if i == winner else rest) for i in ids}


def _targets(call: dict[str, Any], block_id: str) -> bool:
    """Whether the scored TARGET line is the given transcript block."""
    return bool(block_id) and f"[TARGET id={block_id}" in call["state"]


def _windows(tracker: SpeakingTracker) -> None:
    """Recorded-case speakers, including target-window crosstalk."""
    tracker.on_speaking("laura", True, t=1.0, name="Laura")
    tracker.on_speaking("laura", False, t=3.0)
    tracker.on_speaking("travis", True, t=1.8, name="Travis")
    tracker.on_speaking("travis", False, t=5.0)
    tracker.on_speaking("liam", True, t=5.0, name="Liam")
    tracker.on_speaking("liam", False, t=7.0)


async def test_new_context_rescores_prior_confident_block(monkeypatch: Any) -> None:
    """Core contract: later dialogue re-scores an initially certain block.

    Block A is timing-certain (Laura) at publication. Blocks B/C arrive with
    new evidence; A must be scored AGAIN using B/C, its history must change,
    and the same UI row must be revised without duplicate lines or cards.
    """
    tracker = SpeakingTracker()
    _windows(tracker)
    published: list[dict] = []
    engine = _engine(published, tracker)
    block_a = {"id": ""}

    def script(call: dict[str, Any]) -> dict[str, float]:
        if _targets(call, block_a["id"]):
            if "Crownsguard" in call["state"]:
                return _dist(call, "member:travis", 0.85)
            return _dist(call, "member:laura", 0.90)
        first = call["options"][0]["id"]
        return _dist(call, first, 0.60)

    rank = _ScriptedRank(script)
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final(
            "browser_mixed", "I cast Disguise Self", 1.0, 2.0)
        first = _transcripts(published)[0]
        block_a["id"] = first["id"]
        assert first["user_id"] == "laura"
        assert first["attribution"]["state"] == "certain"
        await _settle(engine)

        await engine._on_stream_final(
            "browser_mixed", "And I turn into a Crownsguard", 3.0, 4.0)
        await engine._on_stream_final(
            "browser_mixed", "Disguise Self and follow suit", 5.0, 6.0)
        await _settle(engine)

        a_calls = rank.for_block(block_a["id"])
        assert len(a_calls) >= 2, "A must be scored again after B/C arrive"
        assert any("Crownsguard" in c["state"] for c in a_calls[1:]), \
            "later passes must see the new following context"
        assert all("TARGET" in c["state"] for c in a_calls)

        revs = [r for r in _revisions(published) if r["id"] == block_a["id"]]
        assert revs, "same UI row must be revised by stable ID"
        final = revs[-1]["attribution"]
        assert revs[-1]["user_id"] == "travis", "later evidence corrects A"
        history = final["score_history"]
        assert len(history) >= 2
        assert history[0]["winner"] != history[-1]["winner"]
        assert history[0]["scores"] != history[-1]["scores"]

        assert len(_transcripts(published)) == 3, "no duplicate transcript lines"
        assert not [e for e in published if e["type"] == "card"], "no cards"
    finally:
        await engine.aclose()


async def test_low_score_then_resolution(monkeypatch: Any) -> None:
    """A low first distribution is preserved, then a later pass resolves it."""
    tracker = SpeakingTracker()
    _windows(tracker)
    published: list[dict] = []
    engine = _engine(published, tracker)
    seen: list[str] = []
    block_a = {"id": ""}

    def script(call: dict[str, Any]) -> dict[str, float]:
        if _targets(call, block_a["id"]):
            seen.append(call["row_id"])
            if len(seen) == 1:
                return _dist(call, "member:laura", 0.40)
            return _dist(call, "member:travis", 0.90)
        return _dist(call, call["options"][0]["id"], 0.60)

    monkeypatch.setattr(engine._openjev_gate, "rank", _ScriptedRank(script))
    try:
        await engine._on_stream_final(
            "browser_mixed", "I cast Disguise Self", 1.0, 2.0)
        block_a["id"] = _transcripts(published)[0]["id"]
        await _settle(engine)
        await engine._on_stream_final(
            "browser_mixed", "And I turn into a Crownsguard", 3.0, 4.0)
        await _settle(engine)

        revs = [r for r in _revisions(published) if r["id"] == block_a["id"]]
        assert len(seen) >= 2
        assert revs and revs[-1]["user_id"] == "travis"
        history = revs[-1]["attribution"]["score_history"]
        assert len(history) >= 2
        assert history[0]["score"] == 0.40, "low distribution preserved"
        assert history[-1]["score"] == 0.90
        assert revs[-1]["attribution"]["low_confidence"] is False
    finally:
        await engine.aclose()


async def test_confidence_reduction_and_oscillation(monkeypatch: Any) -> None:
    """Scores may fall and the winner may flip back: all of it is kept."""
    tracker = SpeakingTracker()
    _windows(tracker)
    published: list[dict] = []
    engine = _engine(published, tracker, {"initial_reviews": 3})
    plan = [("member:laura", 0.95), ("member:travis", 0.70),
            ("member:laura", 0.80)]
    count = {"n": 0}
    block_a = {"id": ""}

    def script(call: dict[str, Any]) -> dict[str, float]:
        if _targets(call, block_a["id"]):
            winner, score = plan[min(count["n"], len(plan) - 1)]
            count["n"] += 1
            return _dist(call, winner, score)
        return _dist(call, call["options"][0]["id"], 0.60)

    monkeypatch.setattr(engine._openjev_gate, "rank", _ScriptedRank(script))
    try:
        await engine._on_stream_final(
            "browser_mixed", "I cast Disguise Self", 1.0, 2.0)
        block_a["id"] = _transcripts(published)[0]["id"]
        await _settle(engine)
        # Settle between finals: without it, pending work coalesces to the
        # newest context and separate passes collapse into one (by design).
        await engine._on_stream_final("browser_mixed", "filler one", 3.0, 4.0)
        await _settle(engine)
        await engine._on_stream_final("browser_mixed", "filler two", 5.0, 6.0)
        await _settle(engine)

        revs = [r for r in _revisions(published) if r["id"] == block_a["id"]]
        assert count["n"] >= 3
        history = revs[-1]["attribution"]["score_history"]
        winners = [h["winner"] for h in history[-3:]]
        assert winners == ["member:laura", "member:travis", "member:laura"]
        assert history[-2]["score"] < history[-3]["score"], "reduction kept"
        assert revs[-1]["user_id"] == "laura", "oscillation lands on Laura"
    finally:
        await engine.aclose()


async def test_initial_pass_limit_and_old_stable_blocks_stop(
        monkeypatch: Any) -> None:
    """Bounded initial passes: an old, stable, high-confidence block stops."""
    tracker = SpeakingTracker()
    _windows(tracker)
    published: list[dict] = []
    engine = _engine(published, tracker,
                     {"initial_reviews": 1, "max_total_reviews": 1})
    rank = _ScriptedRank(
        lambda call: _dist(call, "member:laura", 0.95)
        if "member:laura" in [o["id"] for o in call["options"]]
        else _dist(call, call["options"][0]["id"], 0.95))
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final(
            "browser_mixed", "I cast Disguise Self", 1.0, 2.0)
        block_a = _transcripts(published)[0]["id"]
        for i, text in enumerate(["b", "c", "d", "e", "f"]):
            await engine._on_stream_final(
                "browser_mixed", text, 10.0 + i, 11.0 + i)
        await _settle(engine)
        assert len(rank.for_block(block_a)) == 1, "initial pass limit binds"
    finally:
        await engine.aclose()


async def test_older_uncertain_blocks_selectively_reconsidered(
        monkeypatch: Any) -> None:
    """After the initial phase only low/unstable blocks keep being revisited."""
    tracker = SpeakingTracker()
    # A is timing-ambiguous (near-tied overlap); B is timing-certain.
    tracker.on_speaking("laura", True, t=1.0, name="Laura")
    tracker.on_speaking("laura", False, t=1.7)
    tracker.on_speaking("travis", True, t=1.1, name="Travis")
    tracker.on_speaking("travis", False, t=1.8)
    tracker.on_speaking("liam", True, t=3.0, name="Liam")
    tracker.on_speaking("liam", False, t=5.0)
    published: list[dict] = []
    engine = _engine(published, tracker, {"initial_reviews": 1})
    block_ids: dict[str, str] = {}

    def script(call: dict[str, Any]) -> dict[str, float]:
        if block_ids.get("a") and block_ids["a"] in call["row_id"]:
            return _dist(call, "unknown", 0.70)
        return _dist(call, "member:liam", 0.95) \
            if "member:liam" in [o["id"] for o in call["options"]] \
            else _dist(call, call["options"][0]["id"], 0.60)

    rank = _ScriptedRank(script)
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final("browser_mixed", "uncertain one", 1.0, 2.0)
        block_ids["a"] = _transcripts(published)[0]["id"]
        await _settle(engine)
        await engine._on_stream_final("browser_mixed", "stable one", 3.0, 4.0)
        block_ids["b"] = _transcripts(published)[1]["id"]
        await _settle(engine)
        # Settle between finals so each new context drives its own pass.
        for i, text in enumerate(["c", "d"]):
            await engine._on_stream_final(
                "browser_mixed", text, 10.0 + i, 11.0 + i)
            await _settle(engine)
        a_calls = len(rank.for_block(block_ids["a"]))
        b_calls = len(rank.for_block(block_ids["b"]))
        assert a_calls > 1, "uncertain A is reconsidered on new context"
        assert b_calls == 1, "stable B stops after its initial pass"
    finally:
        await engine.aclose()


async def test_unchanged_context_does_not_retry(monkeypatch: Any) -> None:
    """Settling twice with no new finals must not rescore anything."""
    tracker = SpeakingTracker()
    _windows(tracker)
    published: list[dict] = []
    engine = _engine(published, tracker)
    rank = _ScriptedRank(lambda call: _dist(call, call["options"][0]["id"], 0.6))
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final("browser_mixed", "one", 1.0, 2.0)
        await engine._on_stream_final("browser_mixed", "two", 3.0, 4.0)
        await _settle(engine)
        calls = len(rank.calls)
        revs = len(_revisions(published))
        await _settle(engine)
        await asyncio.sleep(0.05)
        assert len(rank.calls) == calls, "no busy-loop rescoring"
        assert len(_revisions(published)) == revs
    finally:
        await engine.aclose()


async def test_busy_worker_new_context_does_not_drop_work(
        monkeypatch: Any) -> None:
    """Backpressure preserves reviews: a busy scorer must not drop blocks."""
    tracker = SpeakingTracker()
    # All three finals are timing-ambiguous so every block needs a review.
    for i, uid in enumerate(["laura", "travis", "liam"]):
        tracker.on_speaking(uid, True, t=1.0 + i * 10, name=uid.title())
        tracker.on_speaking(uid, False, t=1.7 + i * 10)
        other = ["laura", "travis", "liam"][(i + 1) % 3]
        tracker.on_speaking(other, True, t=1.1 + i * 10, name=other.title())
        tracker.on_speaking(other, False, t=1.8 + i * 10)
    published: list[dict] = []
    engine = _engine(published, tracker, {"max_pending": 1})
    gate_open = asyncio.Event()
    rank = _ScriptedRank([
        _blocking_first(gate_open),
        lambda call: _dist(call, call["options"][0]["id"], 0.6),
    ])
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final("browser_mixed", "alpha", 1.0, 2.0)
        for _ in range(100):
            if rank.calls:
                break
            await asyncio.sleep(0.01)
        await engine._on_stream_final("browser_mixed", "beta", 11.0, 12.0)
        await engine._on_stream_final("browser_mixed", "gamma", 21.0, 22.0)
        gate_open.set()
        await _settle(engine)
        ids = [t["id"] for t in _transcripts(published)]
        assert len(ids) == 3
        for block_id in ids:
            assert rank.for_block(block_id), f"{block_id} review was dropped"
    finally:
        await engine.aclose()


def _blocking_first(gate_open: asyncio.Event):  # type: ignore[no-untyped-def]
    """First scorer call waits for release; later calls answer at once."""
    state = {"first": True}

    async def _rank(row_id: str, st: str, q: str,
                    options: list[dict[str, str]]) -> dict[str, float]:
        if state["first"]:
            state["first"] = False
            await gate_open.wait()
        ids = [o["id"] for o in options]
        rest = 0.4 / max(len(ids) - 1, 1)
        return {i: (0.6 if i == ids[0] else rest) for i in ids}

    async def _call(call: dict[str, Any]) -> dict[str, float]:
        return await _rank(call["row_id"], call["state"], call["question"],
                           call["options"])

    return _call


async def test_stale_result_does_not_overwrite_newer_evidence(
        monkeypatch: Any) -> None:
    """A slow pass computed on old evidence must not beat a newer pass."""
    tracker = SpeakingTracker()
    published: list[dict] = []
    engine = _engine(published, tracker)
    gate_open = asyncio.Event()
    calls = {"n": 0}

    async def script(call: dict[str, Any]) -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] == 1:
            await gate_open.wait()
            return _dist(call, "member:laura", 0.90)
        return _dist(call, "member:travis", 0.80)

    tracker.on_speaking("laura", True, t=1.0, name="Laura")
    tracker.on_speaking("laura", False, t=2.0)
    tracker.on_speaking("travis", True, t=1.8, name="Travis")
    tracker.on_speaking("travis", False, t=2.0)
    rank = _ScriptedRank(script)
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final("browser_mixed", "mystery line", 1.0, 2.0)
        block_a = _transcripts(published)[0]["id"]
        for _ in range(100):
            if rank.calls:
                break
            await asyncio.sleep(0.01)
        # New timing evidence lands while the first pass is still in flight.
        tracker.on_speaking("laura", True, t=1.0, name="Laura")
        tracker.on_speaking("laura", False, t=2.0)
        await engine._on_stream_final("browser_mixed", "next line", 3.0, 4.0)
        gate_open.set()
        await _settle(engine)

        revs = [r for r in _revisions(published) if r["id"] == block_a]
        assert len(rank.for_block(block_a)) >= 2
        reasons = [h.get("reason") for h in revs[-1]["attribution"]["history"]]
        assert "jev_stale" in reasons, "stale pass recorded, not applied"
        assert revs[-1]["user_id"] == "travis", "fresh pass wins"
    finally:
        await engine.aclose()


async def test_candidate_gate_is_discord_target_speakers_only(monkeypatch: Any) -> None:
    """A later adjacent speaker never becomes a target's candidate."""
    tracker = SpeakingTracker()
    tracker.on_speaking("laura", True, t=1.0, name="Laura")
    tracker.on_speaking("laura", False, t=2.0)
    published: list[dict] = []
    engine = _engine(published, tracker)
    rank = _ScriptedRank(lambda call: _dist(call, call["options"][0]["id"], 0.6))
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final(
            "browser_mixed", "I cast Disguise Self", 1.0, 2.0)
        block_a = _transcripts(published)[0]["id"]
        await _settle(engine)
        first_ids = {o["id"] for o in rank.for_block(block_a)[0]["options"]}
        assert "member:travis" not in first_ids

        tracker.on_speaking("travis", True, t=2.5, name="Travis")
        tracker.on_speaking("travis", False, t=4.0)
        await engine._on_stream_final(
            "browser_mixed", "And I turn into a Crownsguard", 3.0, 4.0)
        await _settle(engine)

        later = rank.for_block(block_a)[1:]
        assert later, "A rescored after new context"
        assert all("member:travis" not in {o["id"] for o in c["options"]}
                   for c in later), "adjacent speaker leaked into candidates"
    finally:
        await engine.aclose()


async def test_unknown_and_multiple_drop_earlier_guess(monkeypatch: Any) -> None:
    """unknown/multiple outcomes revert to capture identity, never a guess."""
    tracker = SpeakingTracker()
    tracker.on_speaking("laura", True, t=1.0, name="Laura")
    tracker.on_speaking("laura", False, t=1.9)
    tracker.on_speaking("travis", True, t=1.1, name="Travis")
    tracker.on_speaking("travis", False, t=2.0)
    published: list[dict] = []
    engine = _engine(published, tracker, {"initial_reviews": 2})
    plan = ["unknown", "multiple"]
    count = {"n": 0}
    block_a = {"id": ""}

    def script(call: dict[str, Any]) -> dict[str, float]:
        if _targets(call, block_a["id"]):
            winner = plan[min(count["n"], len(plan) - 1)]
            count["n"] += 1
            return _dist(call, winner, 0.80)
        return _dist(call, call["options"][0]["id"], 0.60)

    monkeypatch.setattr(engine._openjev_gate, "rank", _ScriptedRank(script))
    try:
        await engine._on_stream_final("browser_mixed", "mystery line", 1.0, 2.0)
        block_a["id"] = _transcripts(published)[0]["id"]
        await _settle(engine)
        await engine._on_stream_final("browser_mixed", "filler", 3.0, 4.0)
        await _settle(engine)

        revs = [r for r in _revisions(published) if r["id"] == block_a["id"]]
        assert count["n"] >= 2
        assert revs[0]["attribution"]["state"] == "unknown"
        assert revs[0]["user_id"] == "browser_mixed"
        assert revs[-1]["attribution"]["state"] == "multiple"
        assert revs[-1]["user_id"] == "browser_mixed", \
            "no earlier guessed user retained as fallback"
        assert revs[-1]["attribution"]["score_history"][-1]["winner"] == \
            "multiple"
    finally:
        await engine.aclose()


async def test_invalid_scores_are_discarded(monkeypatch: Any) -> None:
    """Malformed distributions change nothing and invent no confidence."""
    tracker = SpeakingTracker()
    tracker.on_speaking("laura", True, t=1.0, name="Laura")
    tracker.on_speaking("laura", False, t=1.7)
    tracker.on_speaking("travis", True, t=1.1, name="Travis")
    tracker.on_speaking("travis", False, t=1.8)
    published: list[dict] = []
    engine = _engine(published, tracker)
    bad = [
        {"member:laura": float("nan"), "unknown": 0.5, "multiple": 0.5},
        {"member:laura": 1.5, "unknown": -0.5, "multiple": 0.0},
        {"member:laura": 1.0},  # incomplete: missing options
    ]
    rank = _ScriptedRank(bad)
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final("browser_mixed", "mystery", 1.0, 2.0)
        block_a = _transcripts(published)[0]["id"]
        initial = _transcripts(published)[0]
        await engine._on_stream_final("browser_mixed", "filler", 3.0, 4.0)
        await _settle(engine)

        assert rank.calls, "scoring was attempted, then discarded"
        assert _revisions(published) == [], "no revision from invalid scores"
        revs = [r for r in _revisions(published) if r["id"] == block_a]
        for rev in revs:
            scores = rev["attribution"].get("review_scores", {})
            for value in scores.values():
                assert math.isfinite(value) and 0.0 <= value <= 1.0
        assert all(r["user_id"] == initial["user_id"] for r in revs), \
            "invalid scores must not move identity"
    finally:
        await engine.aclose()


async def test_endpoint_failure_invents_no_confidence(monkeypatch: Any) -> None:
    """A dead scorer leaves identity and confidence untouched, without error."""

    async def _boom(*args: Any, **kwargs: Any) -> dict[str, float]:
        raise RuntimeError("scorer down")

    tracker = SpeakingTracker()
    tracker.on_speaking("laura", True, t=1.0, name="Laura")
    tracker.on_speaking("laura", False, t=1.7)
    tracker.on_speaking("travis", True, t=1.1, name="Travis")
    tracker.on_speaking("travis", False, t=1.8)
    published: list[dict] = []
    engine = _engine(published, tracker)
    monkeypatch.setattr(engine._openjev_gate, "rank", _boom)
    try:
        await engine._on_stream_final("browser_mixed", "mystery", 1.0, 2.0)
        await engine._on_stream_final("browser_mixed", "filler", 3.0, 4.0)
        await _settle(engine)  # must not raise
        first = _transcripts(published)[0]
        assert "review_scores" not in first["attribution"]
    finally:
        await engine.aclose()


async def test_scores_labeled_and_history_bounded(monkeypatch: Any) -> None:
    """JEV numbers are labeled uncalibrated; history stays bounded."""
    tracker = SpeakingTracker()
    _windows(tracker)
    published: list[dict] = []
    engine = _engine(published, tracker,
                     {"history_len": 3, "initial_reviews": 2,
                      "max_total_reviews": 6})
    flip = {"n": 0}
    block_a = {"id": ""}

    def script(call: dict[str, Any]) -> dict[str, float]:
        if _targets(call, block_a["id"]):
            flip["n"] += 1
            winner = "member:laura" if flip["n"] % 2 else "member:travis"
            return _dist(call, winner, 0.55)
        return _dist(call, call["options"][0]["id"], 0.60)

    monkeypatch.setattr(engine._openjev_gate, "rank", _ScriptedRank(script))
    try:
        await engine._on_stream_final(
            "browser_mixed", "I cast Disguise Self", 1.0, 2.0)
        block_a["id"] = _transcripts(published)[0]["id"]
        await _settle(engine)
        # Six passes total (2 initial + 4 unstable late-phase) against a
        # history bound of 3: retention must trim, not grow.
        for i, text in enumerate(["b", "c", "d", "e", "f"]):
            await engine._on_stream_final(
                "browser_mixed", text, 10.0 + i, 11.0 + i)
            await _settle(engine)

        revs = [r for r in _revisions(published) if r["id"] == block_a["id"]]
        assert flip["n"] == 6, "A keeps passing while unstable, then stops"
        final = revs[-1]["attribution"]
        assert final["review_scores_label"].startswith("uncalibrated")
        assert "not" in final["review_scores_label"] \
            and "probability" in final["review_scores_label"]
        assert len(final["score_history"]) == 3
        assert len(final["history"]) == 3
    finally:
        await engine.aclose()


async def test_no_duplicate_dispatch_cards_or_lines(monkeypatch: Any) -> None:
    """Revisions never re-enter dispatch: one final == one line, one handle."""

    async def _rank(row_id: str, state: str, question: str,
                    options: list[dict[str, str]]) -> dict[str, float]:
        ids = [o["id"] for o in options]
        rest = 0.2 / max(len(ids) - 1, 1)
        return {i: (0.8 if i == ids[0] else rest) for i in ids}

    tracker = SpeakingTracker()
    _windows(tracker)
    published: list[dict] = []
    engine = _engine(published, tracker)
    monkeypatch.setattr(engine._openjev_gate, "rank", _rank)
    handled: list[str] = []
    orig = engine.handle_utterance

    async def _counting(utter: Any) -> None:
        handled.append(utter.transcript_id or "")
        await orig(utter)

    monkeypatch.setattr(engine, "handle_utterance", _counting)
    try:
        await engine._on_stream_final("browser_mixed", "one", 1.0, 2.0)
        await engine._on_stream_final("browser_mixed", "two", 3.0, 4.0)
        await engine._on_stream_final("browser_mixed", "three", 5.0, 6.0)
        await _settle(engine)
        assert len(_transcripts(published)) == 3
        assert len(handled) == 3, "each final dispatched exactly once"
        known = {t["id"] for t in _transcripts(published)}
        assert all(r["id"] in known for r in _revisions(published))
        assert not [e for e in published if e["type"] == "card"]
    finally:
        await engine.aclose()


async def test_orderly_shutdown_with_reviews_in_flight(
        monkeypatch: Any) -> None:
    """Close during a slow pass cancels cleanly and leaves nothing pending."""

    async def _slow(*args: Any, **kwargs: Any) -> dict[str, float]:
        await asyncio.sleep(30.0)
        return {}

    tracker = SpeakingTracker()
    _windows(tracker)
    published: list[dict] = []
    engine = _engine(published, tracker)
    monkeypatch.setattr(engine._openjev_gate, "rank", _slow)
    await engine._on_stream_final("browser_mixed", "one", 1.0, 2.0)
    await asyncio.sleep(0.05)
    await engine.aclose()  # must not hang or raise
    reviewer = getattr(engine, "_reviewer", None)
    if reviewer is not None:
        assert reviewer.stats()["in_flight"] == 0
        assert reviewer.stats()["pending"] == 0
    await engine.aclose()  # idempotent


def test_review_config_validates_bounds() -> None:
    """Review bounds are validated; bad values fail config load."""
    try:
        _cfg({"min_prob": 1.5})
    except ConfigError:
        pass
    else:
        raise AssertionError("min_prob > 1 must be rejected")
    try:
        _cfg({"initial_reviews": 4, "max_total_reviews": 2})
    except ConfigError:
        pass
    else:
        raise AssertionError("max_total_reviews < initial_reviews rejected")
    try:
        _cfg({"max_pending": 0})
    except ConfigError:
        pass
    else:
        raise AssertionError("max_pending < 1 must be rejected")
    cfg = _cfg()
    assert cfg.attribution_review.initial_reviews >= 1
    assert cfg.attribution_review.max_total_reviews >= \
        cfg.attribution_review.initial_reviews


def test_continuation_notes_detect_openers_and_first_person() -> None:
    """Deterministic continuity signals fire on textual shape, not content."""
    from dmd.attribution_review import continuation_notes
    notes = continuation_notes(
        ("Matt", "You see two guards."),
        "I cast Disguise Self.",
        ("Travis", "And I turn into a Crownsguard."),
    )
    assert any("begins with 'And'" in n and "Travis" in n for n in notes)
    assert any("first person" in n for n in notes)
    closed = continuation_notes(
        ("A", "Hello there."), "Fine. Thanks.", ("B", "Okay. Done."))
    assert closed == [], "closed sentences with no opener carry no signal"


def test_timing_verdict_states_lead_tie_and_weakness() -> None:
    """The verdict states the conclusion; ties/weakness stay honest."""
    from dmd.attribution_review import timing_verdict
    names = {"sam": "Sam", "dm": "Matt"}
    clear = timing_verdict(
        [("sam", 1.1), ("dm", 0.35)], 0.75, 0.458, names.get)
    assert "Sam" in clear and "clearly leads" in clear
    assert "1.100s vs 0.350s" in clear
    tied = timing_verdict(
        [("sam", 0.9), ("dm", 0.8)], 0.75, 0.083, names.get)
    assert "nearly tied" in tied and "cannot decide" in tied
    weak = timing_verdict([("sam", 0.2)], 0.15, 0.15, names.get)
    assert "weak" in weak and "only briefly" in weak
    solo = timing_verdict([("sam", 1.1)], 0.9, 0.9, names.get)
    assert "only member" in solo
    assert "no member" in timing_verdict([], 0.0, 0.0, names.get)


async def test_continuation_signal_reaches_prompt(monkeypatch: Any) -> None:
    """A split sentence carries its continuity note into the review prompt.

    When the line after TARGET begins with a continuation opener, the
    prompt must state that textual observation with the adjacent line's
    speaker hypothesis — a single-pass scorer cannot be trusted to infer
    it. Timing evidence (coverage/margin) is quoted alongside so both
    signals can be weighed together.
    """
    tracker = SpeakingTracker()
    tracker.on_speaking("laura", True, t=1.0, name="Laura")
    tracker.on_speaking("laura", False, t=1.7)
    tracker.on_speaking("travis", True, t=1.1, name="Travis")
    tracker.on_speaking("travis", False, t=4.0)
    published: list[dict] = []
    engine = _engine(published, tracker)
    rank = _ScriptedRank(
        lambda call: _dist(call, call["options"][0]["id"], 0.6))
    monkeypatch.setattr(engine._openjev_gate, "rank", rank)
    try:
        await engine._on_stream_final(
            "browser_mixed", "I cast Disguise Self", 1.0, 2.0)
        block_a = _transcripts(published)[0]["id"]
        await _settle(engine)
        await engine._on_stream_final(
            "browser_mixed", "And I turn into a Crownsguard", 3.0, 4.0)
        await _settle(engine)

        later = [c for c in rank.for_block(block_a)
                 if "Crownsguard" in c["state"]]
        assert later, "A rescored with the following line visible"
        state = later[-1]["state"]
        assert "Dialogue continuity" in state
        assert "begins with 'And'" in state
        assert "Travis" in state.split("Dialogue continuity")[1]
        assert "coverage" in state and "margin" in state
        assert "Timing verdict" in state
        assert "continuity" in later[-1]["question"]
    finally:
        await engine.aclose()


async def test_review_window_config_bounds_prompt(monkeypatch: Any) -> None:
    """recent_n/following_n bound the prompt; null sends all retained lines."""
    for review, want_following in (({"recent_n": 1, "following_n": 0}, False),
                                   (None, True)):
        tracker = SpeakingTracker()
        tracker.on_speaking("laura", True, t=1.0, name="Laura")
        tracker.on_speaking("laura", False, t=2.0)
        tracker.on_speaking("travis", True, t=3.0, name="Travis")
        tracker.on_speaking("travis", False, t=4.0)
        tracker.on_speaking("liam", True, t=5.0, name="Liam")
        tracker.on_speaking("liam", False, t=6.0)
        published: list[dict] = []
        engine = _engine(published, tracker, review)
        rank = _ScriptedRank(
            lambda call: _dist(call, call["options"][0]["id"], 0.6))
        monkeypatch.setattr(engine._openjev_gate, "rank", rank)
        try:
            await engine._on_stream_final("browser_mixed", "alpha", 1.0, 2.0)
            await engine._on_stream_final("browser_mixed", "bravo", 3.0, 4.0)
            block_b = _transcripts(published)[1]["id"]
            await _settle(engine)
            await engine._on_stream_final(
                "browser_mixed", "charlie", 5.0, 6.0)
            await _settle(engine)

            rescored = [c for c in rank.for_block(block_b)
                        if "charlie" in c["state"]]
            if want_following:
                assert rescored, "default window shows following lines"
                assert "alpha" in rescored[-1]["state"]
            else:
                assert not rescored, "following_n=0 hides later lines"
                first = rank.for_block(block_b)[0]["state"]
                assert "alpha" in first, "recent_n=1 keeps one preceding"
        finally:
            await engine.aclose()
