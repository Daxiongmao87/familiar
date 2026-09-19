"""Predictive-retrieval staging ("Predictive RAG", Priority-1 design).

The transcript monitor predicts likely-next entities; their campaign excerpts
are pre-fetched into a small RAM LRU (``dmd/staging.StagedContext``) so an
actual turn can pull already-embedded context instead of paying retrieval
latency inline. Doctrine under test: staged data is advisory only (never
mutates canonical state), and a cache miss must be a no-op that adds zero
latency to the normal path.

Tests prove:
  * the LRU+TTL cache contract (eviction, expiry, hit/miss counters, lookup
    tagging/dedup, empty-never-cached);
  * ``render_staged_block`` caps output and renders nothing for no excerpts;
  * the monitor's judge schema/system now carry ``predicted_entities`` and
    forward them to ``on_predict`` even when ``action='none'`` (without firing
    ``on_action``);
  * the full prediction -> prefetch -> staging -> injection round-trip at the
    engine level, with the injected block reaching the worker agent and the
    card recording ``staged_for``;
  * a cache miss does not block: no embedder/search is touched and the agent
    still runs on the byte-identical normal path.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from dmd.agent import AgentResult
from dmd.config import load_config_dict
from dmd.monitor import TranscriptMonitor, _MONITOR_SCHEMA, _MONITOR_SYSTEM, _predicted_entities
from dmd.pipeline import SessionEngine
from dmd.staging import StagedContext, normalize_key, render_staged_block


# -- cache contract ----------------------------------------------------------


def test_lru_eviction_keeps_most_recent() -> None:
    cache = StagedContext(ttl_s=0, max_entries=2)
    cache.put("A", [{"source": "a.md", "excerpt": "aaa"}])
    cache.put("B", [{"source": "b.md", "excerpt": "bbb"}])
    cache.put("C", [{"source": "c.md", "excerpt": "ccc"}])  # evicts A (LRU)
    assert cache.has("B") and cache.has("C") and not cache.has("A")
    assert cache.snapshot()["size"] == 2


def test_ttl_expiry_turns_hits_into_misses() -> None:
    cache = StagedContext(ttl_s=100, max_entries=8)
    cache.put("Kael", [{"source": "npc.md", "excerpt": "Kael sheet"}])
    assert cache.has("Kael")
    entry = cache._entries[normalize_key("Kael")]
    entry.t_staged -= 101.0  # age it past the TTL
    assert cache.has("Kael") is False
    assert cache.get("Kael") is None  # expired read is a miss, not a raise


def test_empty_excerpt_list_is_never_cached() -> None:
    cache = StagedContext(ttl_s=0, max_entries=8)
    cache.put("Mira", [])
    cache.put("   ", [{"source": "x", "excerpt": "x"}])
    assert cache.snapshot()["size"] == 0


def test_hit_miss_counters_and_lookup_tagging() -> None:
    cache = StagedContext(ttl_s=0, max_entries=8)
    cache.put("Brother Alric", [{"source": "npc.md", "excerpt": "a confession pouch"}])
    excerpts, matched = cache.lookup(["Brother Alric", "Unseen Thing", "Brother Alric"])
    assert matched == ["Brother Alric"]  # dedup: one entry per canonical entity
    assert excerpts and all(e["staged_for"] == "Brother Alric" for e in excerpts)
    assert cache.hits == 1 and cache.misses == 1  # the unknown entity counted once
    snap = cache.snapshot()
    assert snap["size"] == 1 and snap["keys"] == ["brother alric"]


def test_render_staged_block_caps_and_empties() -> None:
    long_excerpts = [
        {"source": "s.md", "staged_for": "E", "excerpt": "x" * 2000} for _ in range(10)
    ]
    block = render_staged_block(long_excerpts, max_chars=500)
    assert block and len(block) <= 500
    assert render_staged_block([]) == ""
    assert render_staged_block([{"source": "s", "excerpt": "   "}]) == ""


# -- monitor prediction forwarding -------------------------------------------


class _FakeGw:
    def __init__(self, verdict: dict[str, Any]) -> None:
        self._verdict = verdict

    async def chat(self, role: str, messages: list, **kw: Any) -> Any:
        return self._verdict


def _monitor(gw: Any, actions: list, predicts: list) -> TranscriptMonitor:
    async def _act(verdict: dict) -> None:
        actions.append(verdict)

    async def _pred(entities: list[str]) -> None:
        predicts.append(entities)

    return TranscriptMonitor(
        gw,
        object(),
        get_transcript=lambda: "the party loots the corpse " * 4,
        get_scene=lambda: "chapel",
        on_action=_act,
        on_predict=_pred,
    )


async def test_monitor_forwards_predictions_on_action_none() -> None:
    verdict = {
        "action": "none",
        "situation": "combat ended",
        "predicted_entities": ["Brother Alric", "Confession Pouch"],
        "likely_next_events": ["loot the body"],
    }
    actions: list[dict] = []
    predicts: list[list[str]] = []
    mon = _monitor(_FakeGw(verdict), actions, predicts)
    await mon.tick_once()
    assert predicts == [["Brother Alric", "Confession Pouch"]]
    assert actions == [], "action='none' must not fire on_action"
    assert _predicted_entities({"predicted_entities": "nope"}) == []


async def test_monitor_surface_also_forwards_predictions() -> None:
    verdict = {"action": "surface", "tier": "card", "reason": "loot the corpse",
               "predicted_entities": ["Brother Alric"]}
    actions: list[dict] = []
    predicts: list[list[str]] = []
    mon = _monitor(_FakeGw(verdict), actions, predicts)
    await mon.tick_once()
    assert predicts == [["Brother Alric"]]
    assert actions == [verdict]


def test_monitor_schema_and_system_carry_prediction_fields() -> None:
    props = _MONITOR_SCHEMA["properties"]
    assert "situation" in props and "predicted_entities" in props and "likely_next_events" in props
    assert "predicted_entities" in _MONITOR_SYSTEM and "situation=" in _MONITOR_SYSTEM
    # Regression (verified live 2026-09-05): a tiny model skips optional schema
    # fields, so the prediction fields must be REQUIRED or staging never fires.
    assert {"action", "situation", "predicted_entities", "likely_next_events"} <= set(
        _MONITOR_SCHEMA["required"]
    )


# -- engine round trip: prediction -> prefetch -> staging -> injection --------


class _HitStore:
    """Fake retrieval store: returns a fixed campaign excerpt for any query."""

    def __init__(self, text: str = "Brother Alric kept a waxed confession pouch under the altar.") -> None:
        self.text = text
        self.search_calls = 0

    def search(self, embedding: Any = None, query_text: str = "", k: int = 8) -> list[Any]:
        self.search_calls += 1
        from dmd.types import Retrieved

        return [Retrieved(doc_id="d1", source="npcs/brother_alric.md", score=0.9, text=self.text)]


class _VecEmbedder:
    """Fake local embedder that records use (never actually embeds)."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embed_calls += 1
        return np.zeros((len(texts), 8), dtype=np.float32)


class _NoPool:
    async def submit(self, job: Any, work: Any) -> None:
        return None

    async def drain(self) -> None:
        return None


class _RecordingAgent:
    """Replaces engine._agent: records every run() call, returns a canned card."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, task: str, tier: str, **kw: Any) -> AgentResult:
        self.calls.append({"task": task, "tier": tier, **kw})
        return AgentResult(
            tier="card",
            card={
                "kind": "loot",
                "title": "Loot table",
                "body_md": "The pouch holds 17 gp.",
                "player_ids": [],
                "items": [],
            },
        )


def _engine(tmp_path: Any, embedder: Any, store: Any, gw: Any = None, events: list | None = None) -> SessionEngine:
    cfg = load_config_dict(
        {
            "project": {"path": str(tmp_path)},
            "models": {
                "synthesis": {"base_url": "http://fake.invalid", "model_id": "m"},
                "stt": {"base_url": "http://fake.invalid"},
            },
        }
    )
    return SessionEngine(
        cfg=cfg,
        store=store,
        gw=gw,
        entries=[],
        embedder=embedder,
        pool=_NoPool(),
        on_event=(events.append if events is not None else (lambda e: None)),
        project_path=str(tmp_path),
    )


async def test_prediction_to_injection_round_trip(tmp_path: Any) -> None:
    store = _HitStore()
    embedder = _VecEmbedder()
    events: list[dict] = []
    engine = _engine(tmp_path, embedder, store, gw=object(), events=events)
    stub = _RecordingAgent()
    engine._agent = stub  # type: ignore[assignment]

    # 1) prediction tick -> engine schedules background prefetch for the entity
    await engine._on_predict(["Brother Alric"])
    deadline = time.monotonic() + 5.0
    while engine._prefetch_tasks and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert engine._staged is not None and engine._staged.has("Brother Alric"), "prefetch did not stage the entity"
    assert store.search_calls >= 1 and embedder.embed_calls >= 1
    assert any(e["type"] == "staging_predict" for e in events)

    # 2) the real turn arrives mentioning the predicted entity -> injected
    ctx = {
        "utterance": "We loot Brother Alric's body",
        "entities": ["Brother Alric"],
        "tier": "card",
        "recent": [],
    }
    card = await engine._generate_card(ctx)
    assert card is not None
    run_kwargs = stub.calls[-1]
    assert "confession pouch" in run_kwargs["staged_block"], "staged excerpts not injected into the agent"
    assert card.meta.get("staged_for") == ["Brother Alric"]
    assert engine._staged.hits >= 1
    await engine.aclose()


async def test_cache_miss_does_not_block_normal_path(tmp_path: Any) -> None:
    store = _HitStore()
    embedder = _VecEmbedder()
    engine = _engine(tmp_path, embedder, store, gw=object())
    stub = _RecordingAgent()
    engine._agent = stub  # type: ignore[assignment]

    # Fresh engine: the cache is empty, so a trigger naming entities must NOT
    # touch the embedder or the store, must hand the agent an empty staged
    # block, and must still produce a card on the normal path.
    search_calls_before = store.search_calls
    embed_calls_before = embedder.embed_calls
    ctx = {"utterance": "We loot his body", "entities": ["Brother Alric"], "tier": "card", "recent": []}
    card = await engine._generate_card(ctx)
    assert card is not None and card.status == "active"
    assert stub.calls[-1]["staged_block"] == ""
    assert store.search_calls == search_calls_before, "cache miss triggered a retrieval on the answer path"
    assert embedder.embed_calls == embed_calls_before, "cache miss triggered an embed on the answer path"
    assert "staged_for" not in card.meta
    await engine.aclose()


async def test_staging_disabled_by_config_yields_no_cache(tmp_path: Any) -> None:
    cfg = load_config_dict(
        {
            "project": {"path": str(tmp_path)},
            "models": {
                "synthesis": {"base_url": "http://fake.invalid", "model_id": "m"},
                "stt": {"base_url": "http://fake.invalid"},
            },
            "staging": {"enabled": False},
        }
    )
    engine = SessionEngine(
        cfg=cfg,
        store=None,
        gw=object(),
        entries=[],
        embedder=None,
        pool=_NoPool(),
        on_event=lambda e: None,
        project_path=str(tmp_path),
    )
    assert engine._staged is None
    await engine.aclose()
