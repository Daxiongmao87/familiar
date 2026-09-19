"""Evidence-judged worker: terms, both legs, relevance, synthesize."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

from dmd.agent import AgentResult
from dmd.config import AgentConfig
from dmd.jevworker import JevWorker
from dmd.openjev import OpenjevError, RouteVerdict
from dmd.types import LexiconEntry


class _FakeGw:
    """Synthesis only: the new path makes zero LLM term calls."""

    def __init__(self) -> None:
        self.chats: list[str] = []

    async def chat(self, role: str, messages: list[dict], **kw: Any) -> str:
        tail = messages[-1]["content"] if messages else ""
        self.chats.append(tail)
        return json.dumps(
            {"kind": "ruling", "title": "T", "body_md": "B", "player_ids": [], "items": []}
        )


class _FakeStore:
    def __init__(self, hits: list | None = None) -> None:
        self.hits = hits if hits is not None else [
            SimpleNamespace(source="notes.md", score=0.9, text="excerpt")
        ]
        self.queries: list[str] = []

    def search(self, embedding: Any = None, query_text: str = "", k: int = 6) -> list:
        self.queries.append(query_text)
        return self.hits


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


class _FakeGate:
    def __init__(self, route: str = "both", rank_error: bool = False) -> None:
        self._route = route
        self._rank_error = rank_error
        self.ranks = 0
        self.rels = 0

    async def rank(self, row_id: str, state: str, question: str,
                   options: list[dict]) -> dict[str, float]:
        self.ranks += 1
        if self._rank_error:
            raise OpenjevError("rank down")
        n = len(options)
        return {o["id"]: (0.9 if i == 0 else 0.1 / max(n - 1, 1))
                for i, o in enumerate(options)}

    async def relevance(self, state: str) -> RouteVerdict:
        self.rels += 1
        return RouteVerdict(self._route, {}, 0.0)


def _worker(gw=None, store=None, gate=None, entries=None, **kw) -> JevWorker:
    w = JevWorker(
        gw or _FakeGw(),
        store or _FakeStore(),
        "/tmp/campaign",
        AgentConfig(),
        gate=gate or _FakeGate(),
        embedder=_FakeEmbedder(),
        entries=entries or [],
        **kw,
    )
    w._tool_web_search = _web_ok.__get__(w)  # type: ignore[method-assign]
    return w


async def _web_ok(self, args: dict) -> dict:
    _web_ok.calls.append(args.get("query", ""))
    return {"results": [{"title": "SRD", "url": "http://x", "snippet": "rule text"}]}


_web_ok.calls: list[str] = []  # type: ignore[attr-defined]

TRANSCRIPT = "dm: Make a deception check.\nsam: I got a fourteen."


async def test_run_searches_both_legs_per_term_and_synthesizes() -> None:
    _web_ok.calls.clear()  # type: ignore[attr-defined]
    store = _FakeStore()
    gate = _FakeGate(route="both")
    w = _worker(store=store, gate=gate)
    res: AgentResult = await w.run("task", "card", trigger_portion="Make a check",
                                   transcript=TRANSCRIPT)
    assert res.error is None
    assert res.card is not None and res.card["title"] == "T"
    assert gate.ranks == 1
    assert gate.rels == 1
    assert len(store.queries) >= 1  # offline leg ran per term
    assert len(_web_ok.calls) >= 1  # type: ignore[attr-defined]  # online leg ran
    assert store.queries == _web_ok.calls  # type: ignore[attr-defined]  # same terms


async def test_no_llm_term_calls() -> None:
    gw = _FakeGw()
    w = _worker(gw=gw)
    await w.run("task", "card", trigger_portion="need", transcript=TRANSCRIPT)
    assert gw.chats, "synthesis must still run"
    assert all("Branch:" not in c for c in gw.chats)


async def test_offline_relevance_filters_online_evidence() -> None:
    gw = _FakeGw()
    w = _worker(gw=gw, gate=_FakeGate(route="offline"))
    res = await w.run("task", "card", trigger_portion="need", transcript=TRANSCRIPT)
    assert res.error is None
    synth = gw.chats[-1]
    assert "EVIDENCE (offline)" in synth
    assert "EVIDENCE (online)" not in synth


async def test_online_relevance_filters_offline_evidence() -> None:
    gw = _FakeGw()
    w = _worker(gw=gw, gate=_FakeGate(route="online"))
    res = await w.run("task", "card", trigger_portion="need", transcript=TRANSCRIPT)
    assert res.error is None
    synth = gw.chats[-1]
    assert "EVIDENCE (online)" in synth
    assert "EVIDENCE (offline)" not in synth


async def test_rank_failure_fails_open_to_collected_terms() -> None:
    _web_ok.calls.clear()  # type: ignore[attr-defined]
    store = _FakeStore()
    w = _worker(store=store, gate=_FakeGate(route="both", rank_error=True))
    res = await w.run("task", "card", trigger_portion="need", transcript=TRANSCRIPT)
    assert res.error is None
    assert len(store.queries) >= 1  # retrieval still ran
    assert len(_web_ok.calls) >= 1  # type: ignore[attr-defined]


async def test_lexicon_spans_join_collected_terms() -> None:
    store = _FakeStore()
    entries = [LexiconEntry(canonical="Nott", variants=["Nott"],
                            etype="character", weight=1.0)]
    w = _worker(store=store, entries=entries)
    res = await w.run("task", "card", trigger_portion="slam Nott",
                      transcript="liam: I take Nott and slam her into a wall.")
    assert res.error is None
    assert any("nott" in q.lower() for q in store.queries)


async def test_empty_transcript_falls_back_to_need() -> None:
    store = _FakeStore()
    w = _worker(store=store)
    res = await w.run("task", "card", trigger_portion="raw need text", transcript="  ")
    assert res.error is None
    assert any("raw need text" in q for q in store.queries)


async def test_ephemeral_tier_returns_structured_text() -> None:
    w = _worker()
    res = await w.run("task", "ephemeral", trigger_portion="need", transcript=TRANSCRIPT)
    assert res.error is None
    assert res.text is not None and res.text.startswith("**T**")
    assert "B" in res.text


async def test_both_legs_run_concurrently_not_sequentially() -> None:
    """Slow legs overlap: 0.5 s retrieve + 0.5 s web must cost ~0.5 s,
    not ~1.0 s (regression: the per-term loop awaited each leg in turn)."""
    w = _worker()

    async def slow_retrieve(args: dict) -> dict:
        await asyncio.sleep(0.5)
        return {"results": []}

    async def slow_web(args: dict) -> dict:
        await asyncio.sleep(0.5)
        return {"results": []}

    w._tool_retrieve = slow_retrieve  # type: ignore[method-assign]
    w._tool_web_search = slow_web  # type: ignore[method-assign]
    t0 = time.monotonic()
    res = await w.run("task", "card", trigger_portion="need", transcript=TRANSCRIPT)
    dt = time.monotonic() - t0
    assert res.error is None
    assert res.card is not None
    assert dt < 0.9, f"legs ran sequentially ({dt:.2f}s for one term x two legs)"
