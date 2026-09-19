"""Tests for dmd.enrich — FakeGateway-driven entity extraction."""
from __future__ import annotations

import pytest

from dmd.enrich import _merge, extract_entities
from dmd.scanner import DocFile
from dmd.types import Entity


class FakeGateway:
    """Minimal Gateway stand-in: queue of canned chat responses, optional raise."""

    def __init__(
        self,
        *,
        responses: list | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.raise_exc = raise_exc
        self.calls = 0
        self.roles_seen: list[str] = []

    async def chat(self, role: str, messages, **kwargs) -> dict | str:
        self.calls += 1
        self.roles_seen.append(role)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.responses:
            return self.responses.pop(0)
        return {}


def _doc(path: str, content: str = "Some content.") -> DocFile:
    return DocFile(
        relpath=path,
        title=path,
        frontmatter={},
        links=[],
        tags=[],
        headings=[],
        content=content,
        mtime=0.0,
    )


def test_merge_alias_union_weight_max_source_files_union() -> None:
    """_merge: aliases unionized (deduped case-insensitive), weight = max, sources unionized."""
    by_key: dict[str, Entity] = {}
    _merge(by_key, "Aragorn", ["Strider"], "character", 0.5, ["a.md"])
    # Second insert: new alias "Elessar", repeat "Strider", higher weight, source b.md; a.md repeated.
    _merge(
        by_key,
        "Aragorn",
        ["Elessar", "Strider", "STRIDER"],
        "character",
        0.9,
        ["b.md", "a.md"],
    )
    # Third insert: lower weight, new source c.md, no new aliases.
    _merge(by_key, "aragorn", [], "character", 0.3, ["c.md"])

    e = by_key["aragorn"]
    assert e.canonical == "Aragorn"
    assert set(e.aliases) == {"Strider", "Elessar"}  # case-insensitive dedupe
    assert e.weight == 0.9  # max wins
    assert sorted(e.source_files) == ["a.md", "b.md", "c.md"]


@pytest.mark.asyncio
async def test_extract_entities_merges_across_batches() -> None:
    """extract_entities merges entity records across batches: aliases union, weight max,
    source_files union."""
    gw = FakeGateway(responses=[
        {
            "entities": [
                {"canonical": "Gandalf", "aliases": ["Mithrandir"], "etype": "character", "weight": 0.7},
                {"canonical": "Mordor", "aliases": [], "etype": "place", "weight": 0.9},
            ],
        },
        {
            "entities": [
                {"canonical": "Gandalf", "aliases": ["the Grey"], "etype": "character", "weight": 0.5},
                {"canonical": "Frodo", "aliases": [], "etype": "character", "weight": 0.8},
            ],
        },
    ])
    docs = [_doc("a.md", "x" * 5000), _doc("b.md", "y" * 5000), _doc("c.md", "z" * 5000)]
    result = await extract_entities(docs, gw, batch_docs=1)

    by_canon = {e.canonical: e for e in result}
    assert set(by_canon) == {"Gandalf", "Mordor", "Frodo"}
    # Merged across two batches (a.md and b.md)
    g = by_canon["Gandalf"]
    assert set(g.aliases) == {"Mithrandir", "the Grey"}
    assert g.weight == 0.7
    assert sorted(g.source_files) == ["a.md", "b.md"]
    # Sort is by -weight: Mordor (0.9) > Frodo (0.8) > Gandalf (0.7)
    assert [e.canonical for e in result[:3]] == ["Mordor", "Frodo", "Gandalf"]


@pytest.mark.asyncio
async def test_extract_entities_caps_at_500_by_weight() -> None:
    """Cap is [:500] ordered by descending weight."""
    ents = [
        {"canonical": f"e{i}", "aliases": [], "etype": "character", "weight": float(i)}
        for i in range(600)
    ]
    gw = FakeGateway(responses=[{"entities": ents}])
    docs = [_doc("a.md", "x" * 5000)]
    result = await extract_entities(docs, gw)
    assert len(result) == 500
    # Top by weight: e599 first, e100 last (slice [:500] of 0..599)
    assert result[0].canonical == "e599"
    assert result[-1].canonical == "e100"
    # Weights strictly descending through the result list
    weights = [e.weight for e in result]
    assert weights == sorted(weights, reverse=True)


@pytest.mark.asyncio
async def test_extract_entities_skips_when_chat_returns_non_dict() -> None:
    """Defensive: a chat result that isn't a dict is silently skipped for that batch."""
    gw = FakeGateway(responses=[
        "just a string",
        {"entities": [{"canonical": "X", "weight": 1.0, "etype": "character"}]},
    ])
    docs = [_doc("a.md", "x" * 5000), _doc("b.md", "y" * 5000)]
    result = await extract_entities(docs, gw, batch_docs=1)
    canon = {e.canonical for e in result}
    # Only X (from the dict batch) survives; the string batch is dropped.
    assert canon == {"X"}
    assert gw.calls == 2


@pytest.mark.asyncio
async def test_extract_entities_raises_when_all_batches_fail() -> None:
    """Total gateway failure raises so init_pass can record a warning.

    Regression: this used to assert silent swallowing (return []), which made
    init_pass's warning path dead code. The fixed contract is raise-on-total-failure.
    """
    gw = FakeGateway(raise_exc=RuntimeError("boom"))
    docs = [_doc("a.md", "x" * 5000)]
    try:
        await extract_entities(docs, gw, batch_docs=1)
        raised = False
    except RuntimeError as e:
        raised = True
        assert "all 1/1 batches" in str(e)
    assert raised
    assert gw.calls == 1


@pytest.mark.asyncio
async def test_extract_entities_partial_failure_returns_partial() -> None:
    """Some batches succeeding still returns their entities without raising."""
    docs = [_doc(f"d{i}.md", "x" * 5000) for i in range(2)]
    responses = [
        {"entities": [{"canonical": "Kept", "etype": "character", "weight": 1.0}]},
        RuntimeError("boom"),
    ]
    gw = FakeGateway(responses=responses)
    result = await extract_entities(docs, gw, batch_docs=1)
    assert [e.canonical for e in result] == ["Kept"]
    assert gw.calls == 2


@pytest.mark.asyncio
async def test_extract_entities_etype_unknown_falls_back() -> None:
    """Invalid or missing etype values fall back to 'unknown'."""
    gw = FakeGateway(responses=[
        {
            "entities": [
                {"canonical": "A", "etype": "character", "weight": 1.0},
                {"canonical": "B", "etype": "wat", "weight": 1.0},      # unknown -> "unknown"
                {"canonical": "C", "weight": 1.0},                       # missing etype
            ],
        },
    ])
    docs = [_doc("a.md", "x" * 5000)]
    result = await extract_entities(docs, gw)
    by_canon = {e.canonical: e for e in result}
    assert by_canon["A"].etype == "character"
    assert by_canon["B"].etype == "unknown"
    assert by_canon["C"].etype == "unknown"