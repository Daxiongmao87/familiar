"""Unit tests for dmd.index_store — SQLite + sqlite-vec, no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmd.index_store import IndexStore
from dmd.scanner import DocFile, chunk_docs
from dmd.types import Entity


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path):
    db = tmp_path / "test.db"
    s = IndexStore(str(db))
    yield s
    s.close()


def _doc(
    relpath: str,
    content: str,
    *,
    title: str = "",
    frontmatter: dict | None = None,
    mtime: float = 1.0,
) -> DocFile:
    return DocFile(
        relpath=relpath,
        title=title or relpath,
        frontmatter=frontmatter or {},
        links=[],
        tags=[],
        headings=[],
        content=content,
        mtime=mtime,
    )


# ---------------------------------------------------------------------------
# Schema + persistence basics
# ---------------------------------------------------------------------------


def test_schema_init_is_idempotent_on_reopen(tmp_path: Path):
    db = tmp_path / "test.db"

    s1 = IndexStore(str(db))
    s1.set_meta("hello", "world")
    s1.close()

    # Reopening must not throw (everything is IF NOT EXISTS).
    s2 = IndexStore(str(db))
    try:
        assert s2.get_meta("hello") == "world"
    finally:
        s2.close()


def test_upsert_docs_and_replace_chunks_roundtrip(store: IndexStore):
    doc = _doc(
        "lore/keep.md",
        "## Heading\nThis is some searchable lore content here.\n",
        title="Keep",
        frontmatter={"kind": "lore"},
    )
    store.upsert_docs([doc])

    chunks = chunk_docs([doc])
    assert len(chunks) >= 1
    store.replace_chunks_for_doc(doc.relpath, chunks)

    # Roundtrip via the public search API — the inserted chunk text is
    # reachable through FTS.
    results = store.search(query_text="searchable")
    assert any(r.source == "lore/keep.md" for r in results)

    # Counts reflect both inserts.
    counts = store.counts()
    assert counts["docs"] == 1
    assert counts["chunks"] >= 1


def test_stale_docs_detects_mtime_drift_and_new_path(store: IndexStore):
    stored = _doc("x.md", "first content body", mtime=1.0)
    store.upsert_docs([stored])

    # Same mtime -> not stale.
    assert store.stale_docs({"x.md": 1.0}) == []

    # Drift > 1e-6 -> stale.
    drifted = store.stale_docs({"x.md": 1.5})
    assert drifted == ["x.md"]

    # Path present in current but missing from store -> stale (new file).
    fresh = store.stale_docs({"x.md": 1.0, "new.md": 1.0})
    assert fresh == ["new.md"]


def test_remove_doc_cascades_and_search_returns_nothing(store: IndexStore):
    unique = "xyzmarkersearchterm"
    doc = _doc("drop.md", f"## Section\n{unique} appears here only.\n")
    store.upsert_docs([doc])
    chunks = chunk_docs([doc])
    store.replace_chunks_for_doc(doc.relpath, chunks)

    # Before removal, the unique term is searchable.
    before = store.search(query_text=unique)
    assert len(before) > 0
    assert any(r.source == "drop.md" for r in before)

    store.remove_doc("drop.md")

    # After removal, neither the docs table nor FTS retains the doc.
    assert "drop.md" not in store.list_paths()
    after = store.search(query_text=unique)
    assert after == []
    assert store.counts()["docs"] == 0
    assert store.counts()["chunks"] == 0


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_upsert_entities_get_entity_all_entities_roundtrip(store: IndexStore):
    e1 = Entity(
        canonical="Alice",
        aliases=["Alicia", "Al"],
        etype="character",
        weight=1.5,
        source_files=["a.md", "b.md"],
    )
    e2 = Entity(
        canonical="Bob",
        aliases=[],
        etype="character",
        weight=1.0,
        source_files=[],
    )
    store.upsert_entities([e1, e2])

    got = store.get_entity("Alice")
    assert got is not None
    assert got.canonical == "Alice"
    assert got.aliases == ["Alicia", "Al"]
    assert got.etype == "character"
    assert got.weight == 1.5
    assert got.source_files == ["a.md", "b.md"]

    # Aliases round-trip as JSON, not as a literal Python repr.
    assert isinstance(got.aliases, list)
    assert isinstance(got.source_files, list)

    all_e = store.all_entities()
    assert {e.canonical for e in all_e} == {"Alice", "Bob"}
    assert all_e == sorted(all_e, key=lambda e: e.canonical)


def test_set_meta_and_get_meta_public_api(store: IndexStore):
    assert store.get_meta("anything") is None
    store.set_meta("model_id", "bge-small")
    assert store.get_meta("model_id") == "bge-small"
    # Upsert semantics: overwriting works.
    store.set_meta("model_id", "other-model")
    assert store.get_meta("model_id") == "other-model"


def test_counts_reflects_inserts(store: IndexStore):
    initial = store.counts()
    assert initial == {"docs": 0, "chunks": 0, "entities": 0}

    doc = _doc("x.md", "## A\n" + ("a" * 250) + "\n## B\n" + ("b" * 250) + "\n")
    store.upsert_docs([doc])
    chunks = chunk_docs([doc])
    store.replace_chunks_for_doc(doc.relpath, chunks)
    store.upsert_entities(
        [Entity(canonical="E1", aliases=[], etype="concept", weight=1.0, source_files=[])]
    )

    after = store.counts()
    assert after["docs"] == 1
    assert after["chunks"] >= 2
    assert after["entities"] == 1


def test_list_paths(store: IndexStore):
    assert store.list_paths() == []
    docs = [
        _doc("a.md", "alpha"),
        _doc("b.md", "beta"),
        _doc("c.md", "gamma"),
    ]
    store.upsert_docs(docs)
    assert sorted(store.list_paths()) == ["a.md", "b.md", "c.md"]


# ---------------------------------------------------------------------------
# Hybrid search (RRF)
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_store(tmp_path: Path) -> IndexStore:
    """Three distinct docs with hand-picked orthogonal embeddings."""
    db = tmp_path / "vec.db"
    s = IndexStore(str(db))

    texts = {
        "alpha.md": "alpha content with searchable alpha text here",
        "beta.md": "beta content with searchable beta text here",
        "gamma.md": "gamma content with searchable gamma text here",
    }
    vecs = {
        "alpha.md": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "beta.md": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "gamma.md": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }

    docs = [
        _doc(name, content, title=name) for name, content in texts.items()
    ]
    s.upsert_docs(docs)

    items: list[tuple[str, np.ndarray]] = []
    for d in docs:
        for c in chunk_docs([d]):
            s.replace_chunks_for_doc(d.relpath, [c])  # idempotent per-doc
            items.append((c.chunk_id, vecs[d.relpath]))
    s.upsert_chunk_embeddings(items)

    yield s
    s.close()


def test_search_vector_only(vector_store: IndexStore):
    # [1,0,0,0] is identical to alpha.md's vector -> distance 0.
    res = vector_store.search(
        embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=8
    )
    assert res, "vector search returned nothing"
    assert res[0].source == "alpha.md"
    # Only vector results carry RRF scores; FTS contributes nothing here.
    assert all(r.score > 0 for r in res)


def test_search_fts_only(vector_store: IndexStore):
    res = vector_store.search(query_text="beta", k=8)
    assert res, "FTS search returned nothing"
    assert any(r.source == "beta.md" for r in res)
    assert all(r.score > 0 for r in res)


def test_search_hybrid_union_of_vector_and_fts(vector_store: IndexStore):
    vec_only = vector_store.search(
        embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=8
    )
    fts_only = vector_store.search(query_text="beta", k=8)
    assert any(r.source == "alpha.md" for r in vec_only)
    assert any(r.source == "beta.md" for r in fts_only)

    both = vector_store.search(
        embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        query_text="beta",
        k=8,
    )
    sources = {r.source for r in both}
    # Union: the alpha hit from vector AND the beta hit from FTS both appear.
    assert "alpha.md" in sources
    assert "beta.md" in sources
    # Scores remain positive RRF accumulations.
    assert all(r.score > 0 for r in both)


def test_search_k_cap_is_respected(vector_store: IndexStore):
    res = vector_store.search(
        embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        query_text="beta",
        k=2,
    )
    assert len(res) <= 2


def test_search_with_neither_embedding_nor_text_returns_empty(store: IndexStore):
    assert store.search() == []
    assert store.search(embedding=None, query_text=None) == []
    assert store.search(embedding=None, query_text="") == []


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------


def test_upsert_chunk_embeddings_dim_mismatch_raises(tmp_path: Path):
    store = IndexStore(str(tmp_path / "test.db"))
    try:
        doc = _doc("x.md", "## H\nbody content for chunking here\n")
        store.upsert_docs([doc])
        chunks = chunk_docs([doc])
        store.replace_chunks_for_doc(doc.relpath, chunks)

        # Seed the vector table at dim 4.
        store.upsert_chunk_embeddings(
            [(chunks[0].chunk_id, np.zeros(4, dtype=np.float32))]
        )

        # A second write at dim 8 must refuse — different model, different shape.
        with pytest.raises(RuntimeError, match="dim mismatch"):
            store.upsert_chunk_embeddings(
                [(chunks[0].chunk_id, np.zeros(8, dtype=np.float32))]
            )
    finally:
        store.close()


def test_fts_sanitization_does_not_crash_on_punctuation_query(store: IndexStore):
    doc = _doc(
        "x.md",
        "## H\nwhat's in a name? a mystery to explore thoroughly here.\n",
    )
    store.upsert_docs([doc])
    chunks = chunk_docs([doc])
    store.replace_chunks_for_doc(doc.relpath, chunks)

    # The sanitizer drops the parenthetical token; the rest stays as quoted
    # phrase queries. FTS5 must accept the result without raising.
    results = store.search(query_text="what's (in) a name?", k=8)
    assert isinstance(results, list)
