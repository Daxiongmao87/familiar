"""Tests for dmd.init_pass — full pipeline over a tmp campaign."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from dmd.config import load_config_dict
from dmd.gateway import GatewayError
from dmd.index_store import IndexStore
from dmd.init_pass import InitResult, run_init


class FakeEmbedder:
    """Deterministic 8-dim embedder. No real model loaded."""

    def __init__(self, name: str = "fake-embed-8d") -> None:
        self.name = name
        self._dim = 8
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts):
        self.calls += 1
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # md5 is deterministic across runs, unlike Python's randomized hash().
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            for j in range(self._dim):
                out[i, j] = float((h >> (j * 4)) & 0xF) / 15.0
        return out


class FakeGateway:
    """Stand-in for Gateway: returns canned chat response or raises."""

    def __init__(
        self,
        *,
        response: dict | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.response = response if response is not None else {"entities": []}
        self.raise_exc = raise_exc
        self.calls = 0

    async def chat(self, role: str, messages, **kwargs):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


@pytest.fixture
def store(tmp_path: Path):
    s = IndexStore(str(tmp_path / "index.db"))
    yield s
    s.close()


def _cfg() -> object:
    return load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://test", "model_id": "s"},
            "stt": {"base_url": "http://test"},
        }
    })


def _campaign(tmp_path: Path, names: list[str] = ("a.md", "b.md")) -> Path:
    camp = tmp_path / "camp"
    camp.mkdir()
    for n in names:
        # Each doc is long enough to produce at least one chunk.
        (camp / n).write_text(f"# {n}\n\nContent of {n}.\n" * 20)
    return camp


@pytest.mark.asyncio
async def test_run_init_counts_and_progress_stages(tmp_path: Path, store: IndexStore) -> None:
    """InitResult counts correct; progress stages from scan_folder through tools stages."""
    campaign = _campaign(tmp_path)
    progress: list[str] = []

    gw = FakeGateway(response={"entities": [
        {"canonical": "Alpha", "aliases": [], "etype": "character", "weight": 0.8},
        {"canonical": "Beta", "aliases": ["The Beta"], "etype": "place", "weight": 0.5},
    ]})
    embedder = FakeEmbedder()

    result = await run_init(str(campaign), _cfg(), store, gw, embedder, progress.append)

    assert isinstance(result, InitResult)
    assert result.n_docs == 2
    assert result.n_chunks > 0
    assert result.n_entities == 2
    assert result.lexicon, "lexicon should have at least one entry"
    assert all(e.canonical in {"Alpha", "Beta"} for e in result.lexicon)

    expected_subsequence = [
        "scan_folder",
        "upsert_docs",
        "chunk_docs",
        "embed",
        "upsert_chunk_embeddings",
        "extract_entities",
        "upsert_entities",
        "build_lexicon",
        "discover_tools",
        "register_tools",
    ]
    for stage in expected_subsequence:
        assert stage in progress, f"missing stage {stage} in {progress}"
    # Strict ordering: each stage index < next
    indices = [progress.index(s) for s in expected_subsequence]
    paired = list(zip(expected_subsequence, indices, strict=True))
    assert indices == sorted(indices), f"stage order violated: {paired}"


@pytest.mark.asyncio
async def test_run_init_embedding_model_pinned(tmp_path: Path, store: IndexStore) -> None:
    """store.get_meta('embedding_model') reflects the embedder.name passed in."""
    campaign = _campaign(tmp_path)
    embedder = FakeEmbedder(name="my-test-embedder")
    gw = FakeGateway()
    await run_init(str(campaign), _cfg(), store, gw, embedder)
    assert store.get_meta("embedding_model") == "my-test-embedder"


@pytest.mark.asyncio
async def test_run_init_warns_on_entity_failure_but_still_indexes_chunks(
    tmp_path: Path, store: IndexStore
) -> None:
    """When extract_entities fails, indexing still completes AND warnings are emitted."""
    campaign = _campaign(tmp_path)
    gw = FakeGateway(raise_exc=GatewayError("chat failed"))
    embedder = FakeEmbedder()

    result = await run_init(str(campaign), _cfg(), store, gw, embedder)

    assert store.counts()["chunks"] > 0, "chunks should still be embedded on entity failure"
    assert result.warnings, "expected enrich-related warning when chat fails"
    assert any("enrich" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_run_init_deletes_removed_doc_on_second_run(
    tmp_path: Path, store: IndexStore
) -> None:
    """Deleted document is removed from the store on the second run."""
    campaign = _campaign(tmp_path)
    embedder = FakeEmbedder()
    gw = FakeGateway()

    await run_init(str(campaign), _cfg(), store, gw, embedder)
    paths_first = set(store.list_paths())
    assert paths_first == {"a.md", "b.md"}

    # Delete one file from disk and re-run.
    (campaign / "a.md").unlink()
    await run_init(str(campaign), _cfg(), store, gw, embedder)
    paths_second = set(store.list_paths())
    assert "a.md" not in paths_second
    assert "b.md" in paths_second
    counts = store.counts()
    assert counts["docs"] == 1
    assert counts["chunks"] > 0  # b.md still has its chunks
    # No leftover chunks belong to the removed doc.
    leftover = [
        r[0] for r in store._conn.execute(
            "SELECT DISTINCT doc_path FROM chunks"
        ).fetchall()
    ]
    assert "a.md" not in leftover