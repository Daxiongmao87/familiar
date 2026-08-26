"""Top-level project initialization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import AppConfig
from .embedder import Embedder
from .enrich import extract_entities
from .gateway import Gateway
from .index_store import IndexStore
from .lexicon import build_lexicon
from .scanner import Chunk, chunk_docs, scan_folder
from .tools_reg import ToolRegistry, discover_tools
from .types import Entity, LexiconEntry, ToolSpec

_EMBED_BATCH = 64


@dataclass(slots=True)
class InitResult:
    """Summary of an init pass over a project folder."""

    n_docs: int = 0
    n_chunks: int = 0
    n_entities: int = 0
    lexicon: list[LexiconEntry] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _cb(progress_cb: Callable[[str], None] | None, msg: str) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(msg)
    except Exception:
        pass


def _remove_missing_docs(
    store: IndexStore, current_paths: set[str]
) -> list[str]:
    removed: list[str] = []
    stored = set(store.list_paths())
    for path in sorted(stored - current_paths):
        try:
            store.remove_doc(path)
            removed.append(path)
        except Exception as e:
            pass
    return removed


async def run_init(
    project_path: str,
    cfg: AppConfig,
    store: IndexStore,
    gw: Gateway,
    embedder: Embedder,
    progress_cb: Callable[[str], None] | None = None,
) -> InitResult:
    """Run the deterministic init pipeline over a project folder."""
    warnings: list[str] = []
    try:
        store.set_meta("embedding_model", str(getattr(embedder, "name", "") or ""))
    except Exception:
        pass

    _cb(progress_cb, "scan_folder")
    docs = scan_folder(project_path)

    _cb(progress_cb, "stale_docs")
    current = {d.relpath: d.mtime for d in docs}
    _ = store.stale_docs(current)
    _remove_missing_docs(store, set(current.keys()))

    _cb(progress_cb, "upsert_docs")
    store.upsert_docs(docs)

    _cb(progress_cb, "chunk_docs")
    all_chunks: list[Chunk] = chunk_docs(docs)
    by_doc: dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for d in docs:
        store.replace_chunks_for_doc(d.relpath, by_doc.get(d.relpath, []))

    _cb(progress_cb, "embed")
    chunk_texts = [c.text for c in all_chunks]
    chunk_ids = [c.chunk_id for c in all_chunks]
    if chunk_texts:
        vecs_list: list[np.ndarray] = []
        for i in range(0, len(chunk_texts), _EMBED_BATCH):
            batch = chunk_texts[i : i + _EMBED_BATCH]
            arr = embedder.embed(batch)
            vecs_list.append(np.asarray(arr, dtype=np.float32))
        if vecs_list:
            vecs = (
                np.concatenate(vecs_list, axis=0)
                if len(vecs_list) > 1
                else vecs_list[0]
            )
        else:
            vecs = np.zeros((0, 0), dtype=np.float32)
        if vecs.shape[0] == len(chunk_ids) and vecs.size > 0:
            _cb(progress_cb, "upsert_chunk_embeddings")
            store.upsert_chunk_embeddings(
                list(zip(chunk_ids, [vecs[i] for i in range(vecs.shape[0])]))
            )

    _cb(progress_cb, "extract_entities")
    entities: list[Entity] = []
    try:
        entities = await extract_entities(docs, gw)
    except Exception as e:
        warnings.append(f"enrich:{type(e).__name__}:{e}")
        entities = []

    if entities:
        _cb(progress_cb, "upsert_entities")
        store.upsert_entities(entities)

    _cb(progress_cb, "build_lexicon")
    lexicon = build_lexicon(entities)

    _cb(progress_cb, "discover_tools")
    candidates = discover_tools(project_path)
    _cb(progress_cb, "register_tools")
    registry = ToolRegistry()
    tools: list[ToolSpec] = []
    try:
        tools = await registry.register(candidates)
    except Exception as e:
        warnings.append(f"tools:{type(e).__name__}:{e}")
    if registry.rejected:
        warnings.append(
            f"tools:rejected={len(registry.rejected)}"
        )

    return InitResult(
        n_docs=len(docs),
        n_chunks=len(all_chunks),
        n_entities=len(entities),
        lexicon=lexicon,
        tools=tools,
        warnings=warnings,
    )
