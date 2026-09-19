"""Focused live card test: real 8081 models + Vellmarsh campaign, generous agent budget.

Confirms the full WorkerAgent loop produces a campaign-grounded CARD when given
enough wall-clock budget (the 4b model is slow under current GPU load).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dmd.config import load_config
from dmd.embedder import Embedder
from dmd.gateway import Gateway
from dmd.index_store import IndexStore
from dmd.lexicon import build_lexicon
from dmd.orchestrator import JobPool
from dmd.pipeline import SessionEngine
from dmd.scanner import chunk_docs, scan_folder
from dmd.triggers import detect_trigger
from dmd.types import Utterance

CAMPAIGN = str(ROOT / "tests" / "e2e" / "campaign")
INDEX = "/tmp/live_index.db"


async def main() -> None:
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.project.path = CAMPAIGN
    cfg.agent.agent_timeout_s = 180.0  # generous: 4b is slow under current GPU load
    cfg.agent.max_tool_calls = 8
    gw = Gateway(cfg)

    # warmup so the card isn't paying for a cold slot
    t0 = time.time()
    await detect_trigger(gw, "warmup call")
    print(f"warmup classifier: {time.time()-t0:.1f}s")

    store = IndexStore(INDEX)
    if store.counts()["docs"] == 0:
        docs = scan_folder(CAMPAIGN)
        store.upsert_docs(docs)
        chunks = chunk_docs(docs)
        by_doc: dict[str, list] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        emb = Embedder(cfg.models.embeddings.model_id or "BAAI/bge-small-en-v1.5")
        for d in docs:
            store.replace_chunks_for_doc(d.relpath, by_doc.get(d.relpath, []))
        items = [(c.chunk_id, emb.embed([c.text])[0]) for c in chunks]
        store.upsert_chunk_embeddings(items)
    emb = Embedder(cfg.models.embeddings.model_id or "BAAI/bge-small-en-v1.5")
    entries = build_lexicon(store.all_entities())

    cards: list = []
    events: list[dict] = []
    async def on_card(card):
        cards.append(card)

    pool = JobPool(max_concurrent=2, job_timeout_s=200.0, stale_after_s=120.0, on_card=on_card)
    engine = SessionEngine(cfg=cfg, store=store, gw=gw, entries=entries, embedder=emb,
                           pool=pool, on_event=events.append, project_path=CAMPAIGN)

    t0 = time.time()
    await engine.handle_utterance(Utterance(user_id="kael", text="I search the goblin corpse for loot", t_start=t0, t_end=t0 + 1.0))
    print(f"loot line processed: {time.time()-t0:.1f}s")
    t0 = time.time()
    await engine.handle_utterance(Utterance(user_id="bryn", text="tell me the history of the sunken crypt", t_start=t0, t_end=t0 + 1.0))
    print(f"lore line processed: {time.time()-t0:.1f}s")
    await pool.close()

    print("\n== RESULTS ==")
    print(f"cards: {len(cards)}")
    for c in cards:
        print(f"  kind={getattr(c,'kind',None)!r} title={getattr(c,'title',None)!r} error={getattr(c,'error',None)!r}")
        print(f"    body={ (getattr(c,'body_md','') or '')[:400]!r }")
    scene = [e for e in events if e.get("type") == "scene_context"]
    print(f"scene_context: {len(scene)}")
    for e in scene:
        print(f"    text={ e.get('text','')[:300]!r }")
    await gw.aclose()
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
