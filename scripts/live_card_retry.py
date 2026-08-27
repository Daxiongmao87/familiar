"""Retry-based live card: capture a real grounded card from the 4b, waiting for a
free llama-server slot (production traffic saturates the 6 parallel slots).
Bounded: up to N attempts with short waits.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dmd.config import load_config  # noqa: E402
from dmd.gateway import Gateway  # noqa: E402
from dmd.index_store import IndexStore  # noqa: E402
from dmd.embedder import Embedder  # noqa: E402
from dmd.lexicon import build_lexicon  # noqa: E402
from dmd.scanner import scan_folder, chunk_docs  # noqa: E402
from dmd.orchestrator import JobPool  # noqa: E402
from dmd.pipeline import SessionEngine  # noqa: E402
from dmd.types import Utterance  # noqa: E402

CAMPAIGN = str(ROOT / "tests" / "e2e" / "campaign")
INDEX = "/tmp/live_index.db"
ATTEMPTS = 6
WAIT = 12


def build_stack(cfg, store):
    emb = Embedder(cfg.models.embeddings.model_id or "BAAI/bge-small-en-v1.5")
    if store.counts()["docs"] == 0:
        docs = scan_folder(CAMPAIGN)
        store.upsert_docs(docs)
        chunks = chunk_docs(docs)
        by_doc: dict[str, list] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        for d in docs:
            store.replace_chunks_for_doc(d.relpath, by_doc.get(d.relpath, []))
        store.upsert_chunk_embeddings([(c.chunk_id, emb.embed([c.text])[0]) for c in chunks])
    entries = build_lexicon(store.all_entities())
    return emb, entries


async def main() -> None:
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.project.path = CAMPAIGN
    cfg.agent.agent_timeout_s = 170.0
    cfg.agent.max_tool_calls = 8
    gw = Gateway(cfg)
    store = IndexStore(INDEX)
    emb, entries = build_stack(cfg, store)

    for attempt in range(1, ATTEMPTS + 1):
        cards: list = []
        async def on_card(card):
            cards.append(card)
        pool = JobPool(max_concurrent=2, job_timeout_s=180.0, stale_after_s=120.0, on_card=on_card)
        engine = SessionEngine(cfg=cfg, store=store, gw=gw, entries=entries, embedder=emb,
                               pool=pool, on_event=lambda *a: None, project_path=CAMPAIGN)
        t0 = time.time()
        try:
            await engine.handle_utterance(Utterance(user_id="kael", text="I search the goblin corpse for loot", t_start=t0, t_end=t0 + 1.0))
        except Exception as e:
            print(f"attempt {attempt}: engine error {type(e).__name__}: {e}")
        await pool.close()
        dt = time.time() - t0
        got = cards and not getattr(cards[0], "error", None)
        print(f"attempt {attempt} ({dt:.0f}s): cards={len(cards)} grounded={bool(got)}")
        if got:
            c = cards[0]
            print("\n== GROUNDED CARD (real model) ==")
            print(f"kind={getattr(c,'kind',None)!r} title={getattr(c,'title',None)!r}")
            print(f"body={ (getattr(c,'body_md','') or '') }")
            break
        if attempt < ATTEMPTS:
            await asyncio.sleep(WAIT)
    await gw.aclose()
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
