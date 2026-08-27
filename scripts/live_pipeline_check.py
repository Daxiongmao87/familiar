"""Live pipeline test: real localhost:8081 models (qwen3.8-4b) + e2e fixture campaign.

Drives the FULL live path against the real local models:
  1. fast-lane classifier (detect_trigger) on sample utterances;
  2. a loot trigger through WorkerAgent -> expects a campaign-grounded CARD;
  3. a lore trigger -> expects a grounded scene-context (ephemeral tier).
Confirms the implementation works end-to-end with the real models (not the mock).
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
from dmd.triggers import detect_trigger  # noqa: E402
from dmd.types import Utterance  # noqa: E402

CAMPAIGN = str(ROOT / "tests" / "e2e" / "campaign")


async def main() -> None:
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.project.path = CAMPAIGN
    gw = Gateway(cfg)

    print("== 1. fast-lane classifier (real qwen3.8-4b) ==")
    samples = [
        ("kael", "I search the goblin corpse for loot"),
        ("kael", "rough luck, nothing on him"),
        ("bryn", "what's the history of this crypt?"),
    ]
    for uid, text in samples:
        t0 = time.time()
        try:
            is_trig, kind = await detect_trigger(gw, text)
            print(f"  trigger={is_trig!s:5} kind={kind:8} ({time.time()-t0:5.1f}s)  {text!r}")
        except Exception as e:
            print(f"  classifier ERROR: {type(e).__name__}: {e}")
            break

    print("\n== 2. full agentic loop (real models + Vellmarsh campaign) ==")
    store = IndexStore("/tmp/live_index.db")
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
    entries = build_lexicon(store.all_entities())
    print(f"  indexed {len(docs)} docs, {len(chunks)} chunks, {len(entries)} lexicon entries")

    cards: list = []
    events: list[dict] = []

    async def on_card(card):
        cards.append(card)

    pool = JobPool(max_concurrent=2, job_timeout_s=90.0, stale_after_s=120.0, on_card=on_card)
    engine = SessionEngine(
        cfg=cfg, store=store, gw=gw, entries=entries, embedder=emb,
        pool=pool, on_event=events.append, project_path=CAMPAIGN,
    )

    live_session = [
        ("kael", "I search the goblin corpse for loot"),  # loot trigger -> CARD
        ("bryn", "tell me the history of the sunken crypt"),  # lore trigger -> scene context
    ]
    for uid, text in live_session:
        t0 = time.time()
        try:
            await engine.handle_utterance(Utterance(user_id=uid, text=text, t_start=t0, t_end=t0 + 1.0))
            print(f"  processed {text!r} ({time.time()-t0:5.1f}s)")
        except Exception as e:
            print(f"  handle_utterance ERROR: {type(e).__name__}: {e}")
    await pool.close()

    print("\n== RESULTS ==")
    print(f"  cards produced: {len(cards)}")
    for c in cards:
        body = (getattr(c, "body_md", "") or "")
        print(f"  - kind={getattr(c,'kind',None)!r} title={getattr(c,'title',None)!r}")
        print(f"    body[:280]={body[:280]!r}")
    scene = [e for e in events if e.get("type") == "scene_context"]
    print(f"  scene_context events: {len(scene)}")
    for e in scene:
        print(f"    text[:200]={e.get('text','')[:200]!r}")
    await gw.aclose()
    print("\nDONE")


if __name__ == "__main__":
    asyncio.run(main())
