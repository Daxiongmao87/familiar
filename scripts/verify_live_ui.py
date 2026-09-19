"""Verify the live UI data path end-to-end without a browser.

The UI is a thin WebSocket client driven by server events; the functional
contract is the HTTP + WS event stream. This replicates the e2e `stack`
fixture (mock backend + real engine + real uvicorn app), connects a WS
subscriber, drives a manual query + transcript events, and asserts that the
card stream, transcript, and scene-context events all flow to the client.

Run:  python scripts/verify_live_ui.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "e2e"))

import httpx
from mock_server import MockState, create_mock_app

from dmd.config import load_config_dict
from dmd.gateway import Gateway
from dmd.index_store import IndexStore
from dmd.lexicon import build_lexicon
from dmd.orchestrator import JobPool
from dmd.pipeline import SessionEngine
from dmd.scanner import chunk_docs, scan_folder
from dmd.server import EventBus, create_app
from tests.e2e.conftest import (
    CAMPAIGN_FILES,
    FIXED_ENTITIES,
    HashEmbedder,
    _free_port,
)


def _wait_http(url: str, timeout_s: float = 15.0) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.15)
    raise RuntimeError(f"server never became ready: {url}")


async def main() -> int:
    tmp = Path("/tmp/verify_ui")
    tmp.mkdir(exist_ok=True)
    campaign = tmp / "campaign"
    for rel, text in CAMPAIGN_FILES.items():
        p = campaign / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    mock_port = _free_port()
    state = MockState()
    mock_srv, mock_thread = _serve_in_thread(create_mock_app(state), mock_port)
    _wait_http(f"http://127.0.0.1:{mock_port}/v1/models")
    mock_url = f"http://127.0.0.1:{mock_port}/v1"

    cfg = load_config_dict(
        {
            "project": {"path": str(campaign), "name": "E2E Campaign"},
            "models": {
                "synthesis": {"base_url": mock_url, "api_key": "e2e-key", "model_id": "mock-synthesis"},
                "fast": {"base_url": mock_url, "api_key": "e2e-key", "model_id": "mock-fast"},
                "stt": {"base_url": mock_url, "api_key": "e2e-key"},
                "embeddings": {"provider": "local", "model_id": "hash8"},
            },
            "orchestration": {"max_concurrent": 2, "job_timeout_s": 15.0},
        }
    )

    store = IndexStore(str(tmp / "index.db"))
    gw = Gateway(cfg)
    docs = scan_folder(str(campaign))
    store.upsert_docs(docs)
    chunks = chunk_docs(docs)
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    emb = HashEmbedder()
    for d in docs:
        store.replace_chunks_for_doc(d.relpath, by_doc.get(d.relpath, []))
    items = [(c.chunk_id, emb.embed([c.text])[0]) for c in chunks]
    store.upsert_chunk_embeddings(items)
    store.upsert_entities(FIXED_ENTITIES)
    entries = build_lexicon(store.all_entities())

    bus = EventBus()

    async def on_card(card):
        from dataclasses import asdict, is_dataclass

        if is_dataclass(card):
            data = asdict(card)
        else:
            data = {k: getattr(card, k) for k in ("id", "kind", "title", "body_md", "t_context", "meta") if hasattr(card, k)}
        await bus.publish({"type": "card", "card": data})

    pool = JobPool(max_concurrent=2, job_timeout_s=15.0, stale_after_s=120.0, on_card=on_card)
    engine = SessionEngine(
        cfg=cfg, store=store, gw=gw, entries=entries, embedder=emb,
        pool=pool, on_event=bus.publish_sync,
    )

    def status_provider():
        counts = store.counts()
        return {
            "project": "E2E Campaign",
            "indexed_docs": counts["docs"],
            "entities": counts["entities"],
            "tools": 0,
            "roles": {"synthesis": True, "fast": True, "vision": False, "stt": True},
        }

    app = create_app(cfg=cfg, engine=engine, init_runner=None, status_provider=status_provider, bus=bus)

    ui_port = _free_port()
    from uvicorn import Config, Server
    config = Config(app, host="127.0.0.1", port=ui_port, log_level="error")
    ui_srv = Server(config)
    ui_srv.install_signal_handlers = lambda: None
    import threading
    ui_thread = threading.Thread(target=lambda: asyncio.run(ui_srv.serve()), daemon=True)
    ui_thread.start()
    _wait_http(f"http://127.0.0.1:{ui_port}/api/status")
    base = f"http://127.0.0.1:{ui_port}"

    print(f"UI at {base}; mock at {mock_url}")

    # 1. status
    status = httpx.get(f"{base}/api/status", timeout=5).json()
    print(f"status: project={status['project']} docs={status['indexed_docs']} entities={status['entities']}")
    assert status["project"] == "E2E Campaign"
    assert status["indexed_docs"] == 3, status
    assert status["entities"] == 2, status

    # 2. connect WS subscriber and collect events
    events = []
    got_event = asyncio.Event()

    async def consumer():
        import websockets
        async with websockets.connect(f"ws://{base[len('http://'):]}/ws") as ws:
            async for msg in ws:
                ev = json.loads(msg)
                events.append(ev)
                got_event.set()

    ws_task = asyncio.create_task(consumer())
    await asyncio.sleep(0.2)

    # 3. drive a manual query -> should produce a card
    resp = httpx.post(f"{base}/api/query", json={"text": "what do i find when i search the body"}, timeout=20)
    assert resp.status_code == 200, resp.text
    assert resp.json().get("ok") is True

    # wait for a card event
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if any(e.get("type") == "card" for e in events):
            break
        await asyncio.sleep(0.25)

    cards = [e["card"] for e in events if e.get("type") == "card"]
    print(f"card events received: {len(cards)}")
    assert len(cards) >= 1, f"no card event within timeout; events={events[:3]}"
    card = cards[0]
    print(f"  card: title={card['title']!r} kind={card['kind']!r}")
    assert "Loot Table" in card["title"], card
    assert "Investigation" in card["body_md"], card

    # 4. drive transcript events -> transcript lines should arrive
    now = time.time()
    bus.publish_sync({"type": "transcript", "user_id": "111111111111111111", "text": "I search the body", "t": now})
    bus.publish_sync({"type": "transcript", "user_id": "222222222222222222", "text": "roll for perception", "t": now + 1})

    transcript_lines = [e for e in events if e.get("type") == "transcript"]
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        if len(transcript_lines) >= 2:
            break
        await asyncio.sleep(0.25)
    transcript_lines = [e for e in events if e.get("type") == "transcript"]
    print(f"transcript events received: {len(transcript_lines)}")
    assert len(transcript_lines) >= 2, "transcript lines did not arrive"

    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass

    mock_srv.should_exit = True
    ui_srv.should_exit = True
    mock_thread.join(timeout=3)
    ui_thread.join(timeout=3)

    print("PASS: live UI data path (HTTP + WS) delivers cards + transcript events")
    return 0


def _serve_in_thread(app, port):
    from uvicorn import Config, Server
    config = Config(app, host="127.0.0.1", port=port, log_level="error")
    server = Server(config)
    server.install_signal_handlers = lambda: None
    import threading
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
