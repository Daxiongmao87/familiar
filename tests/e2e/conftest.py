"""Full-stack E2E fixtures: mock OpenAI backend + real app wiring + live uvicorn."""

from __future__ import annotations

import asyncio
import hashlib
import socket
import struct
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import uvicorn
from mock_server import MockState, create_mock_app

from dmd.config import load_config_dict
from dmd.gateway import Gateway
from dmd.index_store import IndexStore
from dmd.lexicon import build_lexicon
from dmd.orchestrator import JobPool
from dmd.pipeline import SessionEngine
from dmd.server import EventBus, create_app
from dmd.types import Entity

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMPAIGN_FILES: dict[str, str] = {
    "locations/ashforge.md": (
        "# The Ashforge\n\nA dwarven foundry district east of the temple square.\n\n"
        "## Guard Detail\n\nSix watches a night. Scouts carry coded pouches."
    ),
    "npcs/vexahlia.md": (
        "# Vex'ahlia\n\nA ranger bearing the sealed signet of the Ashforge scouts."
    ),
    "items/scouts_pouch.md": (
        "# Scout's Pouch\n\nContains 5 gp and a coded note sealed with Vex'ahlia's mark.\n"
        "DC 12 Investigation to find, DC 14 Perception for the boot dagger."
    ),
}

FIXED_ENTITIES = [
    Entity("Vex'ahlia", ["Vexie"], "character", 1.0, ["npcs/vexahlia.md"]),
    Entity("Ashforge", [], "place", 0.9, ["locations/ashforge.md"]),
]


def _stable_vec(text: str, dim: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = [struct.unpack("<I", digest[i * 4 : i * 4 + 4])[0] for i in range(dim)]
    scale = max(vals) or 1
    return [v / scale for v in vals]


class HashEmbedder:
    name = "hash8"
    dim = 8

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray([_stable_vec(t) for t in texts], dtype=np.float32)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve_in_thread(app: Any, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    return server, thread


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


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp = tmp_path_factory.mktemp("e2e")
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
                "synthesis": {
                    "base_url": mock_url,
                    "api_key": "e2e-key",
                    "model_id": "mock-synthesis",
                },
                "fast": {
                    "base_url": mock_url,
                    "api_key": "e2e-key",
                    "model_id": "mock-fast",
                    "extra_body": {"enable_thinking": False},
                },
                "stt": {"base_url": mock_url, "api_key": "e2e-key"},
                "embeddings": {"provider": "local", "model_id": "hash8"},
            },
            "orchestration": {"max_concurrent": 2, "job_timeout_s": 10.0},
        }
    )

    store = IndexStore(str(tmp / "index.db"))
    gw = Gateway(cfg)

    from dmd.scanner import chunk_docs, scan_folder

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

    async def on_card(card: Any) -> None:
        await bus.publish({"type": "card", "card": asdict(card)})

    pool = JobPool(max_concurrent=2, job_timeout_s=10.0, stale_after_s=120.0, on_card=on_card)
    engine = SessionEngine(
        cfg=cfg,
        store=store,
        gw=gw,
        entries=entries,
        embedder=emb,
        pool=pool,
        on_event=bus.publish_sync,
    )

    def status_provider() -> dict[str, Any]:
        counts = store.counts()
        return {
            "project": "E2E Campaign",
            "indexed_docs": counts["docs"],
            "entities": counts["entities"],
            "tools": 0,
            "roles": {
                "synthesis": True,
                "fast": True,
                "vision": False,
                "stt": True,
            },
        }

    app = create_app(cfg=cfg, engine=engine, init_runner=None, status_provider=status_provider, bus=bus)

    ui_port = _free_port()
    ui_srv, ui_thread = _serve_in_thread(app, ui_port)
    _wait_http(f"http://127.0.0.1:{ui_port}/api/status")

    class Stack:
        pass

    s = Stack()
    s.ui_url = f"http://127.0.0.1:{ui_port}"
    s.mock_url = mock_url
    s.state = state
    s.bus = bus
    s.engine = engine
    s.cfg = cfg
    s.store = store
    s.gw = gw
    s.embedder = emb
    s.entries = entries
    s.campaign_path = str(campaign)
    yield s

    ui_srv.should_exit = True
    mock_srv.should_exit = True
    ui_thread.join(timeout=3)
    mock_thread.join(timeout=3)
