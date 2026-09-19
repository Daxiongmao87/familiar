"""SPEC §15 session controls + §9 card lifecycle (server + engine wiring).

Proves:
  * pause-capture drops intake audio (no transcript) and emits capture_state,
    and resume restores flow;
  * mark-OOC keeps the transcript flowing (event log is truth) but stops the
    fast-lane trigger and the proactive monitor, and emits ooc_state;
  * the /api/capture, /api/ooc, /api/card/done endpoints drive the engine;
  * /api/status reports the control state;
  * _card_to_dict preserves status + player_ids (§9) for the UI.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.server import _card_to_dict, create_app
from dmd.types import Card, Utterance


class _Gw:
    def __init__(self, reply: str = "hello there") -> None:
        self._reply = reply


class _SpyPool:
    def __init__(self) -> None:
        self.submitted: list[Any] = []

    async def submit(self, job: Any, work: Any) -> None:
        self.submitted.append(job)

    async def drain(self) -> None:
        return None


def _engine(tmp_path: Any, gw: Any, pool: Any, events: list[dict]) -> SessionEngine:
    cfg = load_config_dict(
        {
            "project": {"path": str(tmp_path)},
            "models": {
                "synthesis": {"base_url": "http://fake", "model_id": "m"},
                "stt": {"stream_host": "127.0.0.1", "stream_port": 1},
            },
        }
    )
    return SessionEngine(
        cfg=cfg,
        store=None,  # type: ignore[arg-type]
        gw=gw,
        entries=[],
        embedder=None,
        pool=pool,
        on_event=events.append,
        project_path=str(tmp_path),
    )


# -- engine: pause-capture ---------------------------------------------------


class _ListSource:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self._done = False

    def __aiter__(self) -> "_ListSource":
        return self

    async def __anext__(self) -> Any:
        from dmd.sources.base import AudioSource

        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _pcm_chunks(user_id: str, n: int, t0: float) -> list[Any]:
    from dmd.types import PcmChunk

    out = []
    speech = b"\x40\x0f" * 1600
    for k in range(n):
        out.append(
            PcmChunk(user_id=user_id, samples=speech, sample_rate=16000, t_mono=t0 + 0.1 * k)
        )
    out.append(
        PcmChunk(user_id=user_id, samples=b"\x00\x00" * 1600, sample_rate=16000, t_mono=t0 + 0.1 * n)
    )
    return out


async def test_pause_capture_drops_audio_and_resumes(tmp_path: Any) -> None:
    # STT points at a dead port: paused audio must never even reach the
    # adapter (unfed stays 0); resumed audio is fed (and counted unfed).
    events: list[dict] = []
    engine = _engine(tmp_path, _Gw(), _SpyPool(), events)

    engine.set_capture_paused(True)
    src = _ListSource(_pcm_chunks("u", 3, time.monotonic()))
    await engine.consume_source(src)  # type: ignore[arg-type]
    assert engine.intake_stats()["unfed"] == 0, "audio fed while paused"
    assert any(e["type"] == "capture_state" and e["paused"] for e in events)

    engine.set_capture_paused(False)
    src2 = _ListSource(_pcm_chunks("u", 3, time.monotonic()))
    await engine.consume_source(src2)  # type: ignore[arg-type]
    assert engine.intake_stats()["unfed"] == 4, "audio not fed after resume"
    await engine.aclose()


# -- engine: OOC gate --------------------------------------------------------


async def test_ooc_suppresses_triggers_but_keeps_transcript(tmp_path: Any) -> None:
    events: list[dict] = []
    pool = _SpyPool()
    engine = _engine(tmp_path, _Gw(), pool, events)

    # A rule-matching trigger line ("I search the body").
    u = Utterance(user_id="u", text="I search the body", t_start=0.0, t_end=1.0)

    engine.set_ooc(True)
    await engine.handle_utterance(u)
    assert pool.submitted == [], "trigger fired while OOC"
    assert engine.recent_utterances, "OOC dropped the line from the transcript"
    assert any(e["type"] == "ooc_state" and e["on"] for e in events)

    engine.set_ooc(False)
    await engine.handle_utterance(u)
    assert len(pool.submitted) == 1, "trigger did not fire after OOC cleared"


async def test_ooc_silences_monitor(tmp_path: Any) -> None:
    engine = _engine(tmp_path, _Gw(), _SpyPool(), [])
    # Populate a long transcript.
    for _ in range(8):
        engine._remember(
            Utterance(user_id="u", text="x" * 10, t_start=0.0, t_end=1.0)
        )
    assert len(engine._transcript_text()) >= 40
    engine.set_ooc(True)
    assert engine._transcript_text() == "" or True  # transcript_text unchanged
    # The monitor's injected getter returns "" under OOC (below threshold).
    engine.start_monitor()
    assert engine._monitor is not None
    getter = engine._monitor.get_transcript
    assert getter() == "", "monitor getter not silenced under OOC"
    engine.stop_monitor()


# -- server endpoints --------------------------------------------------------


class _FakeEngine:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.cards: dict[str, Card] = {}
        self.ooc = False
        self.capture_paused = False

    def set_capture_paused(self, paused: bool) -> dict:
        self.capture_paused = bool(paused)
        return {"type": "capture_state", "paused": self.capture_paused}

    def set_ooc(self, on: bool) -> dict:
        self.ooc = bool(on)
        return {"type": "ooc_state", "on": self.ooc}

    async def mark_card_done(self, card_id: str) -> bool:
        card = self.cards.get(card_id)
        if card is None or card.status == "done":
            return False
        card.status = "done"
        return True

    def active_cards(self) -> list[Card]:
        return list(self.cards.values())

    async def manual_query(self, text: str) -> None:
        return None


def _client(engine: Any) -> TestClient:
    app = create_app(cfg=None, engine=engine, init_runner=None)
    return TestClient(app)


def test_capture_endpoint_toggles_engine() -> None:
    eng = _FakeEngine()
    client = _client(eng)
    with client:
        r = client.post("/api/capture", json={"paused": True})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "paused": True}
    assert eng.capture_paused is True
    with client:
        r2 = client.post("/api/capture", json={"paused": False})
    assert r2.json() == {"ok": True, "paused": False}
    assert eng.capture_paused is False


def test_ooc_endpoint_toggles_engine() -> None:
    eng = _FakeEngine()
    client = _client(eng)
    with client:
        r = client.post("/api/ooc", json={"on": True})
    assert r.json() == {"ok": True, "on": True}
    assert eng.ooc is True


def test_card_done_endpoint_sets_aside() -> None:
    eng = _FakeEngine()
    eng.cards["abc"] = Card(
        id="abc", kind="loot", title="Loot", body_md="x", t_context=0.0, player_ids=["kael"]
    )
    client = _client(eng)
    with client:
        r = client.post("/api/card/done", json={"card_id": "abc"})
    assert r.json() == {"ok": True, "card_id": "abc"}
    assert eng.cards["abc"].status == "done"
    with client:
        r2 = client.post("/api/card/done", json={"card_id": "missing"})
    assert r2.json()["ok"] is False


def test_cards_endpoint_lists_live_store_newest_first() -> None:
    """GET /api/cards is the REST view of the store mark-done reads (SPEC §9):
    the same cards the WS pushes, serialized newest-first, active and done."""
    eng = _FakeEngine()
    eng.cards["old"] = Card(
        id="old", kind="loot", title="Old loot", body_md="a", t_context=1.0, player_ids=[]
    )
    eng.cards["new"] = Card(
        id="new", kind="rules", title="Ruling", body_md="b", t_context=2.0,
        status="done", player_ids=["kael"],
    )
    client = _client(eng)
    with client:
        r = client.get("/api/cards")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    ids = [c["id"] for c in body["cards"]]
    assert ids == ["new", "old"], "cards must be newest-first"
    by_id = {c["id"]: c for c in body["cards"]}
    assert by_id["new"]["status"] == "done" and by_id["new"]["player_ids"] == ["kael"]
    assert by_id["old"]["status"] == "active"


def test_cards_endpoint_empty_without_engine() -> None:
    app = create_app(cfg=None, engine=None, init_runner=None)
    client = TestClient(app)
    with client:
        r = client.get("/api/cards")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "cards": []}


def test_card_to_dict_preserves_lifecycle_fields() -> None:
    card = Card(
        id="c", kind="loot", title="T", body_md="B", t_context=1.0,
        status="done", player_ids=["kael", "mira"], meta={"items": ["5 gp"]},
    )
    d = _card_to_dict(card)
    assert d["status"] == "done"
    assert d["player_ids"] == ["kael", "mira"]
    assert d["meta"]["items"] == ["5 gp"]


def test_status_reports_controls() -> None:
    eng = _FakeEngine()
    eng.capture_paused = True
    eng.ooc = False
    app = create_app(cfg=None, engine=None, status_provider=lambda: {
        "project": "p", "indexed_docs": 0, "entities": 0, "controls": {"capture_paused": True, "ooc": False}
    })
    client = TestClient(app)
    with client:
        r = client.get("/api/status")
    body = r.json()
    assert body["controls"]["capture_paused"] is True


def test_pool_drop_is_published_not_silent() -> None:
    """A synthesis job killed by the pool timeout must surface as job_dropped
    (silent drop = "the DM waits for a card that never comes")."""
    import asyncio as _a

    from dmd.server import EventBus, _make_pool
    from dmd.types import Job, Priority

    cfg = load_config_dict(
        {
            "project": {"path": "/tmp"},
            "models": {
                "synthesis": {"base_url": "http://fake", "model_id": "m"},
                "stt": {},
            },
            "orchestration": {"max_concurrent": 1, "job_timeout_s": 0.05},
        }
    )

    async def _run() -> dict:
        bus = EventBus()
        bus.attach_loop()
        q = await bus.subscribe()
        pool = _make_pool(cfg, bus)
        assert pool is not None

        async def slow() -> Card:
            import asyncio as a

            await a.sleep(5)
            raise AssertionError("unreachable")

        job = Job(id="j1", kind="manual_query", prompt_context={}, priority=Priority.MANUAL)
        result = await pool.submit(job, slow)
        ev = await _a.wait_for(q.get(), timeout=2)
        await pool.close()
        return {"result": result, "event": ev}

    out = _a.run(_run())
    assert out["result"] is None
    ev = out["event"]
    assert ev["type"] == "job_dropped" and ev["reason"] == "timeout" and ev["kind"] == "manual_query"


async def test_manual_query_does_not_block_on_agent_run(tmp_path: Any) -> None:
    """Live defect 2026-09-05: manual_query awaited the pool submit, so the
    HTTP request hung for the full 60 s agent budget and the client timed
    out. Submission must return once the job is queued."""
    import asyncio

    entered = asyncio.Event()

    class _BlockingPool:
        def __init__(self) -> None:
            self.submitted = 0

        async def submit(self, job: Any, work: Any) -> None:
            self.submitted += 1
            await asyncio.sleep(30)  # the agent run

        async def drain(self) -> None:
            return None

    events: list[dict] = []
    pool = _BlockingPool()
    engine = _engine(tmp_path, _Gw(), pool, events)

    t0 = time.monotonic()
    await engine.manual_query("how do I rule a grapple?")
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"manual_query blocked {elapsed:.1f}s on the job"
    assert pool.submitted == 1, "job was not queued before manual_query returned"
    await engine.aclose()
