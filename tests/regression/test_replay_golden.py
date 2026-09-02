"""Golden regression: scripted replay through the real pipeline composition.

Exercises run_init (deterministic fake LLM) + SessionEngine + JobPool + IndexStore
end-to-end in-process, then compares the normalized event stream against a committed
golden snapshot. Regenerate with DMD_REGEN_GOLDEN=1.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dmd.config import load_config_dict
from dmd.index_store import IndexStore
from dmd.init_pass import run_init
from dmd.lexicon import build_lexicon
from dmd.orchestrator import JobPool
from dmd.pipeline import SessionEngine
from dmd.types import Card, Entity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "golden" / "replay_events.json"

CAMPAIGN_FILES: dict[str, str] = {
    "locations/ashforge.md": (
        "---\ntitle: The Ashforge\ntags: [location]\n---\n\n"
        "# The Ashforge\n\nA dwarven foundry district east of the temple square.\n"
        "Governed by [[Brann Stonevein|the Forge-master]].\n\n"
        "## Guard Detail\n\nSix watches a night. Scouts carry coded pouches."
    ),
    "npcs/vexahlia.md": (
        "---\ntitle: Vex'ahlia\ntags: [npc, ranger]\n---\n\n"
        "# Vex'ahlia\n\nA ranger bearing the sealed signet of the Ashforge scouts.\n"
        "Keen-eyed; distrusts strangers near the forge."
    ),
    "items/scouts_pouch.md": (
        "---\ntitle: Scout's Pouch\n---\n\n"
        "# Scout's Pouch\n\nContains 5 gp and a coded note sealed with Vex'ahlia's mark.\n"
        "DC 12 Investigation to find, DC 14 Perception for the boot dagger."
    ),
}

FIXED_ENTITIES = [
    Entity("Vex'ahlia", ["Vexie"], "character", 1.0, ["npcs/vexahlia.md"]),
    Entity("Ashforge", ["the Ashforge"], "place", 0.9, ["locations/ashforge.md"]),
    Entity("Scout's Pouch", [], "item", 0.8, ["items/scouts_pouch.md"]),
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


def _loot_card_json() -> dict[str, Any]:
    return {
        "kind": "skill_table",
        "title": "Loot Table — Fallen Scout",
        "body_md": (
            "| Skill | DC |\n|---|---|\n| Investigation | 12 |\n| Perception | 14 |"
        ),
    }


def _rules_card_json() -> dict[str, Any]:
    return {
        "kind": "rules",
        "title": "Rules — Grappled",
        "body_md": "**Grappled** (SRD 5.1): speed becomes 0.",
    }


class FakeGateway:
    def __init__(self, stt_script: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._stt_script = list(stt_script or [])
        self.stt_calls = 0

    @property
    def cfg(self) -> Any:
        class _M:
            fast = None

        return type("C", (), {"models": _M()})()

    async def chat(
        self,
        role: str,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | dict:
        props = sorted(
            ((json_schema or {}).get("properties") or {}).keys()
        )
        user_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        self.calls.append({"role": role, "props": props, "tail": user_text[-200:]})
        if "entities" in props:
            return {
                "entities": [
                    {"canonical": e.canonical, "aliases": e.aliases, "etype": e.etype, "weight": e.weight}
                    for e in FIXED_ENTITIES
                ]
            }
        if "grappl" in user_text.lower():
            return _rules_card_json()
        return _loot_card_json()

    async def transcribe(self, audio_bytes: bytes, **kw: Any) -> str:
        self.stt_calls += 1
        if self._stt_script:
            return self._stt_script.pop(0)
        raise AssertionError("STT script exhausted")

    async def aclose(self) -> None:
        pass


@pytest.fixture()
async def rig(tmp_path: Path):
    campaign = tmp_path / "campaign"
    for rel, text in CAMPAIGN_FILES.items():
        p = campaign / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    cfg = load_config_dict(
        {
            "project": {"path": str(campaign), "name": "Golden Campaign"},
            "models": {
                "synthesis": {
                    "base_url": "http://fake.invalid/v1",
                    "api_key": "k",
                    "model_id": "fake-synthesis",
                },
                "stt": {"base_url": "http://fake.invalid/v1"},
                "embeddings": {"provider": "local", "model_id": "hash8"},
            },
        }
    )

    store = IndexStore(str(tmp_path / "index.db"))
    gw = FakeGateway(
        stt_script=[
            "I search the body",
            "never mind, rough luck there",
        ]
    )
    embedder = HashEmbedder()

    result = await run_init(str(campaign), cfg, store, gw, embedder)
    assert result.warnings == []
    entries = build_lexicon(store.all_entities())

    events: list[dict[str, Any]] = []

    async def on_card(card: Card) -> None:
        events.append({"type": "card", "card": asdict(card)})

    pool = JobPool(max_concurrent=1, job_timeout_s=5.0, stale_after_s=120.0, on_card=on_card)
    engine = SessionEngine(
        cfg=cfg,
        store=store,
        gw=gw,
        entries=entries,
        embedder=embedder,
        pool=pool,
        on_event=events.append,
    )
    yield engine, pool, events, gw
    await pool.close()


def _normalize(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    card_n = 0
    for ev in events:
        ev = json.loads(json.dumps(ev))
        if ev.get("type") == "card":
            card = ev["card"]
            card["id"] = f"card-{card_n}"
            card["t_context"] = 0.0
            meta = card.get("meta") or {}
            meta["sources"] = sorted(meta.get("sources", []))
            meta["entities"] = sorted(meta.get("entities", []))
            card["meta"] = meta
            card_n += 1
        out.append(ev)
    return out


async def test_full_replay_matches_golden(rig: Any, tmp_path: Path) -> None:
    engine, pool, events, _gw = rig
    silence = b"\x00\x00" * 320

    for user, t0 in (("dm", 0.0), ("alice", 1.0)):
        u = await engine.transcribe_pcm(user, silence, t0, t0 + 1.0)
        if u is not None:
            await engine.handle_utterance(u)
    await engine.manual_query("how do grappling rules work")
    await pool.drain()

    kinds = [ev["type"] for ev in events]
    assert kinds.count("transcript") == 2
    assert kinds.count("card") == 2
    cards = [ev["card"] for ev in events if ev["type"] == "card"]
    assert [c["kind"] for c in cards] == ["skill_table", "rules"]

    normalized = _normalize(events)
    regen = os.environ.get("DMD_REGEN_GOLDEN") == "1" or not GOLDEN.exists()
    if regen:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    golden = json.loads(GOLDEN.read_text())
    assert normalized == golden, (
        "event stream drifted from committed golden snapshot; "
        "if intentional, regenerate with DMD_REGEN_GOLDEN=1 and review the diff"
    )


async def test_non_trigger_utterance_produces_no_card(rig: Any) -> None:
    engine, pool, events, _ = rig
    silence = b"\x00\x00" * 320
    n_cards_before = sum(1 for e in events if e["type"] == "card")
    await engine.transcribe_pcm("bob", silence, 3.0, 4.0)
    await pool.drain()
    n_cards_after = sum(1 for e in events if e["type"] == "card")
    assert n_cards_after == n_cards_before
