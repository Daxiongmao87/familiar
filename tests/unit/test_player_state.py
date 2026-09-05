"""§11 per-player state store: seeded at init, updated on mark-done / AI-observe.

Proves the wiring the gap ledger had as "unverified":
  * players are seeded from the campaign's characters/ sheets (frontmatter
    titles respected) when the engine is built;
  * marking a card done (UI button) appends the card ref to the player's
    done_cards and merges observed loot items into inventory — append-only;
  * the monitor's AI-observed ``card_done`` verdict reaches the same store;
  * re-seeding never clobbers accumulated state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.player_state import PlayerState
from dmd.server import _seed_players
from dmd.types import Card


class _NoPool:
    async def submit(self, job: Any, work: Any) -> None:
        return None


def _card(pid: str = "kael", items: list[str] | None = None) -> Card:
    return Card(
        id="c1",
        kind="loot",
        title="Loot Table — Fallen Scout",
        body_md="| Item |\n| 5 gp |",
        t_context=1.0,
        meta={"items": items or ["5 gp", "coded note"]},
        player_ids=[pid],
    )


def _engine(tmp_path: Path, ps: PlayerState, events: list[dict]) -> SessionEngine:
    cfg = load_config_dict(
        {
            "project": {"path": str(tmp_path)},
            "models": {
                "synthesis": {"base_url": "http://fake", "model_id": "m"},
                "stt": {"base_url": "http://fake"},
            },
        }
    )
    return SessionEngine(
        cfg=cfg,
        store=None,  # type: ignore[arg-type]
        gw=object(),
        entries=[],
        embedder=None,
        pool=_NoPool(),
        on_event=events.append,
        project_path=str(tmp_path),
        player_state=ps,
    )


# -- seeding ----------------------------------------------------------------


def test_seed_players_from_characters_dir(tmp_path: Path) -> None:
    chars = tmp_path / "characters"
    chars.mkdir()
    (chars / "kael.md").write_text(
        "---\ntitle: Kael Brightaxe\ntags: [pc]\n---\n\n# Kael\n", encoding="utf-8"
    )
    (chars / "mira.md").write_text("# Mira\nno frontmatter\n", encoding="utf-8")

    ps = PlayerState(str(tmp_path / "players.db"))
    _seed_players(ps, str(tmp_path))

    kael = ps.get("kael")
    assert kael is not None
    assert kael["name"] == "Kael Brightaxe"
    assert kael["sheet"] == "characters/kael.md"
    mira = ps.get("mira")
    assert mira is not None
    assert mira["name"] == "Mira"


def test_seed_players_noop_without_dir(tmp_path: Path) -> None:
    ps = PlayerState(str(tmp_path / "players.db"))
    _seed_players(ps, str(tmp_path / "missing"))
    _seed_players(ps, "")
    assert ps.all_players() == []


def test_reseed_preserves_accumulated_state(tmp_path: Path) -> None:
    ps = PlayerState(str(tmp_path / "players.db"))
    ps.seed([{"id": "kael", "name": "Kael", "sheet": "characters/kael.md"}])
    ps.update("kael", hp="12/18")
    ps.record_card_done("kael", {"id": "c1", "kind": "loot", "title": "t", "items": ["5 gp"]})
    ps.seed(
        [{"id": "kael", "name": "Kael Brightaxe", "sheet": "characters/kael.md"}]
    )
    row = ps.get("kael")
    assert row is not None
    assert row["hp"] == "12/18"
    assert [d["id"] for d in row["done_cards"]] == ["c1"]
    assert row["inventory"] == ["5 gp"]


# -- mark-done updates (UI path) ---------------------------------------------


async def test_mark_card_done_records_state(tmp_path: Path) -> None:
    ps = PlayerState(str(tmp_path / "players.db"))
    ps.seed([{"id": "kael", "name": "Kael", "sheet": ""}])
    events: list[dict] = []
    engine = _engine(tmp_path, ps, events)
    card = _card()
    engine._active_cards[card.id] = card

    assert await engine.mark_card_done(card.id) is True

    row = ps.get("kael")
    assert row is not None
    assert [d["id"] for d in row["done_cards"]] == ["c1"]
    assert row["inventory"] == ["5 gp", "coded note"]
    assert any(e["type"] == "card_done" for e in events)
    # Idempotent: a second mark is a no-op and does not duplicate the ref.
    assert await engine.mark_card_done(card.id) is False
    row = ps.get("kael")
    assert row is not None and len(row["done_cards"]) == 1


# -- AI-observed updates (monitor path) ---------------------------------------


async def test_monitor_card_done_updates_state(tmp_path: Path) -> None:
    """The transcript monitor auto-marking a card done (AI observed the
    resolution in play) must update the store exactly like the UI button."""
    ps = PlayerState(str(tmp_path / "players.db"))
    ps.seed([{"id": "mira", "name": "Mira", "sheet": ""}])
    events: list[dict] = []
    engine = _engine(tmp_path, ps, events)
    card = _card(pid="mira", items=["potion"])
    engine._active_cards[card.id] = card

    await engine._on_monitor_action({"action": "card_done", "card_id": card.id})

    row = ps.get("mira")
    assert row is not None
    assert [d["id"] for d in row["done_cards"]] == ["c1"]
    assert row["inventory"] == ["potion"]
    assert card.status == "done"


def test_record_card_done_creates_unknown_player(tmp_path: Path) -> None:
    ps = PlayerState(str(tmp_path / "players.db"))
    ps.record_card_done("stranger", {"id": "c9", "kind": "loot", "title": "x", "items": []})
    row = ps.get("stranger")
    assert row is not None
    assert [d["id"] for d in row["done_cards"]] == ["c9"]
