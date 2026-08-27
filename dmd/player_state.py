"""Per-player state store (SQLite).

Tracks light per-player state (name, sheet path, HP, inventory, knowledge,
done-cards) so the agent can reason about who already has what and so the
monitor can auto-mark cards done. Seeded from character sheets at init;
updated as cards are resolved. Mark-done is append-only bookkeeping — it never
deletes a card.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id   TEXT PRIMARY KEY,
    name        TEXT,
    sheet       TEXT,
    hp          TEXT,
    inventory   TEXT,
    knowledge   TEXT,
    done_cards  TEXT,
    updated_at  REAL
);
"""


def _load(blob: Optional[str], default: Any) -> Any:
    if not blob:
        return default
    try:
        return json.loads(blob)
    except (TypeError, ValueError):
        return default


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class PlayerState:
    """Tiny per-player state table over SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # -- seeding -----------------------------------------------------------
    def seed(self, players: List[Dict[str, Any]]) -> None:
        """Insert or update players from a list of {id, name, sheet, hp?}."""
        with self._lock:
            for p in players:
                pid = str(p.get("id") or p.get("name") or "")
                if not pid:
                    continue
                self._conn.execute(
                    """
                    INSERT INTO players (player_id, name, sheet, hp, inventory, knowledge, done_cards, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_id) DO UPDATE SET
                        name = excluded.name,
                        sheet = excluded.sheet,
                        hp = COALESCE(excluded.hp, players.hp),
                        updated_at = excluded.updated_at
                    """,
                    (
                        pid,
                        p.get("name", pid),
                        p.get("sheet", ""),
                        p.get("hp", ""),
                        _dump(p.get("inventory", [])),
                        _dump(p.get("knowledge", [])),
                        _dump([]),
                        time.time(),
                    ),
                )
            self._conn.commit()

    # -- reads -------------------------------------------------------------
    def get(self, player_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM players WHERE player_id = ?", (player_id,)
            ).fetchone()
        return self._row(row) if row else None

    def all_players(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM players ORDER BY name").fetchall()
        return [self._row(r) for r in rows]

    def to_world_map_rows(self) -> List[Dict[str, Any]]:
        """Rows shaped for build_world_map's players section."""
        out: List[Dict[str, Any]] = []
        for p in self.all_players():
            out.append({"id": p["player_id"], "name": p["name"], "sheet": p["sheet"], "hp": p.get("hp")})
        return out

    # -- writes ------------------------------------------------------------
    def update(self, player_id: str, **fields: Any) -> None:
        allowed = {"name", "sheet", "hp", "inventory", "knowledge"}
        sets = []
        vals: list = []
        for k, v in fields.items():
            if k not in allowed:
                continue
            col = k
            val = _dump(v) if k in ("inventory", "knowledge") else v
            sets.append(f"{col} = ?")
            vals.append(val)
        if not sets:
            return
        sets.append("updated_at = ?")
        vals.append(time.time())
        vals.append(player_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE players SET {', '.join(sets)} WHERE player_id = ?", vals
            )
            self._conn.commit()

    def record_card_done(self, player_id: str, card: Dict[str, Any]) -> None:
        """Append a resolved card to the player's knowledge (mark-done bookkeeping)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT player_id, done_cards, inventory FROM players WHERE player_id = ?",
                (player_id,),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO players (player_id, name, sheet, hp, inventory, knowledge, done_cards, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        player_id,
                        card.get("player_id", player_id),
                        "",
                        "",
                        _dump([]),
                        _dump([]),
                        _dump([self._card_ref(card)]),
                        time.time(),
                    ),
                )
            else:
                done = _load(row[1], [])
                if not any(d.get("id") == card.get("id") for d in done):
                    done.append(self._card_ref(card))
                self._conn.execute(
                    "UPDATE players SET done_cards = ?, updated_at = ? WHERE player_id = ?",
                    (_dump(done), time.time(), player_id),
                )
            self._conn.commit()

    @staticmethod
    def _card_ref(card: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": card.get("id", ""),
            "kind": card.get("kind", ""),
            "title": card.get("title", ""),
            "items": card.get("items", []),
            "t": card.get("t", time.time()),
        }

    @staticmethod
    def _row(row: tuple) -> Dict[str, Any]:
        return {
            "player_id": row[0],
            "name": row[1],
            "sheet": row[2],
            "hp": row[3],
            "inventory": _load(row[4], []),
            "knowledge": _load(row[5], []),
            "done_cards": _load(row[6], []),
            "updated_at": row[7],
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
