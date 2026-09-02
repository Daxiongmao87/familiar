"""Persistence and hybrid retrieval over SQLite + sqlite-vec."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

import numpy as np

from dmd.scanner import Chunk, DocFile
from dmd.types import Entity, Retrieved

_RRF_K = 60
_RRF_OVERFETCH = 4


def _json_load(blob: str | None, default: Any) -> Any:
    if not blob:
        return default
    try:
        return json.loads(blob)
    except (TypeError, ValueError):
        return default


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class IndexStore:
    """SQLite + sqlite-vec store for docs, chunks, vectors, entities, FTS."""

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            self._conn.load_extension(sqlite_vec.loadable_path())
        except Exception as exc:
            self._conn.close()
            raise RuntimeError(f"failed to load sqlite-vec extension: {exc}") from exc
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS docs (
                    path  TEXT PRIMARY KEY,
                    title TEXT,
                    mtime REAL,
                    meta  TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     TEXT PRIMARY KEY,
                    doc_path     TEXT,
                    text         TEXT,
                    heading_path TEXT,
                    pos          INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_path);

                CREATE TABLE IF NOT EXISTS entities (
                    canonical     TEXT PRIMARY KEY,
                    aliases       TEXT,
                    etype         TEXT,
                    weight        REAL,
                    source_files  TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='rowid'
                );

                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END;
                """
            )
            self._conn.commit()
            dim = self._meta_get("vec_dim")
            if dim:
                self._create_vec_table(int(dim))

    def get_meta(self, key: str) -> str | None:
        """Public accessor for index metadata (embedding model identity, etc.)."""
        with self._lock:
            return self._meta_get(key)

    def set_meta(self, key: str, value: str) -> None:
        """Public writer for index metadata."""
        with self._lock:
            self._meta_set(key, value)
            self._conn.commit()

    def list_paths(self) -> list[str]:
        """All indexed document paths (public API; replaces direct _conn access)."""
        with self._lock:
            return [
                row["path"] for row in self._conn.execute("SELECT path FROM docs").fetchall()
            ]

    def counts(self) -> dict[str, int]:
        """Row counts for docs, chunks and entities (public status surface)."""
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) AS n FROM docs").fetchone()["n"]
            chunks = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            entities = self._conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"]
        return {"docs": int(docs), "chunks": int(chunks), "entities": int(entities)}

    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _create_vec_table(self, dim: int) -> None:
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        stored = self._meta_get("vec_dim")
        if existing is None:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE chunks_vec USING vec0("
                f"chunk_id TEXT, embedding float[{dim}])"
            )
            self._meta_set("vec_dim", str(dim))
            self._conn.commit()
        elif stored is not None and int(stored) != dim:
            raise RuntimeError(
                f"embedding dim mismatch: stored={stored} new={dim}; "
                "drop the index or rebuild it before changing models"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_docs(self, docs: list[DocFile]) -> None:
        with self._lock:
            for d in docs:
                self._conn.execute(
                    "INSERT INTO docs(path,title,mtime,meta) VALUES(?,?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET "
                    "title=excluded.title, mtime=excluded.mtime, meta=excluded.meta",
                    (d.relpath, d.title, d.mtime, _json_dump(d.frontmatter)),
                )
            self._conn.commit()

    def replace_chunks_for_doc(self, doc_path: str, chunks: list[Chunk]) -> None:
        with self._lock:
            old_ids = [
                r["chunk_id"]
                for r in self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_path=?", (doc_path,)
                ).fetchall()
            ]
            for cid in old_ids:
                self._conn.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))
                try:
                    self._conn.execute(
                        "DELETE FROM chunks_vec WHERE chunk_id=?", (cid,)
                    )
                except sqlite3.OperationalError:
                    pass
            for c in chunks:
                self._conn.execute(
                    "INSERT INTO chunks(chunk_id,doc_path,text,heading_path,pos) "
                    "VALUES(?,?,?,?,?) ON CONFLICT(chunk_id) DO UPDATE SET "
                    "text=excluded.text, heading_path=excluded.heading_path, "
                    "pos=excluded.pos",
                    (
                        c.chunk_id,
                        doc_path,
                        c.text,
                        _json_dump(c.heading_path),
                        c.pos,
                    ),
                )
            self._conn.commit()

    def upsert_chunk_embeddings(
        self, items: list[tuple[str, np.ndarray]]
    ) -> None:
        if not items:
            return
        with self._lock:
            first = np.asarray(items[0][1], dtype=np.float32)
            if first.ndim != 1:
                raise ValueError(f"embedding must be 1D, got shape {first.shape}")
            dim = int(first.shape[0])
            self._create_vec_table(dim)
            for cid, vec in items:
                arr = np.asarray(vec, dtype=np.float32).ravel()
                if arr.shape[0] != dim:
                    raise ValueError(
                        f"embedding dim mismatch for {cid}: "
                        f"expected {dim} got {arr.shape[0]}"
                    )
                self._conn.execute(
                    "DELETE FROM chunks_vec WHERE chunk_id=?", (cid,)
                )
                self._conn.execute(
                    "INSERT INTO chunks_vec(chunk_id, embedding) VALUES(?, ?)",
                    (cid, arr.tobytes()),
                )
            self._conn.commit()

    def upsert_entities(self, entities: list[Entity]) -> None:
        with self._lock:
            for e in entities:
                self._conn.execute(
                    "INSERT INTO entities(canonical,aliases,etype,weight,source_files) "
                    "VALUES(?,?,?,?,?) ON CONFLICT(canonical) DO UPDATE SET "
                    "aliases=excluded.aliases, etype=excluded.etype, "
                    "weight=excluded.weight, source_files=excluded.source_files",
                    (
                        e.canonical,
                        _json_dump(e.aliases),
                        e.etype,
                        e.weight,
                        _json_dump(e.source_files),
                    ),
                )
            self._conn.commit()

    def get_entity(self, canonical: str) -> Entity | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE canonical=?", (canonical,)
            ).fetchone()
        if not row:
            return None
        return Entity(
            canonical=row["canonical"],
            aliases=_json_load(row["aliases"], []),
            etype=row["etype"],
            weight=row["weight"],
            source_files=_json_load(row["source_files"], []),
        )

    def all_entities(self) -> list[Entity]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY canonical"
            ).fetchall()
        return [
            Entity(
                canonical=r["canonical"],
                aliases=_json_load(r["aliases"], []),
                etype=r["etype"],
                weight=r["weight"],
                source_files=_json_load(r["source_files"], []),
            )
            for r in rows
        ]

    def stale_docs(self, current: dict[str, float]) -> list[str]:
        with self._lock:
            stored = {
                r["path"]: r["mtime"]
                for r in self._conn.execute("SELECT path,mtime FROM docs").fetchall()
            }
        out: list[str] = []
        for path, mtime in current.items():
            s = stored.get(path)
            if s is None or abs(float(s) - float(mtime)) > 1e-6:
                out.append(path)
        return sorted(out)

    def remove_doc(self, path: str) -> None:
        with self._lock:
            ids = [
                r["chunk_id"]
                for r in self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_path=?", (path,)
                ).fetchall()
            ]
            for cid in ids:
                self._conn.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))
                try:
                    self._conn.execute(
                        "DELETE FROM chunks_vec WHERE chunk_id=?", (cid,)
                    )
                except sqlite3.OperationalError:
                    pass
            self._conn.execute("DELETE FROM docs WHERE path=?", (path,))
            self._conn.commit()

    @staticmethod
    def _sanitize_fts(query: str) -> str:
        out: list[str] = []
        for tok in query.split():
            t = tok.strip().strip('"').strip("'")
            if not t:
                continue
            if any(c in t for c in '()"*'):
                continue
            out.append(f'"{t}"')
        return " ".join(out)

    def search(
        self,
        embedding: np.ndarray | None = None,
        query_text: str | None = None,
        k: int = 8,
    ) -> list[Retrieved]:
        if embedding is None and not query_text:
            return []
        scores: dict[str, float] = {}
        cap = max(k * _RRF_OVERFETCH, 16)

        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32).ravel()
            with self._lock:
                if self._meta_get("vec_dim") is None:
                    raise RuntimeError(
                        "no embeddings have been indexed yet; "
                        "call upsert_chunk_embeddings first"
                    )
                rows = self._conn.execute(
                    "SELECT chunk_id, distance FROM chunks_vec "
                    "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                    (vec.tobytes(), cap),
                ).fetchall()
            for rank, row in enumerate(rows):
                cid = row["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        if query_text:
            sanitized = self._sanitize_fts(query_text)
            if sanitized:
                with self._lock:
                    fts_rows = self._conn.execute(
                        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (sanitized, cap),
                    ).fetchall()
                rids = [r["rowid"] for r in fts_rows]
                rid_to_cid: dict[int, str] = {}
                if rids:
                    placeholders = ",".join("?" for _ in rids)
                    with self._lock:
                        id_rows = self._conn.execute(
                            f"SELECT chunk_id, rowid FROM chunks "
                            f"WHERE rowid IN ({placeholders})",
                            rids,
                        ).fetchall()
                    rid_to_cid = {r["rowid"]: r["chunk_id"] for r in id_rows}
                for rank, rid in enumerate(rids):
                    cid = rid_to_cid.get(rid)
                    if cid is None:
                        continue
                    scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        ids = [cid for cid, _ in ranked]
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            chunk_rows = self._conn.execute(
                f"SELECT chunk_id, doc_path, text FROM chunks "
                f"WHERE chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
        cdata = {r["chunk_id"]: (r["doc_path"], r["text"]) for r in chunk_rows}
        results: list[Retrieved] = []
        for cid, score in ranked:
            doc_path, text = cdata.get(cid, ("", ""))
            results.append(
                Retrieved(
                    doc_id=cid, source=doc_path, score=float(score), text=text
                )
            )
        return results