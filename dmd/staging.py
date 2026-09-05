"""Predictive-retrieval staging cache ("Predictive RAG", Priority-1 design).

Idea: as the conversation develops the transcript monitor predicts which
entities the next beats will need; their campaign excerpts are pre-embedded and
retrieved *ahead of* the turn that would otherwise pay that latency inline.
When a real turn arrives, context assembly pulls from this cache first and only
falls back to the normal retrieval path on a miss.

Doctrine (the guardrails the owner-ordered design imposes):

* Advisory only. Staged excerpts are context for the worker agent; they NEVER
  mutate canonical state. The event log stays the sole truth.
* A miss is not an error and must never add latency: ``lookup`` is a pure
  in-memory dict read, so a cache miss costs the answer nothing beyond the
  normal path it was already going to take.
* Bounded RAM. Entries are an LRU with a TTL and a hard ``max_entries`` — the
  memory guard for this cache — so a long session can't grow it without limit.
* Observability. Every turn records a hit/miss so prediction precision can be
  tuned; prefetches are logged with the entity that triggered them.

The retrieval itself reuses the same primitives as ``WorkerAgent._tool_retrieve``
(local ``Embedder`` -> ``IndexStore.search``), so staged excerpts and agent-
fetched excerpts are identical in shape and provenance.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterable

logger = logging.getLogger(__name__)


def normalize_key(entity: str) -> str:
    """Canonical cache key for a predicted/mentioned entity (case/space-folded)."""
    return " ".join((entity or "").strip().lower().split())


@dataclass(slots=True)
class StagedEntry:
    """One prefetched entity's excerpts, stamped for TTL expiry."""

    key: str
    entity: str
    excerpts: list[dict[str, Any]] = field(default_factory=list)
    t_staged: float = 0.0


class StagedContext:
    """LRU + TTL cache of ``entity -> retrieved excerpts`` for advisory context.

    Pure in-memory and non-blocking on the read side; writes (``put``) happen
    only after a background prefetch completes, never on the answer path.
    """

    def __init__(self, ttl_s: float = 120.0, max_entries: int = 32) -> None:
        self._ttl_s = max(0.0, float(ttl_s))
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, StagedEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def put(self, entity: str, excerpts: list[dict[str, Any]]) -> None:
        """Store ``excerpts`` for ``entity``, evicting least-recently-used on overflow.

        An empty excerpt list is not cached (there is nothing to inject, and
        storing the miss would let a later hit report stale emptiness).
        """
        key = normalize_key(entity)
        if not key or not excerpts:
            return
        self._entries[key] = StagedEntry(
            key=key,
            entity=entity,
            excerpts=list(excerpts),
            t_staged=time.monotonic(),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get(self, entity: str) -> list[dict[str, Any]] | None:
        """Return live excerpts for ``entity`` (LRU touch + hit/miss count)."""
        key = normalize_key(entity)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if self._ttl_s and (time.monotonic() - entry.t_staged) > self._ttl_s:
            self._entries.pop(key, None)
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry.excerpts

    def has(self, entity: str) -> bool:
        """Non-counting membership probe (used to skip redundant prefetches)."""
        key = normalize_key(entity)
        entry = self._entries.get(key)
        if entry is None:
            return False
        if self._ttl_s and (time.monotonic() - entry.t_staged) > self._ttl_s:
            return False
        return True

    def lookup(
        self, entities: Iterable[str]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Resolve ``entities`` against the cache.

        Returns ``(excerpts, matched_entities)`` for cache HITS only; a miss is
        simply absent. Never retrieves (that would move work onto the answer
        path), so a full miss costs nothing but a dict lookup.
        """
        merged: list[dict[str, Any]] = []
        matched: list[str] = []
        seen_keys: set[str] = set()
        for ent in entities:
            key = normalize_key(ent)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            excerpts = self.get(ent)
            if excerpts:
                matched.append(ent)
                for e in excerpts:
                    merged.append({**e, "staged_for": ent})
        return merged, matched

    def snapshot(self) -> dict[str, Any]:
        """Cheap introspection for the session log / UI."""
        return {
            "size": len(self._entries),
            "max_entries": self._max_entries,
            "ttl_s": self._ttl_s,
            "hits": self.hits,
            "misses": self.misses,
            "keys": list(self._entries.keys()),
        }

    def clear(self) -> None:
        self._entries.clear()


def render_staged_block(
    excerpts: list[dict[str, Any]], max_chars: int = 6000
) -> str:
    """Format staged excerpts as an advisory context block for the agent prompt.

    Empty input renders an empty string (the caller then sends the turn down the
    normal path unchanged). Output is capped so a large prefetch cannot blow the
    agent's context budget.
    """
    if not excerpts:
        return ""
    lines: list[str] = []
    used = 0
    for e in excerpts:
        source = str(e.get("source", "?"))
        staged_for = str(e.get("staged_for", "")).strip()
        body = str(e.get("excerpt", e.get("text", ""))).strip()
        if not body:
            continue
        tag = f"[staged for {staged_for}] " if staged_for else ""
        block = f"- {source} {tag}{body[:400]}"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    if not lines:
        return ""
    return "\n".join(lines)


async def prefetch_entity(
    entity: str,
    embedder: Any,
    store: Any,
    gw: Any,
    *,
    k: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve campaign excerpts for one predicted ``entity``.

    Mirrors ``WorkerAgent._tool_retrieve`` (local embedding when available,
    gateway embeddings as fallback) so staged and agent-fetched excerpts share
    shape and provenance. Returns a (possibly empty) excerpt list; every failure
    path degrades to ``[]`` — a prediction that cannot be staged is simply a
    miss at injection time, never a raised error on the (background) caller.
    """
    if store is None or not entity.strip():
        return []
    try:
        try:
            if embedder is not None:
                vec = embedder.embed([entity])[0]
            elif gw is not None:
                vec = (await gw.embed([entity]))[0]
            else:
                vec = None
            hits = store.search(embedding=vec, query_text=entity, k=k)
        except Exception as exc:
            logger.debug("staging prefetch failed for %r: %s", entity, exc)
            return []
        return [
            {
                "source": h.source,
                "score": round(float(h.score), 4),
                "excerpt": h.text[:400],
            }
            for h in hits
        ]
    except Exception:
        return []
