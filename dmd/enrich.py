"""LLM-driven entity extraction from scanned documents."""

from __future__ import annotations

from typing import Iterable

from .gateway import Gateway
from .scanner import DocFile
from .types import Entity

_VALID_ETYPES: frozenset[str] = frozenset({
    "character",
    "place",
    "item",
    "faction",
    "spell",
    "concept",
})

_EXTRACT_SYSTEM = (
    "you extract canon names/aliases/types from TTRPG worldbuilding notes; "
    "return strict JSON with canonical names, alias variants, entity types, "
    "and a narrative weight."
)

_EXTRACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "etype": {
                        "type": "string",
                        "enum": [
                            "character",
                            "place",
                            "item",
                            "faction",
                            "spell",
                            "concept",
                        ],
                    },
                    "weight": {"type": "number"},
                },
                "required": ["canonical"],
            },
        }
    },
    "required": ["entities"],
}


def _doc_text(doc: DocFile, max_chars: int) -> str:
    body = doc.content or ""
    if len(body) > max_chars:
        body = body[:max_chars]
    return f"# {doc.title}\n\n{body}"


def _merge(
    by_key: dict[str, Entity],
    canonical: str,
    aliases: Iterable[str],
    etype: str,
    weight: float,
    source_files: list[str],
) -> None:
    key = canonical.lower()
    aliases_list = [a for a in aliases if a]
    existing = by_key.get(key)
    if existing is None:
        by_key[key] = Entity(
            canonical=canonical,
            aliases=list(aliases_list),
            etype=etype,
            weight=weight,
            source_files=list(source_files),
        )
        return
    seen_aliases = {a.lower() for a in existing.aliases}
    for a in aliases_list:
        if a.lower() not in seen_aliases:
            existing.aliases.append(a)
            seen_aliases.add(a.lower())
    if weight > existing.weight:
        existing.weight = weight
    seen_sources = set(existing.source_files)
    for sf in source_files:
        if sf not in seen_sources:
            existing.source_files.append(sf)
            seen_sources.add(sf)


async def extract_entities(
    docs: list[DocFile],
    gw: Gateway,
    *,
    batch_docs: int = 6,
) -> list[Entity]:
    """Extract canonical entities from docs via the synthesis role."""
    by_key: dict[str, Entity] = {}
    if not docs:
        return []

    n_batches = 0
    n_failed = 0
    for i in range(0, len(docs), batch_docs):
        batch = docs[i : i + batch_docs]
        user_content = "\n\n---\n\n".join(_doc_text(d, 4000) for d in batch)
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        try:
            result = await gw.chat(
                "synthesis",
                messages=messages,
                json_schema=_EXTRACT_SCHEMA,
                temperature=0.2,
            )
        except Exception:
            n_batches += 1
            n_failed += 1
            continue
        if not isinstance(result, dict):
            n_batches += 1
            n_failed += 1
            continue
        raw_ents = result.get("entities")
        if not isinstance(raw_ents, list):
            continue
        source_files = [d.relpath for d in batch]
        for item in raw_ents:
            if not isinstance(item, dict):
                continue
            canonical = item.get("canonical")
            if not isinstance(canonical, str):
                continue
            canonical = canonical.strip()
            if not canonical:
                continue
            raw_aliases = item.get("aliases")
            aliases: list[str] = []
            if isinstance(raw_aliases, list):
                for a in raw_aliases:
                    if isinstance(a, str):
                        a = a.strip()
                        if a:
                            aliases.append(a)
            etype = item.get("etype")
            if not isinstance(etype, str) or etype not in _VALID_ETYPES:
                etype = "unknown"
            raw_weight = item.get("weight", 1.0)
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                weight = 1.0
            _merge(by_key, canonical, aliases, etype, weight, source_files)

    entities = sorted(by_key.values(), key=lambda e: -e.weight)[:500]
    if n_batches > 0 and n_failed == n_batches and not entities:
        raise RuntimeError(
            f"entity extraction failed for all {n_failed}/{n_batches} batches; "
            "synthesis endpoint unreachable or returned unparseable output"
        )
    return entities
