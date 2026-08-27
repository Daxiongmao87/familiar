"""Build a compact 'world map' the worker agent reads as orientation.

The world map is a short, structured markdown orientation over the campaign:
folder/file structure, canonical entities, players (character sheets), and
repo tools the agent can invoke. It is produced at init time and refreshed
when the index changes; the agent receives it in its prompt and may also
`repo_read` any file it names.

Each section degrades gracefully (missing store / empty table -> omitted) so
the map is always a usable, if partial, orientation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .index_store import IndexStore
from .types import Entity

# Keep the map small: it lands in every agent prompt.
_MAX_ENTITIES = 40
_MAX_FILES = 60
_MAX_OVERVIEW_CHARS = 1200


def _structure_section(docs: list) -> str:
    """Group indexed files by top-level folder into a compact tree."""
    if not docs:
        return ""
    groups: "OrderedDict[str, list[str]]" = OrderedDict()
    for d in docs:
        path = d.path.replace("\\", "/")
        top = path.split("/", 1)[0] if "/" in path else "."
        groups.setdefault(top, []).append(f"- `{path}`")
    lines = ["## Structure", ""]
    shown = 0
    for top, items in groups.items():
        lines.append(f"**{top}/**")
        for item in items:
            lines.append(item)
            shown += 1
            if shown >= _MAX_FILES:
                break
        if shown >= _MAX_FILES:
            break
    return "\n".join(lines) + "\n"


def _entities_section(entities: List[Entity]) -> str:
    if not entities:
        return ""
    lines = ["## Entities", ""]
    for e in entities[:_MAX_ENTITIES]:
        desc = (e.description or "").strip().replace("\n", " ")
        if len(desc) > 120:
            desc = desc[:117] + "..."
        src = e.source_files[0] if e.source_files else "?"
        line = f"- **{e.name}** ({e.type})"
        if desc:
            line += f": {desc}"
        line += f"  [`{src}`]"
        lines.append(line)
    if len(entities) > _MAX_ENTITIES:
        lines.append(f"- … {len(entities) - _MAX_ENTITIES} more (use `retrieve`)")
    return "\n".join(lines) + "\n"


def _players_section(players: List[Dict[str, Any]]) -> str:
    if not players:
        return ""
    lines = ["## Players", ""]
    for p in players:
        name = p.get("name", p.get("id", "?"))
        sheet = p.get("sheet", "")
        line = f"- **{name}** (id `{p.get('id', name)}`)"
        if sheet:
            line += f" — full sheet: `repo_read('{sheet}')`"
        hp = p.get("hp")
        if hp:
            line += f" — hp {hp}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _tools_section(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""
    lines = ["## Tools (runnable via the `run_tool` tool)", ""]
    for t in tools:
        desc = (t.get("description") or "").strip().replace("\n", " ")
        line = f"- `{t.get('name')}`"
        if desc:
            line += f": {desc[:100]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _overview_section(project_path: str) -> str:
    """Pull a short campaign overview from an obvious top-level file."""
    import os

    for candidate in ("campaign.md", "overview.md", "README.md", "campaign_overview.md"):
        p = os.path.join(project_path, candidate)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read(_MAX_OVERVIEW_CHARS + 200)
                return "## Campaign Overview\n\n" + text.strip()[:_MAX_OVERVIEW_CHARS] + "\n"
            except OSError:
                continue
    return ""


def build_world_map(
    project_path: str,
    store: Optional[IndexStore],
    players: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Assemble the compact world-map orientation text."""
    sections: List[str] = []
    if store is not None:
        try:
            docs = store.all_documents()
        except Exception:
            docs = []
        s = _structure_section(docs)
        if s:
            sections.append(s)
        try:
            entities = store.all_entities()
        except Exception:
            entities = []
        s = _entities_section(entities)
        if s:
            sections.append(s)
    p = _players_section(players or [])
    if p:
        sections.append(p)
    t = _tools_section(tools or [])
    if t:
        sections.append(t)
    o = _overview_section(project_path)
    if o:
        sections.append(o)
    header = "# World Map\n\nOrientation for this campaign. Use the tools to verify details; cite what you read.\n"
    return header + "\n".join(sections)
