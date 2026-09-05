"""Worker agent: an agentic tool loop that does DM work and produces a verified output.

v2 core. The agent is NOT a fixed pipeline: it is a loop that, given a task
(trigger / manual query / monitor observation) + the world map + recent
transcript, reasons about what to do and calls tools to look things up
(retrieve the index, read a campaign file, web search/fetch, run a repo tool).
It verifies its output against the triggering transcript and returns either a
durable **card** or a short **ephemeral** context note.

The protocol is text-based (not native function calling) so it works reliably
with local OpenAI-compatible models: the model either emits
``{"tool": "<name>", "args": {...}}`` to call a tool, or emits the final
answer object. Results are fed back as user messages and the loop repeats
until a final answer, the tool budget, or the wall-clock budget is exhausted.
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .config import AgentConfig
from .gateway import Gateway
from .index_store import IndexStore

# ---------------------------------------------------------------------------
# Result


@dataclass
class AgentResult:
    """What a worker-agent run produced."""

    tier: str  # "ephemeral" | "card"
    card: dict[str, Any] | None = None  # card tier: {kind,title,body_md,player_ids,items}
    text: str | None = None  # ephemeral tier: short scene-relevant note
    tool_calls: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# JSON extraction (robust to fences / surrounding prose)
# ---------------------------------------------------------------------------


def _extract_json(text) -> dict[str, Any] | None:
    """Parse the first JSON object out of a model response (str or pre-parsed dict)."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
        t = t.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except (json.JSONDecodeError, ValueError):
                        break
        start = t.find("{", start + 1)
    hermes = _extract_hermes_call(t)
    return hermes


# Pattern assembled from fragments: literal XML tool-call tag sequences in
# source files corrupt some OpenAI-compatible tool-calling layers (observed
# 2026-09-05: three agent runs killed by 'Expected function.name to be a
# string' after reading this file). Keep this region tag-literal-free.
_TC_OPEN = "<" + "tool_call" + ">"
_TC_CLOSE = "<" + "/tool_call" + ">"
_XML_CALL_RE = re.compile(
    _TC_OPEN + r"\s*([A-Za-z_][\w.-]*)\s*(.*?)\s*" + _TC_CLOSE, re.DOTALL
)
_AK_OPEN = "<" + "arg_key" + ">"
_AK_CLOSE = "<" + "/arg_key" + ">"
_AV_OPEN = "<" + "arg_value" + ">"
_AV_CLOSE = "<" + "/arg_value" + ">"
_XML_ARGS_RE = re.compile(
    _AK_OPEN + r"\s*(.*?)\s*" + _AK_CLOSE + r"\s*"
    + _AV_OPEN + r"\s*(.*?)\s*" + _AV_CLOSE, re.DOTALL
)


def _extract_hermes_call(t: str) -> dict[str, Any] | None:
    """Parse the Hermes-family XML tool-call format off the wire.

    The configured local endpoints (e.g. ling-3.0-tiny) emit
    ``HERMES-XML tool-call wire format`` — name + key/value args.
    regardless of the JSON protocol asked for in the prompt. The agent loop
    must read what the endpoint actually speaks (SPEC §12 swappable
    endpoints), or every tool turn looks like unparseable output — the
    2026-09-05 live defect where cards died as "agent budget exhausted"
    with 0 tool calls.
    """
    m = _XML_CALL_RE.search(t)
    if m is None:
        return None
    name = m.group(1)
    args: dict[str, Any] = {}
    for k, v in _XML_ARGS_RE.findall(m.group(2)):
        try:
            args[k] = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            args[k] = v
    if not args:
        raw = m.group(2).strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
            except (json.JSONDecodeError, ValueError):
                pass
    return {"tool": name, "args": args}


def _content_str(content: Any) -> str:
    """Stringify a model response (str or pre-parsed dict) for a message body."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


# ---------------------------------------------------------------------------
# Worker agent
# ---------------------------------------------------------------------------

# Structured-output schema used when the model must commit to a final CARD.
# Deliberately schema-only: DCs are plain integers supplied by the model per
# scene, never baked rule constants (SPEC §2: no baked rules).
_CARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["skill_table", "lore", "rules", "info", "npc", "location", "loot", "ruling", "transcript_notice"]},
        "title": {"type": "string"},
        "body_md": {"type": "string"},
        "player_ids": {"type": "array", "items": {"type": "string"}},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "dc_find": {"type": ["integer", "null"]},
                    "notes": {"type": "string"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["kind", "title", "body_md"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are a worker agent for a DM copilot. You do REAL work:
you answer the DM's trigger and produce a verified output, using your tools.
You do not guess; you look things up.

To call a tool, respond with EXACTLY one JSON object and nothing else:
  {"tool": "<name>", "args": {...}}
Available tools:
- retrieve: {"query": "..."} — search the campaign index (worldbuilding notes) for relevant excerpts.
- repo_read: {"path": "..."} — read a campaign file by repo-relative path (see the World Map structure).
- web_search: {"query": "..."} — search the web for a rule or lookup (may be unavailable; then it degrades).
- web_fetch: {"url": "..."} — fetch a URL's text (may be unavailable).
- run_tool: {"name": "..."} — run a repo-provided tool script (see Tools in the World Map).

When you have enough to answer, respond with the FINAL JSON (no "tool" key):
- CARD: {"kind": "...", "title": "...", "body_md": "...", "player_ids": [...], "items": [...]}
  - kind is one of: skill_table, lore, rules, info, npc, location, loot, ruling, transcript_notice
  - body_md is markdown. For a loot table / skill check table use a markdown table, and set "items" to a list of {"name": "...", "quantity": <int>, "dc_find": <int or null>, "notes": "..."}.
  - player_ids: the ids of players this card concerns (from the Players section), else [].
- EPHEMERAL: {"text": "..."} — a short (1-3 sentence), verified, scene-relevant note.

Rules:
- Base your output ONLY on the World Map, the transcript, and tool results.
- Verify your output against the triggering transcript before answering.
- If a fact cannot be verified, mark it "unverified" in the output; never invent it.
- MANDATORY GROUNDING: if the TASK asks for a RULING or RULES card, you MUST
  call web_search (or retrieve) at least once BEFORE the final card. A ruling
  produced from memory alone is not verified — the DM needs the actual rule
  source. State the source (URL or campaign file) in body_md when you can.
- Respond with JSON only — no prose outside the JSON object."""


class WorkerAgent:
    """A text-protocol agentic loop over a model + campaign capabilities."""

    def __init__(
        self,
        gw: Gateway,
        store: IndexStore,
        project_path: str,
        cfg: AgentConfig,
        world_map: str = "",
        embedder: Any | None = None,  # local Embedder for query vectors (matches init)
        tool_registry: Any | None = None,
        player_names: dict[str, str] | None = None,
    ) -> None:
        self.gw = gw
        self.store = store
        self.project_path = project_path
        self.cfg = cfg
        self.world_map = world_map
        self.embedder = embedder
        self.tool_registry = tool_registry
        self.player_names = player_names or {}
        self._http = None  # lazy httpx client

    # -- public ------------------------------------------------------------
    async def run(
        self,
        task: str,
        tier: str,
        trigger_portion: str = "",
        transcript: str = "",
        staged_block: str = "",
    ) -> AgentResult:
        """Run the agent loop for a task at the given tier.

        ``staged_block`` is optional advisory context pre-fetched ahead of this
        turn ("Predictive RAG"): already-embedded excerpts for entities the
        monitor predicted the conversation would need. It is a head start the
        model may use or ignore; it never mutates state and an empty block
        leaves the normal path unchanged.
        """
        role = "synthesis" if tier == "card" else "fast"
        max_calls = (
            self.cfg.max_tool_calls
            if tier == "card"
            else self.cfg.ephemeral_max_tool_calls
        )
        try:
            return await asyncio.wait_for(
                self._loop(task, tier, role, trigger_portion, transcript, max_calls, staged_block),
                self.cfg.agent_timeout_s,
            )
        except asyncio.TimeoutError:
            return AgentResult(tier=tier, error="agent budget exhausted (timeout)")

    # -- loop --------------------------------------------------------------
    async def _loop(
        self,
        task: str,
        tier: str,
        role: str,
        trigger_portion: str,
        transcript: str,
        max_calls: int,
        staged_block: str = "",
    ) -> AgentResult:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._task_message(task, tier, trigger_portion, transcript, staged_block)},
        ]
        tool_calls = 0
        for _i in range(max_calls + 1):
            content = await self.gw.chat(role, messages, json_schema=None, temperature=0.2)
            parsed = _extract_json(content)
            if parsed is None:
                # Model produced no parseable JSON; nudge it once and retry.
                messages.append({"role": "assistant", "content": _content_str(content)})
                messages.append(
                    {
                        "role": "user",
                        "content": "Respond with a single valid JSON object only (a tool call or the final answer).",
                    }
                )
                continue
            if "tool" in parsed:
                name = parsed.get("tool")
                args = parsed.get("args", {})
                if name not in self._tool_names():
                    result: Any = {"error": f"unknown tool '{name}'; available: {sorted(self._tool_names())}"}
                else:
                    try:
                        result = await self._run_tool(name, args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                tool_calls += 1
                messages.append({"role": "assistant", "content": _content_str(content)})
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL RESULT for {name}:\n{json.dumps(result, ensure_ascii=False)[:6000]}",
                    }
                )
                continue
            # Final answer
            if tier == "card":
                if not parsed.get("body_md") and not parsed.get("title"):
                    # The model returned something that is not a usable card.
                    # First nudge in prose; if it still won't shape up, re-ask
                    # through the gateway's structured-output path so the card
                    # is forced to validate against the card schema.
                    messages.append({"role": "assistant", "content": _content_str(content)})
                    messages.append(
                        {
                            "role": "user",
                            "content": 'Respond with the final CARD JSON: {"kind","title","body_md","player_ids","items"}.',
                        }
                    )
                    content = await self.gw.chat(
                        role, messages, json_schema=_CARD_SCHEMA, temperature=0.2
                    )
                    parsed = _extract_json(content)
                    if not (parsed or {}).get("body_md") and not (parsed or {}).get("title"):
                        # Even structured output failed; salvage whatever JSON
                        # came back so the DM still sees a card rather than
                        # nothing (SPEC: cards arrive readable, never vanish).
                        if parsed:
                            return AgentResult(
                                tier="card",
                                card=self._normalize_card(parsed),
                                tool_calls=tool_calls,
                                error="structured card re-ask produced partial card",
                            )
                        continue
                # Mandatory-grounding guard: a rules/ruling card produced with
                # zero tool calls is unverified (ling-tiny answers from memory).
                # Push it back through the loop once so it actually searches.
                kind = str((parsed or {}).get("kind", ""))
                if (
                    kind in ("rules", "ruling")
                    and tool_calls == 0
                    and _i < max_calls
                ):
                    messages.append({"role": "assistant", "content": _content_str(content)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "This RULING card must be grounded. Call web_search "
                                "now to find the actual rule source, then produce the "
                                "final CARD JSON citing it."
                            ),
                        }
                    )
                    continue
                return AgentResult(
                    tier="card",
                    card=self._normalize_card(parsed or {}),
                    tool_calls=tool_calls,
                )
            # ephemeral tier
            text = parsed.get("text") or parsed.get("body_md") or ""
            if not text:
                return AgentResult(tier="ephemeral", text="", tool_calls=tool_calls, error="no text produced")
            return AgentResult(tier="ephemeral", text=str(text).strip(), tool_calls=tool_calls)
        # Budget exhausted: force one final answer.
        messages.append(
            {
                "role": "user",
                "content": "Tool budget exhausted. Answer now with the final JSON only, based on what you have.",
            }
        )
        content = await self.gw.chat(role, messages, json_schema=None, temperature=0.2)
        parsed = _extract_json(content)
        if tier == "card":
            if parsed and (parsed.get("body_md") or parsed.get("title")):
                return AgentResult(tier="card", card=self._normalize_card(parsed), tool_calls=tool_calls)
            return AgentResult(tier="card", tool_calls=tool_calls, error="no card produced within budget")
        text = (parsed or {}).get("text") or _content_str(content).strip()[:400]
        return AgentResult(tier="ephemeral", text=text or "(no context produced)", tool_calls=tool_calls)

    def _task_message(
        self,
        task: str,
        tier: str,
        trigger_portion: str,
        transcript: str,
        staged_block: str = "",
    ) -> str:
        parts = [f"TASK: {task}", "", f"TIER: {tier}"]
        if trigger_portion:
            parts += ["", "TRIGGERING TRANSCRIPT (what this answers):", trigger_portion]
        if transcript:
            # Cap to a recent window: the card only needs recent context, and re-sending the
            # whole rolling transcript on every loop call inflates context (and latency) unboundedly.
            _MAX_TRANSCRIPT = 4000
            parts += ["", "RECENT TRANSCRIPT:", transcript[-_MAX_TRANSCRIPT:]]
        if staged_block:
            # Advisory head start from the predictive-staging cache: already
            # retrieved excerpts for entities the monitor predicted. The model
            # may use them instead of (or before) calling retrieve.
            parts += ["", "PRE-STAGED CONTEXT (advisory, retrieved ahead of this turn):", staged_block]
        if transcript:
            parts += ["", "WORLD MAP:", self.world_map]
        return "\n".join(parts)

    def _normalize_card(self, parsed: dict[str, Any]) -> dict[str, Any]:
        items = parsed.get("items")
        if not isinstance(items, list):
            items = []
        # The model often puts the skill-check table only in body_md and omits
        # structured items. Derive items from a markdown table so rows stay
        # addressable (SPEC §9) and the monitor can match "we find the silver
        # dagger" to the right card by item name.
        if not items:
            items = _items_from_table(str(parsed.get("body_md", "")))
        player_ids = parsed.get("player_ids")
        if not isinstance(player_ids, list):
            player_ids = []
        return {
            "kind": str(parsed.get("kind", "info")),
            "title": str(parsed.get("title", "Note")),
            "body_md": str(parsed.get("body_md", "")),
            "player_ids": [str(p) for p in player_ids][:8],
            "items": items[:40],
        }

    def _tool_names(self) -> set:
        names = {"retrieve", "repo_read", "web_search", "web_fetch"}
        if self.tool_registry is not None:
            names.add("run_tool")
        return names

    # -- tools -------------------------------------------------------------
    async def _run_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "retrieve":
            return await self._tool_retrieve(args)
        if name == "repo_read":
            return self._tool_repo_read(args)
        if name == "web_search":
            return await self._tool_web_search(args)
        if name == "web_fetch":
            return await self._tool_web_fetch(args)
        if name == "run_tool":
            return await self._tool_run_tool(args)
        return {"error": f"unknown tool '{name}'"}

    async def _tool_retrieve(self, args: dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "retrieve needs a 'query'"}
        try:
            if self.embedder is not None:
                vec = self.embedder.embed([query])[0]
            else:
                vec = (await self.gw.embed([query]))[0]
            hits = self.store.search(embedding=vec, query_text=query, k=6)
        except Exception as e:
            return {"error": f"retrieve failed: {type(e).__name__}: {e}"}
        return {
            "results": [
                {
                    "source": h.source,
                    "score": round(float(h.score), 4),
                    "excerpt": h.text[:600],
                }
                for h in hits
            ]
        }

    def _tool_repo_read(self, args: dict[str, Any]) -> Any:
        rel = str(args.get("path", "")).strip().lstrip("/")
        if not rel:
            return {"error": "repo_read needs a 'path'"}
        base = os.path.abspath(self.project_path)
        target = os.path.abspath(os.path.join(base, rel))
        if not target.startswith(base + os.sep) and target != base:
            return {"error": f"path escapes project: {rel}"}
        if not os.path.isfile(target):
            return {"error": f"no such file: {rel}"}
        try:
            with open(target, encoding="utf-8") as f:
                text = f.read(20000)
            return {"path": rel, "chars": len(text), "content": text}
        except OSError as e:
            return {"error": f"read failed: {e}"}

    async def _tool_web_search(self, args: dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "web_search needs a 'query'"}
        sc = getattr(self.cfg, "search", None)
        if sc is not None and getattr(sc, "endpoint", ""):
            try:
                return await self._searxng_search(sc, query)
            except Exception as e:
                # SearXNG down (bundled instance stopped, restarting, etc.):
                # degrade to the DDG scrape rather than failing the turn.
                return await self._ddg_search(query, f"searxng unavailable: {type(e).__name__}")
        return await self._ddg_search(query, None)

    async def _searxng_search(self, sc: Any, query: str) -> Any:
        """Search through the bundled SearXNG JSON API (project module)."""
        headers = {}
        if getattr(sc, "key", None):
            headers["Authorization"] = f"Bearer {sc.key}"
        params = {
            "q": query,
            "format": "json",
            "language": getattr(sc, "language", "en") or "en",
        }
        client = self._get_http()
        resp = await client.get(
            f"{sc.endpoint.rstrip('/')}/search",
            params=params,
            headers=headers or None,
            timeout=getattr(sc, "timeout_s", 10.0) or 10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        out = []
        for r in results[:5]:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "") or r.get("snippet", "")
            if not title and not url:
                continue
            out.append({"title": title, "url": url, "snippet": snippet})
        if not out:
            return {"results": [], "note": "searxng: no results"}
        return {"results": out, "engine": "searxng"}

    async def _ddg_search(self, query: str, degraded_note: str | None) -> Any:
        """Legacy fallback: DuckDuckGo HTML scrape (rate-limit prone)."""
        url = "https://html.duckduckgo.com/html/?q=" + _urlquote(query)
        try:
            client = self._get_http()
            resp = await client.get(url, timeout=self.cfg.web_timeout_s)
            resp.raise_for_status()
        except Exception as e:
            return {"error": f"web_search unavailable: {type(e).__name__}", "degraded": True}
        links = _ddg_links(resp.text)
        if not links:
            return {"results": [], "note": "no results (possibly offline)"}
        out = {"results": links[:5], "engine": "duckduckgo"}
        if degraded_note:
            out["degraded"] = True
            out["note"] = degraded_note
        return out

    async def _tool_web_fetch(self, args: dict[str, Any]) -> Any:
        url = str(args.get("url", "")).strip()
        if not url or not url.startswith(("http://", "https://")):
            return {"error": "web_fetch needs an http(s) 'url'"}
        try:
            client = self._get_http()
            resp = await client.get(url, timeout=self.cfg.web_timeout_s)
            resp.raise_for_status()
        except Exception as e:
            return {"error": f"web_fetch unavailable: {type(e).__name__}", "degraded": True}
        return {"url": url, "status": resp.status_code, "text": _html_to_text(resp.text)[:8000]}

    async def _tool_run_tool(self, args: dict[str, Any]) -> Any:
        if self.tool_registry is None:
            return {"error": "no tools configured"}
        name = str(args.get("name", "")).strip()
        if not name:
            return {"error": "run_tool needs a 'name'"}
        try:
            result = await self.tool_registry.run(name, args.get("args", {}))
            return result if isinstance(result, (dict, list)) else {"result": result}
        except Exception as e:
            return {"error": f"tool '{name}' failed: {type(e).__name__}: {e}"}

    def _get_http(self):
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(
                headers={"User-Agent": "dmd-agent/2.0 (+dm copilot)"}, follow_redirects=True
            )
        return self._http


# ---------------------------------------------------------------------------
# web helpers
# ---------------------------------------------------------------------------


def _items_from_table(md: str) -> list[dict[str, Any]]:
    """Best-effort: extract structured items from a markdown table.

    The worker schema asks the model for ``items`` with dc_find, but ling-tiny
    often writes the skill-check table only into body_md. When items are
    absent, derive them from body rows like
    ``| Hidden pouch | Perception <n> | 1 | 150 gp | ... |`` where <n> is the
    DC written by the model. Returns [] when the body has no parseable table.
    """
    out: list[dict[str, Any]] = []
    rows: list[list[str]] = []
    for raw in md.splitlines():
        line = raw.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        # skip the separator row
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return []
    header = [h.lower() for h in rows[0]]
    for cells in rows[1:]:
        row: dict[str, str] = {}
        for i, h in enumerate(header):
            if i < len(cells) and cells[i]:
                row[h] = cells[i]
        name = row.get("find") or row.get("name") or row.get("item")
        if not name:
            continue
        skill_cell = row.get("skill / dc") or row.get("skill/dc") or row.get("dc") or ""
        m = re.search(r"(\d+)", skill_cell)
        item: dict[str, Any] = {
            "name": name[:200],
            "quantity": _int_or(row.get("qty") or row.get("quantity"), 1),
            "notes": row.get("notes", "")[:300],
        }
        if m:
            item["dc_find"] = int(m.group(1))
        else:
            item["dc_find"] = None
        out.append(item)
        if len(out) >= 40:
            break
    return out


def _int_or(s: str | None, default: int) -> int:
    if not s:
        return default
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def _urlquote(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="")


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def _html_to_text(html: str) -> str:
    """Strip tags to readable text (no external deps)."""
    html = re.sub(r"(?is)<(script|style|nav|header|footer).*?</\1>", " ", html)
    text = _TAG_RE.sub(" ", html)
    text = _html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = _WS_RE.sub("\n\n", text)
    return text.strip()


def _ddg_links(page_html: str) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML result links (title + url) without a search API."""
    out: list[dict[str, str]] = []
    # DDG html endpoint wraps results in <a class="result__a" href="...">Title</a>
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page_html, re.DOTALL):
        href = m.group(1)
        title = _html_to_text(m.group(2))[:200]
        # DDG wraps real url in a redirect: //duckduckgo.com/l/?uddg=<encoded>
        if "uddg=" in href:
            from urllib.parse import parse_qs, unquote, urlparse

            qs = parse_qs(urlparse("https:" + href if href.startswith("//") else href).query)
            if qs.get("uddg"):
                href = unquote(qs["uddg"][0])
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("http"):
            out.append({"title": title, "url": href})
    return out
