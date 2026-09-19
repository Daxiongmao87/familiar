"""Evidence-judged deterministic worker: fixed retrieval pipeline, no agent loop.

Per trigger: collect search terms with zero LLM calls (statistical
phrases fused with lexicon spans), rank them in one JEV pass, search
BOTH legs per term (campaign RAG + web), then a JEV relevance verdict
compares the retrieved evidence (offline / online / both) before one
synthesis call writes the card. No presumed locations anywhere: which
leg holds the answer is judged after the fact, every trigger.

Recall-biased throughout: term ranking fails open to all collected
terms, relevance fails over to both legs, and the synthesis role serves
every tier (the fast role is not trusted here).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Sequence

from .agent import _CARD_SCHEMA, _extract_json, AgentResult, WorkerAgent
from .openjev import TERM_RANK_QUESTION, OpenjevError, OpenjevGate
from .terms import collect_terms, mass_cutoff
from .types import LexiconEntry

logger = logging.getLogger(__name__)

_SYNTH_SYSTEM = (
    "You are a DM copilot. Write the card FROM the evidence below. Do not "
    "invent rules, DCs, names, or facts; when the evidence is thin, say "
    "what is known and mark the rest unknown."
)
_NOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "body_md": {"type": "string"},
    },
    "required": ["title", "body_md"],
    "additionalProperties": False,
}
_MAX_STATE_CHARS = 6000
_MAX_TERMS = 4


def _hits_text(results: Any, per_hit: int = 300, limit: int = 3) -> str:
    """Render retrieval hits compactly for prompts and JEV state."""
    if not isinstance(results, dict):
        return "(lookup failed)"
    hits = results.get("results") or []
    if not hits:
        return results.get("note") or results.get("error") or "(no results)"
    lines = []
    for h in hits[:limit]:
        if not isinstance(h, dict):
            continue
        src = h.get("source") or h.get("url") or h.get("title") or "?"
        score = h.get("score")
        tag = f" [score {score}]" if score is not None else ""
        txt = (h.get("excerpt") or h.get("snippet") or h.get("content") or "").strip()
        lines.append(f"- {src}{tag}: {txt[:per_hit]}")
    return "\n".join(lines) if lines else "(no usable hits)"


class JevWorker(WorkerAgent):
    """Deterministic pipeline: terms, both legs, relevance, synthesize."""

    def __init__(
        self,
        *args: Any,
        gate: OpenjevGate,
        entries: Sequence[LexiconEntry] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gate = gate
        self.entries = list(entries)

    async def run(
        self,
        task: str,
        tier: str,
        trigger_portion: str = "",
        transcript: str = "",
        staged_block: str = "",
    ) -> AgentResult:
        """Run the fixed pipeline: route, retrieve, judge, synthesize."""
        try:
            return await asyncio.wait_for(
                self._directed(task, tier, trigger_portion, transcript, staged_block),
                self.cfg.agent_timeout_s,
            )
        except asyncio.TimeoutError:
            return AgentResult(tier=tier, error="agent budget exhausted (timeout)")

    async def _directed(
        self,
        task: str,
        tier: str,
        trigger_portion: str,
        transcript: str,
        staged_block: str,
    ) -> AgentResult:
        role = "synthesis"  # every tier: the fast role is not trusted here
        need = (trigger_portion or task).strip()
        tool_calls = 0

        window = [ln for ln in transcript.splitlines()[-8:] if ln.strip()]
        terms = await self._collect_ranked_terms(window or ([need] if need else []))
        if not terms and need:
            terms = [need[:200]]
        logger.info("jevworker terms=%s task=%.60s", terms, task)

        evidence: dict[str, list[str]] = {"offline": [], "online": []}
        for term in terms:
            res = await self._tool_retrieve({"query": term})
            tool_calls += 1
            evidence["offline"].append(f"Q: {term}\n{_hits_text(res)}")
            res = await self._tool_web_search({"query": term})
            tool_calls += 1
            evidence["online"].append(f"Q: {term}\n{_hits_text(res)}")

        verdict = await self.gate.relevance(self._rel_state(need, evidence))
        logger.info("jevworker relevance=%s task=%.60s", verdict.route, task)
        keep = {"offline": ("offline",), "online": ("online",)}.get(
            verdict.route, ("offline", "online")
        )
        filtered = {b: (evidence[b] if b in keep else []) for b in evidence}

        return await self._synthesize(
            role, tier, task, transcript, staged_block, filtered, tool_calls
        )

    async def _collect_ranked_terms(self, lines: list[str]) -> list[str]:
        """Zero-LLM candidates, one JEV ranking pass, mass cutoff.

        Fails open to all collected terms when ranking errors: retrieval
        still runs and relevance still judges.
        """
        cands = collect_terms(lines, self.entries)
        if len(cands) < 2:
            return cands
        state = "\n".join(lines)
        try:
            probs = await self.gate.rank(
                f"t{time.monotonic_ns()}",
                state,
                TERM_RANK_QUESTION,
                [{"id": c, "description": c} for c in cands],
            )
        except OpenjevError as exc:
            logger.warning("jevworker term ranking failed open: %s", exc)
            return cands[:_MAX_TERMS]
        ordered = [cands[i] for i in mass_cutoff([probs[c] for c in cands])]
        return ordered[:_MAX_TERMS]

    def _rel_state(self, need: str, evidence: dict[str, list[str]]) -> str:
        parts = [f"NEED: {need[:400]}"]
        for branch in ("offline", "online"):
            body = "\n".join(evidence[branch]) or "(no results)"
            parts.append(f"RETRIEVED ({branch}):\n{body}")
        return "\n".join(parts)[:_MAX_STATE_CHARS]

    async def _synthesize(
        self,
        role: str,
        tier: str,
        task: str,
        transcript: str,
        staged_block: str,
        evidence: dict[str, list[str]],
        tool_calls: int,
    ) -> AgentResult:
        blocks = []
        for branch in ("offline", "online"):
            if evidence[branch]:
                blocks.append(f"EVIDENCE ({branch}):\n" + "\n".join(evidence[branch]))
        if staged_block:
            blocks.append(f"PRE-STAGED CONTEXT:\n{staged_block}")
        user = (
            f"{task}\n\nTranscript:\n{transcript[-2000:]}\n\n" + "\n\n".join(blocks)
        )
        messages = [
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": user},
        ]
        if tier == "card":
            for _ in range(2):
                try:
                    content = await self.gw.chat(
                        role, messages, json_schema=_CARD_SCHEMA, temperature=0.2
                    )
                except Exception as exc:
                    return AgentResult(tier=tier, tool_calls=tool_calls, error=f"synthesis failed: {exc}")
                parsed = _extract_json(content)
                if isinstance(parsed, dict) and (parsed.get("body_md") or parsed.get("title")):
                    return AgentResult(
                        tier="card", card=self._normalize_card(parsed), tool_calls=tool_calls
                    )
                messages.append({"role": "assistant", "content": str(content)[:2000]})
                messages.append(
                    {
                        "role": "user",
                        "content": 'Respond with the final CARD JSON only: {"kind","title","body_md","player_ids","items"}.',
                    }
                )
            return AgentResult(tier=tier, tool_calls=tool_calls, error="synthesis unparseable")
        try:
            content = await self.gw.chat(
                role, messages, json_schema=_NOTE_SCHEMA, temperature=0.2
            )
        except Exception as exc:
            return AgentResult(tier=tier, tool_calls=tool_calls, error=f"synthesis failed: {exc}")
        parsed = _extract_json(content)
        if isinstance(parsed, dict) and (parsed.get("body_md") or parsed.get("title")):
            title = str(parsed.get("title") or "").strip()
            subtitle = str(parsed.get("subtitle") or "").strip()
            body = str(parsed.get("body_md") or "").strip()
            text = "\n".join(
                p for p in (f"**{title}**" if title else "", subtitle, body) if p
            ).strip()
        else:
            text = str(content).strip()[:800]
        if not text:
            return AgentResult(tier=tier, tool_calls=tool_calls, error="no text produced")
        return AgentResult(tier=tier, text=text, tool_calls=tool_calls)
