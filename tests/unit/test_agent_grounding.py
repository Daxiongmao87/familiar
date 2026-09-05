"""Mandatory-grounding for rules/ruling cards.

Owner directive 2026-09-05: worker agents must actually search before
producing a rules/ruling card (bundled SearXNG). ling-tiny answers from
memory and is too slow to survive a push-back round-trip, so rules tasks
pre-ground deterministically: one web_search fires BEFORE the model's
first decode and its results are handed to the model as context.
"""

from __future__ import annotations

from typing import Any

import pytest

from dmd.agent import WorkerAgent, _is_rules_task
from dmd.config import AgentConfig, SearchConfig


class _FakeGw:
    """Gateway stub: web_search tool returns grounded results; the model then
    produces a rules card citing them."""

    def __init__(self) -> None:
        self.web_search_calls = 0
        self.final_body = ""

    async def chat(self, role, messages, **kw) -> Any:
        user_text = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_text = m["content"]
                break
        # After the grounding block is injected, the model writes the card.
        if "RULE SOURCE SEARCH RESULTS" in user_text:
            self.final_body = (
                "Grapple per SRD (source: https://example.com/grapple) — "
                "Athletics vs Athletics."
            )
            return (
                '{"kind": "rules", "title": "Ruling: Grapple", '
                f'"body_md": "{self.final_body}", "items": []}}'
            )
        # Any other (non-rules) task: the model answers from repo context.
        return (
            '{"kind": "loot", "title": "Loot", '
            '"body_md": "A pouch (dc_find 12)", "items": []}'
        )

    async def aclose(self) -> None:  # pragma: no cover
        pass


class _RecordingWorker(WorkerAgent):
    """WorkerAgent with a stubbed web_search that records the call."""

    def __init__(self, gw: Any, agent: Any) -> None:
        super().__init__(
            gw=gw,
            store=None,
            project_path="sample-campaign",
            cfg=agent,
            world_map="sample",
            embedder=None,
        )
        self._recorded = 0

    async def _tool_web_search(self, args: dict) -> Any:
        self._recorded += 1
        return {
            "results": [
                {
                    "title": "5e Grapple Rules",
                    "url": "https://example.com/grapple",
                    "snippet": "Athletics vs Athletics; speed 0.",
                }
            ],
            "engine": "searxng",
        }


def _mk_agent(gw: Any) -> tuple[_RecordingWorker, AgentConfig]:
    cfg = AgentConfig(search=SearchConfig(endpoint="http://127.0.0.1:8888"))
    return _RecordingWorker(gw, cfg), cfg


def test_is_rules_task_detects_ruling() -> None:
    assert _is_rules_task("Produce a RULING card: the DC, the skill")
    assert _is_rules_task("The DM triggered a rules intent")
    assert not _is_rules_task("Produce a LOOT card: quantities and values")
    assert not _is_rules_task("what do we find")


@pytest.mark.asyncio
async def test_rules_task_pre_grounds_with_web_search() -> None:
    gw = _FakeGw()
    agent, _ = _mk_agent(gw)

    res = await agent.run(
        "The DM triggered a rules intent. Produce a RULING card: the DC, the skill, and the ruling.",
        tier="card",
    )
    # The deterministic pre-grounding fired exactly one web_search.
    assert agent._recorded == 1
    # tool_calls counts the grounding call.
    assert res.tool_calls == 1
    assert res.card is not None
    assert res.card["kind"] == "rules"
    assert "example.com" in res.card["body_md"]


@pytest.mark.asyncio
async def test_loot_task_does_not_force_web_search() -> None:
    """Loot cards are repo-grounded; the mandatory-search rule must not
    force a web search on them (owner: loot comes from campaign lore)."""
    gw = _FakeGw()
    agent, _ = _mk_agent(gw)

    res = await agent.run("loot intent: what is on Brother Ulrich?", tier="card")
    assert agent._recorded == 0
    assert res.card is not None
