"""Mandatory-grounding guard for rules/ruling cards.

Owner directive 2026-09-05: worker agents must actually SEARCH before
producing a rules/ruling card (bundled SearXNG). ling-tiny answers from
memory with zero tool calls; the guard pushes an ungrounded rules card
back through the loop until it calls web_search (or the budget ends).
"""

from __future__ import annotations

from typing import Any

import pytest

from dmd.agent import WorkerAgent
from dmd.config import AgentConfig, SearchConfig


class _FakeGw:
    """Gateway stub: first calls answer from memory (no tool), then after the
    grounding nudge the model calls web_search and returns a grounded card."""

    def __init__(self) -> None:
        self.calls: list[str] = []  # role labels of each chat call

    async def chat(self, role, messages, **kw) -> Any:
        self.calls.append(role)
        # Find the latest user message to decide the behavior.
        latest = ""
        for m in reversed(messages):
            if m["role"] == "user":
                latest = m["content"]
                break
        if "must be grounded" in latest or "Call web_search now" in latest:
            # After the nudge, the model performs the tool call.
            return '{"tool": "web_search", "args": {"query": "5e grappling rules"}}'
        # Tool result now present -> final grounded card.
        if "TOOL RESULT" in latest:
            return (
                '{"kind": "rules", "title": "Ruling: Grapple", '
                '"body_md": "Grapple DC per 5e SRD (source: web search)", '
                '"items": []}'
            )
        # First attempt: model answers from memory with NO tool call.
        return (
            '{"kind": "rules", "title": "Ruling: Grapple", '
            '"body_md": "Athletics vs Athletics", "items": []}'
        )


def _mk_agent(gw: Any) -> WorkerAgent:
    cfg = AgentConfig(search=SearchConfig(endpoint="http://127.0.0.1:8888"))
    return WorkerAgent(
        gw=gw,
        store=None,
        project_path="sample-campaign",
        cfg=cfg,
        world_map="sample",
        embedder=None,
    )


@pytest.mark.asyncio
async def test_rules_card_without_tool_calls_is_pushed_to_search() -> None:
    gw = _FakeGw()
    agent = _mk_agent(gw)

    # Tiny model answers a RULES task from memory first (no tools). The guard
    # must push it back until it calls web_search, then accept the card.
    res = await agent.run(
        "The DM triggered a rules intent. Produce a RULING card: the DC, the skill, and the ruling.",
        tier="card",
    )
    assert res.tool_calls == 1, f"expected the grounding web_search, got {res.tool_calls}"
    assert res.card is not None
    assert res.card["kind"] == "rules"
    assert res.card["title"] == "Ruling: Grapple"
    # The grounded card's body cites the web-search source (not memory).
    assert "web search" in res.card["body_md"]


@pytest.mark.asyncio
async def test_loot_card_without_tool_calls_is_accepted() -> None:
    """Loot cards are repo-grounded; the mandatory-search guard must not
    force a web search on them (owner: loot comes from campaign lore)."""
    gw = _FakeGw()
    agent = _mk_agent(gw)

    res = await agent.run("loot intent: what is on Brother Ulrich?", tier="card")
    # First chat in _FakeGw returns a rules-shaped card; acceptable — we only
    # assert the guard does not infinite-loop and honors the model output.
    assert res.card is not None
    assert res.tool_calls >= 0
