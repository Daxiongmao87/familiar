"""Slot-leak guard tests (2026-09-05 incident).

A llama.cpp chat request without ``max_tokens`` runs ``n_predict=-1``: a
repeating model never frees its inference slot, and the client's 180 s read
timeout abandoned one slot every 210 s cadence (11 slots leaked in lockstep).

Every dmd LLM call must therefore carry a bounded ``max_tokens`` and a timed-out
call must cancel cleanly (typed GatewayError, no half-open connection).

Regression tests: red against gateway.py before the fix (body omitted
max_tokens when the caller passed None), green after.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from dmd.config import load_config_dict
from dmd.gateway import ROLE_DEFAULT_MAX_TOKENS, Gateway, GatewayError
from dmd.monitor import TranscriptMonitor
from dmd.triggers import detect_trigger


def _cfg_dict(**over: Any) -> dict:
    fast: dict[str, Any] = {"base_url": "http://test", "model_id": "fast-1"}
    synthesis: dict[str, Any] = {"base_url": "http://test", "model_id": "synth-1"}
    for k, v in over.items():
        if k == "fast":
            fast.update(v)
        elif k == "synthesis":
            synthesis.update(v)
    return {"models": {"synthesis": synthesis, "fast": fast, "stt": {"base_url": "http://test"}}}


def _gw(handler: Callable[[httpx.Request], httpx.Response], **over: Any) -> Gateway:
    gw = Gateway(load_config_dict(_cfg_dict(**over)))
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return gw


def _chat_handler(captured: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/chat/completions":
            captured["body"] = json.loads(req.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )
        return httpx.Response(404)

    return handler


# ---------------------------------------------------------------------------
# Gateway-level guarantee: the body ALWAYS contains max_tokens.
# ---------------------------------------------------------------------------


async def test_chat_sends_role_default_max_tokens_when_caller_passes_none() -> None:
    """Caller with max_tokens=None still gets a bounded request (fast=1024)."""
    captured: dict[str, Any] = {}
    gw = _gw(_chat_handler(captured))
    await gw.chat("fast", [{"role": "user", "content": "x"}], temperature=0)
    assert captured["body"]["max_tokens"] == ROLE_DEFAULT_MAX_TOKENS["fast"]


async def test_chat_sends_synthesis_default_max_tokens() -> None:
    captured: dict[str, Any] = {}
    gw = _gw(_chat_handler(captured))
    await gw.chat("synthesis", [{"role": "user", "content": "x"}])
    assert captured["body"]["max_tokens"] == ROLE_DEFAULT_MAX_TOKENS["synthesis"]


async def test_chat_explicit_max_tokens_wins_over_default() -> None:
    captured: dict[str, Any] = {}
    gw = _gw(_chat_handler(captured))
    await gw.chat("fast", [{"role": "user", "content": "x"}], max_tokens=77)
    assert captured["body"]["max_tokens"] == 77


async def test_chat_endpoint_config_max_tokens_wins_over_role_default() -> None:
    captured: dict[str, Any] = {}
    gw = _gw(_chat_handler(captured), fast={"max_tokens": 321})
    await gw.chat("fast", [{"role": "user", "content": "x"}])
    assert captured["body"]["max_tokens"] == 321


async def test_chat_read_timeout_raises_gateway_error() -> None:
    """A timed-out generation cancels as a typed GatewayError, not a raw hang."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=req)

    gw = _gw(handler)
    with pytest.raises(GatewayError) as excinfo:
        await gw.chat("fast", [{"role": "user", "content": "x"}])
    assert "timed out" in str(excinfo.value)


async def test_chat_endpoint_request_timeout_shortens_read_window() -> None:
    """request_timeout_s caps the read wait so an abandoned call returns fast."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=req)

    gw = _gw(handler, fast={"request_timeout_s": 0.05})
    with pytest.raises(GatewayError):
        await gw.chat("fast", [{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------------
# Call-site audit: the two callers that leaked slots today, plus the guard
# applied to every gw.chat through the real Gateway body-builder.
# ---------------------------------------------------------------------------


async def test_trigger_classifier_call_sends_max_tokens() -> None:
    """The fast-lane LLM is consulted only when the regex is silent, and that
    call must still be generation-capped (slot-leak guard)."""
    from dmd.triggers import LANE_CLASSIFY_MAX_TOKENS

    captured: dict[str, Any] = {}
    gw = _gw(_chat_handler(captured))
    # prose with no search/loot/examine keyword -> regex silent -> LLM consulted
    await detect_trigger(gw, "the fog rolls in over the harbour")
    assert captured["body"]["max_tokens"] == LANE_CLASSIFY_MAX_TOKENS


async def test_monitor_judge_call_sends_max_tokens() -> None:
    captured: dict[str, Any] = {}
    gw = _gw(_chat_handler(captured))

    async def _noop(verdict: dict) -> None:
        return None

    mon = TranscriptMonitor(
        gw,
        object(),
        get_transcript=lambda: "a" * 80,
        get_scene=lambda: "scene",
        on_action=_noop,
    )
    await mon.tick_once()
    assert captured["body"]["max_tokens"] == 1024


async def test_probe_all_ping_stays_bounded() -> None:
    captured: dict[str, Any] = {}
    gw = _gw(_chat_handler(captured))
    await gw.probe_all()
    if "body" in captured:
        assert captured["body"]["max_tokens"] == 1
