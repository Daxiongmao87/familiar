"""Tests for dmd.providers: local/remote routing beneath chat()/score().

Covers the four required combinations (LLM local/remote x JEV
local/remote), legacy Gateway behavior without a router, and the
guarantee that remote URLs survive a round-trip to local and back.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from dmd.config import load_config_dict
from dmd.gateway import Gateway, GatewayError
from dmd.openjev import OpenjevGate
from dmd.providers import (
    LOCAL_SYNTHESIS_MODEL,
    EndpointRouter,
    jev_base_url_for,
    synthesis_endpoint_for,
)

REMOTE_SYNTH = "http://remote-llm:8080/v1"
REMOTE_JEV = "http://remote-jev:8199"
BRIDGE = "http://127.0.0.1:8791"
JEV_LOCAL = "http://127.0.0.1:8299"


def _cfg(synth_provider: str = "remote", jev_provider: str = "remote") -> Any:
    return load_config_dict(
        {
            "models": {
                "synthesis": {
                    "provider": synth_provider,
                    "base_url": REMOTE_SYNTH,
                    "model_id": "minicpm5-2b",
                    "api_key": "sk-remote",
                    "extra_body": {"enable_thinking": False},
                    "max_tokens": 512,
                },
                "stt": {},
            },
            "openjev": {
                "enabled": True,
                "provider": jev_provider,
                "base_url": REMOTE_JEV,
            },
            "desktop": {
                "enabled": True,
                "bridge_url": BRIDGE,
                "jev_local_url": JEV_LOCAL,
            },
        }
    )


def test_defaults_are_remote_and_byte_identical() -> None:
    """A config without provider flags routes exactly as before."""
    cfg = load_config_dict(
        {
            "models": {
                "synthesis": {"base_url": REMOTE_SYNTH, "model_id": "m"},
                "stt": {},
            },
        }
    )
    assert synthesis_endpoint_for(cfg).base_url == REMOTE_SYNTH
    assert synthesis_endpoint_for(cfg).model_id == "m"
    assert jev_base_url_for(cfg) == "http://127.0.0.1:8199"


@pytest.mark.parametrize(
    "synth,jev",
    [("remote", "remote"), ("remote", "local"), ("local", "remote"), ("local", "local")],
)
def test_four_provider_combinations(synth: str, jev: str) -> None:
    """All four LLM x JEV combinations resolve independently."""
    cfg = _cfg(synth, jev)
    ep = synthesis_endpoint_for(cfg)
    assert ep.base_url == (BRIDGE if synth == "local" else REMOTE_SYNTH)
    assert jev_base_url_for(cfg) == (JEV_LOCAL if jev == "local" else REMOTE_JEV)


def test_local_synthesis_preserves_call_shape() -> None:
    """Local mode swaps URL/model only; body-affecting fields survive."""
    cfg = _cfg("local", "remote")
    ep = synthesis_endpoint_for(cfg)
    assert ep.model_id == LOCAL_SYNTHESIS_MODEL
    assert ep.extra_body == {"enable_thinking": False}
    assert ep.max_tokens == 512
    assert ep.api_key is None  # localhost bridge takes no key


def test_local_without_bridge_url_fails_loudly() -> None:
    """An empty bridge URL raises instead of dialing a wrong host."""
    cfg = _cfg("local", "remote")
    cfg.desktop.bridge_url = ""
    with pytest.raises(ValueError, match="bridge_url"):
        synthesis_endpoint_for(cfg)
    cfg.desktop.jev_local_url = ""
    cfg.openjev.provider = "local"
    with pytest.raises(ValueError, match="jev_local_url"):
        jev_base_url_for(cfg)


def test_remote_urls_survive_local_round_trip() -> None:
    """Switching providers never rewrites the stored remote config."""
    cfg = _cfg("remote", "remote")
    cfg.models.synthesis.provider = "local"
    cfg.openjev.provider = "local"
    assert synthesis_endpoint_for(cfg).base_url == BRIDGE
    cfg.models.synthesis.provider = "remote"
    cfg.openjev.provider = "remote"
    assert synthesis_endpoint_for(cfg).base_url == REMOTE_SYNTH
    assert cfg.models.synthesis.api_key == "sk-remote"
    assert jev_base_url_for(cfg) == REMOTE_JEV


@pytest.mark.asyncio
async def test_gateway_router_chats_against_bridge() -> None:
    """Gateway with a router POSTs chat/completions to the local bridge."""
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        import json as _json

        seen["body"] = _json.loads(req.content.decode())
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "{\"a\": 1}"}}]}
        )

    cfg = _cfg("local", "remote")
    gw = Gateway(cfg, router=EndpointRouter(cfg))
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await gw.chat(
        "synthesis",
        [{"role": "user", "content": "hi"}],
        json_schema={"type": "object"},
    )
    assert out == {"a": 1}
    assert seen["url"] == BRIDGE + "/chat/completions"
    assert seen["body"]["model"] == LOCAL_SYNTHESIS_MODEL
    assert seen["body"]["max_tokens"] == 512  # slot-leak guard intact
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_gateway_without_router_ignores_provider_flag() -> None:
    """router=None preserves legacy behavior exactly (config URL always)."""
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    cfg = _cfg("local", "remote")
    gw = Gateway(cfg)  # no router: legacy path
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await gw.chat("synthesis", [{"role": "user", "content": "hi"}])
    assert seen["url"] == REMOTE_SYNTH + "/chat/completions"


@pytest.mark.asyncio
async def test_gateway_router_error_surfaces_as_gateway_error() -> None:
    """Router ValueErrors become typed GatewayErrors, never raw."""
    cfg = _cfg("local", "remote")
    cfg.desktop.bridge_url = ""
    gw = Gateway(cfg, router=EndpointRouter(cfg))
    with pytest.raises(GatewayError, match="bridge_url"):
        await gw.chat("synthesis", [{"role": "user", "content": "hi"}])


def test_jev_gate_repoint_keeps_scoring_shape() -> None:
    """set_base_url only moves the URL; options/threshold untouched."""
    gate = OpenjevGate(REMOTE_JEV, threshold=0.7, recent_n=5)
    gate.set_base_url(JEV_LOCAL + "/")
    assert gate._base_url == JEV_LOCAL
    assert gate._threshold == 0.7
    assert gate._recent_n == 5


def test_router_describe_has_no_secrets() -> None:
    """describe() exposes routing state with keys reduced to booleans."""
    cfg = _cfg("local", "local")
    state = EndpointRouter(cfg).describe()
    assert state["synthesis_effective_base_url"] == BRIDGE
    assert state["jev_effective_base_url"] == JEV_LOCAL
    assert state["synthesis"]["has_api_key"] is True
    blob = repr(state)
    assert "sk-remote" not in blob


def test_engine_apply_providers_repoints_gate() -> None:
    """SessionEngine.apply_providers moves the live gate without rebuild."""
    from dmd.pipeline import SessionEngine

    cfg = _cfg("remote", "remote")

    class _Pool:
        async def submit(self, job, work) -> None:
            return None

    engine = SessionEngine(
        cfg=cfg,
        store=None,
        gw=None,
        entries=[],
        embedder=None,
        pool=_Pool(),
        on_event=lambda e: None,
        project_path="/tmp/fam-test",
    )
    assert engine._openjev_gate is not None
    assert engine._openjev_gate._base_url == REMOTE_JEV
    cfg.models.synthesis.provider = "local"
    cfg.openjev.provider = "local"
    eff = engine.apply_providers()
    assert eff["synthesis"] == BRIDGE
    assert eff["jev"] == JEV_LOCAL
    assert engine._openjev_gate._base_url == JEV_LOCAL
