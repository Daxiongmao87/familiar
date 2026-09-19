"""Tests for dmd.gateway — MockTransport-driven, no real network.

NOTE: Gateway.__init__ never initializes self._client. The assignment
`self._client = httpx.AsyncClient(...)` lives inside the @property cfg body
AFTER `return self._cfg` — unreachable dead code. The fixture below works
around this by setting gw._client manually after construction.

This file's tests pass; the init bug is documented and reported.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from dmd.config import load_config_dict
from dmd.gateway import Gateway, GatewayError


def _cfg(*, with_api_key: bool = False, with_extra_body: bool = False,
         synthesis_model: str = "synth-1") -> Any:
    return load_config_dict({
        "models": {
            "synthesis": {
                "base_url": "http://test",
                "model_id": synthesis_model,
                "api_key": "sk-test" if with_api_key else None,
                "extra_body": {"unique_param": "xyz"} if with_extra_body else {},
            },
            "stt": {},
        }
    })


def _build(handler: Callable[[httpx.Request], httpx.Response], *,
           with_api_key: bool = False, with_extra_body: bool = False,
           synthesis_model: str = "synth-1") -> Gateway:
    """Build Gateway + manually attach MockTransport-backed client.

    Workaround for the __init__ bug: see module docstring.
    """
    cfg = _cfg(with_api_key=with_api_key, with_extra_body=with_extra_body,
               synthesis_model=synthesis_model)
    gw = Gateway(cfg)
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return gw


@pytest.mark.asyncio
async def test_models_listing_in_probe_all_ok_and_missing_model() -> None:
    """probe_all sees /models listing and reports ok / missing_model correctly."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "synth-1"}]})
        return httpx.Response(404)

    gw = _build(handler)
    results = await gw.probe_all()
    assert results["synthesis"] == "ok"
    assert results["synthesis.model"] == "ok"

    def handler2(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "different-model"}]})
        return httpx.Response(404)

    gw2 = _build(handler2)
    results2 = await gw2.probe_all()
    assert results2["synthesis"] == "ok"
    assert results2["synthesis.model"] == "missing_model"


@pytest.mark.asyncio
async def test_chat_happy_str_content() -> None:
    """chat returns a string when content is a plain (non-JSON) string."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/chat/completions":
            captured["body"] = json.loads(req.content)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello world"}}]},
            )
        return httpx.Response(404)

    gw = _build(handler)
    out = await gw.chat("synthesis", [{"role": "user", "content": "hi"}])
    assert out == "hello world"
    assert captured["body"]["model"] == "synth-1"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_chat_with_response_format_json_schema_parses_dict() -> None:
    """With json_schema, response_format is set on body; JSON-string content parses to dict."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/chat/completions":
            captured["body"] = json.loads(req.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": json.dumps({"entities": [{"canonical": "Foo"}]})}}
                    ]
                },
            )
        return httpx.Response(404)

    gw = _build(handler)
    out = await gw.chat(
        "synthesis",
        [{"role": "user", "content": "hi"}],
        json_schema={"type": "object"},
    )
    assert isinstance(out, dict)
    assert out == {"entities": [{"canonical": "Foo"}]}
    rf = captured["body"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == {"type": "object"}
    assert rf["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_chat_malformed_json_content_falls_back_to_raw_string() -> None:
    """Content that is not valid JSON returns as the raw string (not parsed)."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/chat/completions":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "this is not json {"}}]},
            )
        return httpx.Response(404)

    gw = _build(handler)
    out = await gw.chat("synthesis", [{"role": "user", "content": "hi"}])
    assert out == "this is not json {"
    assert isinstance(out, str)


@pytest.mark.asyncio
async def test_chat_http_500_raises_gateway_error_with_status() -> None:
    """HTTP 500 -> GatewayError with status code (and role) in the message."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server boom")

    gw = _build(handler)
    with pytest.raises(GatewayError) as excinfo:
        await gw.chat("synthesis", [{"role": "user", "content": "hi"}])
    msg = str(excinfo.value)
    assert "500" in msg
    assert "synthesis" in msg


@pytest.mark.asyncio
async def test_bearer_header_present_iff_api_key_set() -> None:
    """Authorization: Bearer X header is present iff api_key is configured."""
    captured: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/chat/completions":
            captured.append({k.lower(): v for k, v in req.headers.items()})
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )
        return httpx.Response(404)

    gw_no_key = _build(handler, with_api_key=False)
    await gw_no_key.chat("synthesis", [{"role": "user", "content": "x"}])
    assert "authorization" not in captured[0]

    captured.clear()
    gw_with_key = _build(handler, with_api_key=True)
    await gw_with_key.chat("synthesis", [{"role": "user", "content": "x"}])
    assert captured[0].get("authorization") == "Bearer sk-test"


@pytest.mark.asyncio
async def test_probe_all_extra_body_ok_and_rejected() -> None:
    """probe_all reports extra_body_ok and extra_body_rejected:<status>."""

    # Case A: extra_body_ok (model listed + extra_body accepted on /chat/completions)
    captured_a: dict[str, Any] = {}

    def handler_ok(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "synth-1"}]})
        if req.url.path == "/chat/completions":
            captured_a["chat_body"] = json.loads(req.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )
        return httpx.Response(404)

    gw_a = _build(handler_ok, with_extra_body=True)
    results_a = await gw_a.probe_all()
    assert results_a["synthesis"] == "ok"
    assert results_a["synthesis.model"] == "ok"
    assert results_a["synthesis.extra_body"] == "extra_body_ok"
    assert captured_a["chat_body"].get("unique_param") == "xyz"

    # Case B: extra_body_rejected:400
    def handler_reject(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "synth-1"}]})
        if req.url.path == "/chat/completions":
            return httpx.Response(400, text="bad param")
        return httpx.Response(404)

    gw_b = _build(handler_reject, with_extra_body=True)
    results_b = await gw_b.probe_all()
    assert results_b["synthesis"] == "ok"
    assert results_b["synthesis.model"] == "ok"
    assert results_b["synthesis.extra_body"] == "extra_body_rejected:400"


@pytest.mark.asyncio
async def test_extra_body_verbatim_merge_into_chat_body() -> None:
    """extra_body keys appear verbatim in the chat request body."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/chat/completions":
            captured["body"] = json.loads(req.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )
        return httpx.Response(404)

    cfg = load_config_dict({
        "models": {
            "synthesis": {
                "base_url": "http://test",
                "model_id": "m",
                "extra_body": {"unique_param": "xyz", "top_k": 50, "flag": True},
            },
            "stt": {},
        }
    })
    gw = Gateway(cfg)
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await gw.chat("synthesis", [{"role": "user", "content": "hi"}])
    body = captured["body"]
    # All extra_body keys present, values preserved exactly
    assert "unique_param" in body
    assert body["unique_param"] == "xyz"
    assert body["top_k"] == 50
    assert body["flag"] is True

# ---------------------------------------------------------------------------
# STT health: streaming-only TCP probe (no HTTP dialects remain).
# ---------------------------------------------------------------------------


def _stt_cfg(port: int) -> Any:
    return load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://test", "model_id": "s"},
            "stt": {"stream_host": "127.0.0.1", "stream_port": port},
        }
    })


@pytest.mark.asyncio
async def test_stt_health_reports_listening_port_reachable() -> None:
    async def _noop(reader: object, writer: object) -> None:
        # Must close: wait_closed() below blocks until every accepted
        # connection is done.
        w = writer  # type: ignore[union-attr]
        w.close()
        try:
            await w.wait_closed()
        except OSError:
            pass

    server = await asyncio.start_server(_noop, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        gw = Gateway(_stt_cfg(port))
        gw._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(404))
        )
        reachable, detail = await gw.stt_health()
        assert reachable is True
        assert detail == f"tcp 127.0.0.1:{port} ok"
        await gw.aclose()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_stt_health_refused_port_reports_unreachable_tuple() -> None:
    """Regression: stt_health must always return a (bool, str) tuple — the
    old code fell through to None on some failures (uncaught TypeError at
    the tuple-unpack call site)."""
    gw = Gateway(_stt_cfg(1))  # nothing listens on port 1
    gw._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(404))
    )
    reachable, detail = await gw.stt_health()
    assert reachable is False
    assert "refused" in detail
    await gw.aclose()


@pytest.mark.asyncio
async def test_chat_thinking_param_injects_template_kwarg() -> None:
    """thinking=False sends chat_template_kwargs.enable_thinking=false."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    gw = _build(handler)
    await gw.chat("synthesis", [{"role": "user", "content": "hi"}], thinking=False)
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_chat_thinking_none_leaves_body_untouched() -> None:
    """Omitting thinking sends no template kwarg (config decides)."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    gw = _build(handler)
    await gw.chat("synthesis", [{"role": "user", "content": "hi"}])
    assert "chat_template_kwargs" not in captured["body"]


@pytest.mark.asyncio
async def test_chat_thinking_overrides_config_extra_body() -> None:
    """Explicit thinking wins over a conflicting config value."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    cfg = load_config_dict({
        "models": {
            "synthesis": {
                "base_url": "http://test",
                "model_id": "m",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
            "stt": {},
        }
    })
    gw = Gateway(cfg)
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await gw.chat("synthesis", [{"role": "user", "content": "hi"}], thinking=False)
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
