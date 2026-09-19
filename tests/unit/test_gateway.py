"""Tests for dmd.gateway — MockTransport-driven, no real network.

NOTE: Gateway.__init__ never initializes self._client. The assignment
`self._client = httpx.AsyncClient(...)` lives inside the @property cfg body
AFTER `return self._cfg` — unreachable dead code. The fixture below works
around this by setting gw._client manually after construction.

This file's tests pass; the init bug is documented and reported.
"""
from __future__ import annotations

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
            "stt": {
                "base_url": "http://test",
                "api_key": "sk-test" if with_api_key else None,
            },
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
async def test_transcribe_multipart_returns_text_field() -> None:
    """transcribe POSTs multipart/form-data and returns the 'text' field."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["content_type"] = req.headers.get("content-type", "")
        captured["path"] = req.url.path
        captured["method"] = req.method
        return httpx.Response(200, json={"text": "transcribed content"})

    gw = _build(handler)
    out = await gw.transcribe(b"\x00\x00\x00\x00FAKEWAV", filename="chunk.wav")
    assert out == "transcribed content"
    assert captured["path"] == "/audio/transcriptions"
    assert captured["method"] == "POST"
    assert captured["content_type"].startswith("multipart/form-data")


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
            "stt": {"base_url": "http://test"},
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
# SPEC §2 zero-hardcoding: whisperx diarize/align come from config, not the
# gateway body (the old code hardcoded diarize=false&align=false).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whisperx_params_reflect_config_diarize_align() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/transcribe":
            captured["params"] = dict(req.url.params)
            return httpx.Response(200, json={"text": "hello"})
        return httpx.Response(404)

    def build(diarize: bool, align: bool) -> Gateway:
        cfg = load_config_dict({
            "models": {
                "synthesis": {"base_url": "http://test", "model_id": "s"},
                "stt": {
                    "base_url": "http://test",
                    "dialect": "whisperx",
                    "diarize": diarize,
                    "align": align,
                },
            }
        })
        gw = Gateway(cfg)
        gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return gw

    await build(True, False).transcribe(b"WAV")
    assert captured["params"] == {"diarize": "true", "align": "false"}
    await build(False, True).transcribe(b"WAV")
    assert captured["params"] == {"diarize": "false", "align": "true"}


@pytest.mark.asyncio
async def test_transcribe_diarized_returns_speaker_segments() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "text": "one two",
                "segments": [
                    {"speaker": "SPEAKER_00", "start": 0.1, "end": 1.0, "text": " one "},
                    {"speaker": "SPEAKER_01", "start": 1.2, "end": 2.0, "text": " two "},
                ],
            },
        )

    cfg = load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://test", "model_id": "s"},
            "stt": {"base_url": "http://test", "dialect": "whisperx"},
        }
    })
    gw = Gateway(cfg)
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    text, segments = await gw.transcribe_diarized(b"WAV")
    assert text == "one two"
    assert segments == [
        {"speaker": "SPEAKER_00", "start": 0.1, "end": 1.0, "text": "one"},
        {"speaker": "SPEAKER_01", "start": 1.2, "end": 2.0, "text": "two"},
    ]


@pytest.mark.asyncio
async def test_stt_health_5xx_reports_unreachable() -> None:
    """Regression: stt_health fell through to None on 5xx (uncaught TypeError
    at the tuple-unpack call site)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="models not resident")

    cfg = load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://test", "model_id": "s"},
            "stt": {"base_url": "http://test", "dialect": "whisperx"},
        }
    })
    gw = Gateway(cfg)
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reachable, detail = await gw.stt_health()
    assert reachable is False
    assert "503" in detail


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
            "stt": {"base_url": "http://test"},
        }
    })
    gw = Gateway(cfg)
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await gw.chat("synthesis", [{"role": "user", "content": "hi"}], thinking=False)
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
