"""Openjev gate: binary deploy/wait over /score, kind on deploy, fail-closed."""

from __future__ import annotations

import json

import httpx
import pytest

from dmd.openjev import (
    DEPLOY_OPTIONS,
    DEPLOY_QUESTION,
    GATE_VERSION,
    REL_IDS,
    TIER_IDS,
    Debouncer,
    OpenjevError,
    OpenjevGate,
)


def _score_body(ids: list[str], probs: list[float]) -> bytes:
    return json.dumps({"id": "t", "option_ids": ids, "probabilities": probs}).encode()


def _gate(handler, **kw) -> OpenjevGate:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenjevGate("http://oj:8199", client=client, **kw)


async def test_deploy_runs_tier_stage_and_reports_both() -> None:
    calls: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode())
        calls.append(body)
        if body["options"][0]["id"] == "deploy":
            return httpx.Response(200, content=_score_body(["deploy", "wait"], [0.9, 0.1]))
        return httpx.Response(
            200, content=_score_body(list(TIER_IDS), [0.8, 0.2])
        )

    dec = await _gate(handler).decide(["dm: Make a deception check."])
    assert dec.deploy is True
    assert dec.error is None
    assert dec.prob == pytest.approx(0.9)
    assert dec.tier == "card"
    assert dec.tier_prob == pytest.approx(0.8)
    assert dec.latency_s >= 0.0
    assert len(calls) == 2
    assert "deception check" in calls[0]["state"]


async def test_tier_failure_fails_over_to_card() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode())
        if body["options"][0]["id"] == "deploy":
            return httpx.Response(200, content=_score_body(["deploy", "wait"], [0.9, 0.1]))
        return httpx.Response(500, content=b"boom")

    dec = await _gate(handler).decide(["dm: Make a deception check."])
    assert dec.deploy is True
    assert dec.tier == "card"
    assert dec.error is not None


async def test_wait_skips_tier_stage() -> None:
    calls: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content.decode()))
        return httpx.Response(200, content=_score_body(["deploy", "wait"], [0.2, 0.8]))

    dec = await _gate(handler).decide(["sam: (snarling noises)"])
    assert dec.deploy is False
    assert dec.tier == "ephemeral"
    assert dec.tier_prob == 0.0
    assert len(calls) == 1


async def test_threshold_boundary_deploys() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode())
        if body["options"][0]["id"] == "deploy":
            return httpx.Response(200, content=_score_body(["deploy", "wait"], [0.5, 0.5]))
        return httpx.Response(
            200, content=_score_body(list(TIER_IDS), [0.3, 0.7])
        )

    dec = await _gate(handler, threshold=0.5).decide(["liam: But roll with advantage."])
    assert dec.deploy is True
    assert dec.tier == "ephemeral"


async def test_http_error_fails_closed_to_wait() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    dec = await _gate(handler).decide(["matt: Roll initiative."])
    assert dec.deploy is False
    assert dec.tier == "ephemeral"
    assert dec.error is not None and "500" in dec.error


async def test_transport_error_fails_closed_to_wait() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    dec = await _gate(handler).decide(["matt: Roll initiative."])
    assert dec.deploy is False
    assert dec.error is not None


async def test_shape_mismatch_fails_closed_to_wait() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"option_ids": ["x"], "probabilities": []}')

    dec = await _gate(handler).decide(["matt: Roll initiative."])
    assert dec.deploy is False
    assert dec.error is not None


async def test_empty_window_waits_without_a_call() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    dec = await _gate(handler).decide(["   "])
    assert dec.deploy is False
    assert dec.latency_s == 0.0


async def test_recent_window_caps_state_lines() -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content.decode())["state"])
        return httpx.Response(200, content=_score_body(["deploy", "wait"], [0.1, 0.9]))

    lines = [f"u{i}: line {i}" for i in range(20)]
    await _gate(handler, recent_n=8).decide(lines)
    assert seen[0].splitlines() == lines[-8:]


async def test_health_ok_and_bad_shape() -> None:
    ok = _gate(lambda _r: httpx.Response(200, content=b'{"status": "ok"}'))
    assert (await ok.health())["status"] == "ok"
    bad = _gate(lambda _r: httpx.Response(200, content=b"[1]"))
    with pytest.raises(OpenjevError):
        await bad.health()


async def test_relevance_returns_argmax_over_three_branches() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_score_body(list(REL_IDS), [0.2, 0.7, 0.1])
        )

    v = await _gate(handler).relevance("NEED: x\nRETRIEVED (offline): ...")
    assert v.route == "online"
    assert v.error is None
    assert v.probs["online"] == pytest.approx(0.7)


async def test_relevance_fails_over_to_both() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    v = await _gate(handler).relevance("NEED: x")
    assert v.route == "both"
    assert v.error is not None


async def test_rank_returns_prob_map_for_open_options() -> None:
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content.decode()))
        return httpx.Response(200, content=_score_body(["a", "b"], [0.3, 0.7]))

    probs = await _gate(handler).rank(
        "t1", "state", "pick", [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]
    )
    assert probs == {"a": pytest.approx(0.3), "b": pytest.approx(0.7)}
    assert seen[0]["question"] == "pick"


async def test_rank_raises_on_transport_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OpenjevError):
        await _gate(handler).rank("t1", "s", "q", [{"id": "a", "description": "A"}])


def test_debouncer_drops_repeats_inside_window() -> None:
    d = Debouncer(window_s=30.0)
    assert d.check(100.0) is False  # first deploy passes
    assert d.check(110.0) is True  # repeat inside window drops
    assert d.check(131.0) is False  # window expired passes


def test_debouncer_zero_window_disables() -> None:
    d = Debouncer(window_s=0)
    assert d.check(100.0) is False
    assert d.check(100.1) is False


def test_production_prompt_is_locked_to_jev_gate_v39() -> None:
    """The v39 gate prompt is byte-pinned: any wording, id, or order
    change must be deliberate and must rerun the regression suite."""
    assert GATE_VERSION == "jev-gate-v39"
    assert DEPLOY_QUESTION == "Is there useful information work created by this exchange?"
    assert [o["id"] for o in DEPLOY_OPTIONS] == ["deploy", "wait"]
    assert DEPLOY_OPTIONS[0]["description"] == (
        "Yes. The exchange created a reason to retrieve, surface, clarify, "
        "or retain information that can materially help the DM handle the "
        "current situation or preserve continuity."
    )
    assert DEPLOY_OPTIONS[1]["description"] == (
        "No. The exchange created no meaningful information work. Additional "
        "context would be unnecessary noise, or the moment is incomplete, "
        "purely descriptive, sensory, scene-setting, or merely atmospheric."
    )


def test_production_threshold_defaults_to_half() -> None:
    from dmd.config import OpenjevConfig

    assert OpenjevGate("http://oj:8199")._threshold == 0.5
    assert OpenjevConfig().threshold == 0.5
