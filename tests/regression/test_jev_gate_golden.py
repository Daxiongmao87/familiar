"""Live golden regression for the locked jev-gate-v39 prompt.

Scores the nine golden cases through a real ``OpenjevGate`` against a live
openjev-serve (default http://127.0.0.1:8199, override with JEV_GATE_URL).
Skips when the scorer is unreachable — but any change to prompt wording,
option descriptions/ordering, transcript construction, model/quant, or the
scorer readout MUST rerun this file against the production scorer before
it is accepted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from dmd.openjev import GATE_VERSION, OpenjevGate

GOLDEN = Path(__file__).parent / "golden" / "jev_gate_v39.json"
BASE_URL = os.environ.get("JEV_GATE_URL", "http://127.0.0.1:8199")
# Tripwire against margin collapse. Baseline gap is 0.785; anything under
# 0.5 means the lock's separation no longer holds.
MIN_SEPARATION_GAP = 0.5


def _load() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


async def _scorer_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{BASE_URL}/health")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def test_nine_golden_cases_classify_correctly() -> None:
    if not await _scorer_up():
        pytest.skip(f"openjev-serve unreachable at {BASE_URL}")
    spec = _load()
    assert spec["gate_version"] == GATE_VERSION
    gate = OpenjevGate(BASE_URL, threshold=spec["threshold"])
    try:
        results: list[tuple[str, str, bool, float]] = []
        for case in spec["cases"]:
            dec = await gate.decide(case["lines"])
            assert dec.error is None, f"{case['id']}: gate errored: {dec.error}"
            results.append((case["id"], case["expected"], dec.deploy, dec.prob))
    finally:
        await gate.aclose()
    bad = [
        f"{cid} (expected {exp}, p_deploy={p:.4f})"
        for cid, exp, got, p in results
        if got != (exp == "deploy")
    ]
    assert not bad, f"golden misclassifications: {bad}"
    deploy_ps = [p for _, exp, _, p in results if exp == "deploy"]
    wait_ps = [p for _, exp, _, p in results if exp == "wait"]
    gap = min(deploy_ps) - max(wait_ps)
    base = spec["baseline"]
    pinned = (spec.get("scorer") or {}).get("source", "?")
    live_model = "?"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            health = (await client.get(f"{BASE_URL}/health")).json()
        live_model = ((health.get("model") or {}).get("source", "?") + "@" +
                      ((health.get("model") or {}).get("revision", "?")[:12]))
    except httpx.HTTPError:
        pass
    print(
        f"\njev-gate-v39 live: min-deploy-p={min(deploy_ps):.4f} "
        f"(base {base['lowest_expected_deploy_p']:.4f}) "
        f"max-wait-p={max(wait_ps):.4f} "
        f"(base {base['highest_expected_wait_p']:.4f}) "
        f"gap={gap:.4f} (base {base['separation_gap']:.4f}) "
        f"scorer={live_model} (pinned {pinned})"
    )
    assert gap >= MIN_SEPARATION_GAP, f"separation collapsed: gap={gap:.4f}"
