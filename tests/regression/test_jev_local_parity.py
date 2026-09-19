"""Live parity: local JEV sidecar vs the locked jev-gate-v39 golden.

Same shape as test_jev_gate_golden.py but against the LOCAL sidecar
(default http://127.0.0.1:8299, override with JEV_LOCAL_URL). Skips when
the sidecar is unreachable — and refuses to run when the sidecar's
/health model revision differs from the manifest pin, so a stale local
model can never silently stand in for the golden comparison.

The sidecar runs the identical vendored algorithm over the identical
pinned AWQ weights, so zero drift is expected; any gap change vs the
baseline is printed and the separation tripwire still applies.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from dmd.openjev import GATE_VERSION, OpenjevGate

GOLDEN = Path(__file__).parent / "golden" / "jev_gate_v39.json"
BASE_URL = os.environ.get("JEV_LOCAL_URL", "http://127.0.0.1:8299")
MIN_SEPARATION_GAP = 0.5


def _manifest_pin() -> str:
    manifest = (
        Path(__file__).parent.parent.parent
        / "electron" / "shared" / "manifest.json"
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for comp in data["components"]:
        if comp["id"] == "jev-model":
            return str(comp.get("revision") or "")
    return ""


async def _sidecar_info() -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{BASE_URL}/health")
            if resp.status_code != 200:
                return None
            return resp.json()
    except httpx.HTTPError:
        return None


async def test_local_sidecar_matches_golden():
    """Nine golden cases classify identically through the local sidecar."""
    health = await _sidecar_info()
    if health is None:
        pytest.skip(f"local JEV sidecar unreachable at {BASE_URL}")
    live_rev = ((health.get("model") or {}).get("revision") or "")
    want_rev = _manifest_pin()
    if want_rev and live_rev != want_rev:
        pytest.skip(
            f"sidecar weights {live_rev[:12]} != manifest pin {want_rev[:12]}"
        )
    spec = json.loads(GOLDEN.read_text(encoding="utf-8"))
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
    assert not bad, f"local-sidecar misclassifications: {bad}"
    deploy_ps = [p for _, exp, _, p in results if exp == "deploy"]
    wait_ps = [p for _, exp, _, p in results if exp == "wait"]
    gap = min(deploy_ps) - max(wait_ps)
    assert gap >= MIN_SEPARATION_GAP, f"separation collapsed: {gap:.4f}"
    base = spec["baseline"]
    print(
        f"\njev-local parity: min-deploy-p={min(deploy_ps):.4f} "
        f"(base {base['lowest_expected_deploy_p']:.4f}) "
        f"max-wait-p={max(wait_ps):.4f} "
        f"(base {base['highest_expected_wait_p']:.4f}) "
        f"gap={gap:.4f} (base {base['separation_gap']:.4f}) "
        f"rev={live_rev[:12]}"
    )
