"""Sidecar contract tests: vendored scorer HTTP layer without torch/CUDA.

The vendored openjev modules import torch lazily, so the /score + /health
HTTP contract (validation, shapes, status codes) is testable with a fake
scorer. Live weight parity lives in test_jev_local_parity.py (skipped
without a running sidecar).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pytest

SIDECAR = Path(__file__).resolve().parent.parent.parent / "services" / "jev_sidecar"
VENDORED = SIDECAR / "_vendored"


def _load_vendored():
    prov = VENDORED / "PROVENANCE.json"
    if not prov.exists():
        pytest.skip("sidecar not vendored (run services/jev_sidecar/vendor.py)")
    sys.path.insert(0, str(VENDORED))
    import openjev_phase1.core as core
    import openjev_phase1.server as server_mod

    return core, server_mod


def _serve(server_mod, scorer, health):
    srv = server_mod.build_server("127.0.0.1", 0, scorer, health)
    import threading

    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def test_vendor_check_passes():
    """The vendored tree matches the pinned sibling (no drift)."""
    import subprocess

    if not (SIDECAR / "_vendored" / "PROVENANCE.json").exists():
        pytest.skip("sidecar not vendored")
    proc = subprocess.run(
        [sys.executable, str(SIDECAR / "vendor.py"), "--check"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_score_contract_with_fake_scorer():
    """HTTP layer validates rows and serves scorer results verbatim."""
    _core, server_mod = _load_vendored()

    def scorer(row):
        assert row["id"] == "r1"
        return {"id": "r1", "option_ids": ["deploy", "wait"],
                "probabilities": [0.8, 0.2]}

    srv = _serve(server_mod, scorer, {"model": {"source": "fake"}})
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        status, body = _post(base + "/score", {
            "id": "r1", "state": "s", "question": "q?",
            "options": [{"id": "deploy", "description": "d"},
                        {"id": "wait", "description": "w"}],
        })
        assert status == 200
        assert body["option_ids"] == ["deploy", "wait"]
        assert body["probabilities"] == [0.8, 0.2]
        # Invalid rows are 400s, never scorer calls.
        status, body = _post(base + "/score", {"id": "bad"})
        assert status == 400
        assert "error" in body
        status, _ = _post(base + "/nope", {})
        assert status == 404
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            health = json.loads(resp.read().decode())
        assert health["status"] == "ok"
        assert health["model"]["source"] == "fake"
    finally:
        srv.shutdown()


def test_row_validation_matches_openjev_rules():
    """validate_row enforces the same contract Familiar relies on."""
    core, _ = _load_vendored()
    good = {"id": "r", "state": "s", "question": "q",
            "options": [{"id": "a", "description": "x"},
                        {"id": "b", "description": "y"}]}
    core.validate_row(good)  # must not raise
    bad_rows = [
        dict(good, options=[{"id": "a", "description": "x"}]),  # <2
        dict(good, options=[{"id": "a", "description": "x"},
                            {"id": "a", "description": "y"}]),  # dup ids
        dict(good, state=""),  # empty state
        {"id": "r"},  # missing fields
    ]
    for row in bad_rows:
        with pytest.raises(ValueError):
            core.validate_row(row)
    assert core.softmax([0.0, 0.0]) == pytest.approx([0.5, 0.5])
