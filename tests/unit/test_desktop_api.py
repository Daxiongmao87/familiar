"""Tests for dmd.desktop_api: status probes and provider switching.

Switching applies live to the config object and the engine gate, persists
to the YAML file, and never echoes secrets back to the caller.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dmd.config import load_config_dict
from dmd.desktop_api import mount_desktop_api

REMOTE_SYNTH = "http://remote-llm:8080/v1"
REMOTE_JEV = "http://remote-jev:8199"


def _cfg() -> Any:
    return load_config_dict(
        {
            "models": {
                "synthesis": {
                    "provider": "remote",
                    "base_url": REMOTE_SYNTH,
                    "model_id": "minicpm5-2b",
                    "api_key": "sk-live-secret",
                },
                "stt": {"base_url": "http://stt"},
            },
            "openjev": {
                "enabled": True,
                "provider": "remote",
                "base_url": REMOTE_JEV,
                "timeout_s": 1.0,
            },
            "desktop": {
                "enabled": True,
                "bridge_url": "http://127.0.0.1:9",
                "jev_local_url": "http://127.0.0.1:9",
            },
        }
    )


class _Engine:
    """Minimal engine double: records apply_providers calls."""

    def __init__(self) -> None:
        self.calls = 0

    def apply_providers(self) -> dict[str, str]:
        self.calls += 1
        return {"synthesis": "eff-s", "jev": "eff-j"}


def _client(cfg: Any, engine: Any, path: Path | None) -> TestClient:
    app = FastAPI()
    mount_desktop_api(
        app, cfg=cfg, engine=engine, gateway=None,
        config_path=str(path) if path else None,
    )
    return TestClient(app)


def test_status_reports_routing_without_secrets() -> None:
    """GET /api/desktop/status describes routing; keys stay masked."""
    client = _client(_cfg(), _Engine(), None)
    resp = client.get("/api/desktop/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["synthesis"]["provider"] == "remote"
    assert body["synthesis"]["has_api_key"] is True
    assert body["jev"]["provider"] == "remote"
    # Unreachable test ports: probes fail fast and report, never raise.
    assert body["synthesis_reachable"] is False
    assert body["jev_reachable"] is False
    assert "sk-live-secret" not in resp.text


def test_switch_applies_live_and_persists(tmp_path: Path) -> None:
    """POST flips routing live, repoints the engine, and writes YAML."""
    cfg = _cfg()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "synthesis": {
                        "provider": "remote",
                        "base_url": REMOTE_SYNTH,
                        "model_id": "minicpm5-2b",
                    }
                },
                "openjev": {"provider": "remote", "base_url": REMOTE_JEV},
            }
        ),
        encoding="utf-8",
    )
    engine = _Engine()
    client = _client(cfg, engine, cfg_path)
    resp = client.post(
        "/api/desktop/providers",
        json={"synthesis": {"provider": "local"}, "jev": {"provider": "local"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["persisted"] is True
    assert engine.calls == 1
    assert cfg.models.synthesis.provider == "local"
    assert cfg.openjev.provider == "local"
    # Remote URLs survive in both the live object and the file.
    assert cfg.models.synthesis.base_url == REMOTE_SYNTH
    on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert on_disk["models"]["synthesis"]["provider"] == "local"
    assert on_disk["models"]["synthesis"]["base_url"] == REMOTE_SYNTH
    assert on_disk["openjev"]["provider"] == "local"


def test_switch_remote_fields_without_secret_echo(tmp_path: Path) -> None:
    """Remote URL/model/key updates persist; the key is never returned."""
    cfg = _cfg()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}", encoding="utf-8")
    client = _client(cfg, _Engine(), cfg_path)
    resp = client.post(
        "/api/desktop/providers",
        json={
            "synthesis": {
                "provider": "remote",
                "base_url": "http://new:8080/v1/",
                "model_id": "other-model",
                "api_key": "sk-brand-new",
            }
        },
    )
    assert resp.json()["ok"] is True
    assert cfg.models.synthesis.base_url == "http://new:8080/v1"
    assert cfg.models.synthesis.api_key == "sk-brand-new"
    assert "sk-brand-new" not in resp.text


def test_switch_rejects_bad_input() -> None:
    """Typos and malformed values are rejected, never half-applied."""
    cfg = _cfg()
    client = _client(cfg, _Engine(), None)
    for payload in (
        {"synthesis": {"provider": "cloud"}},
        {"synthesis": {"base_url": "not-a-url"}},
        {"synthesis": {"model_id": "  "}},
        {"synthesis": {"nope": 1}},
        {"jev": {"provider": "local", "api_key": "x"}},
        {"other": {}},
        {},
        [],
    ):
        resp = client.post("/api/desktop/providers", json=payload)
        assert resp.json()["ok"] is False, payload
    assert cfg.models.synthesis.provider == "remote"
    assert cfg.openjev.provider == "remote"


def test_switch_without_config_file_still_applies_live() -> None:
    """A missing config path degrades to in-memory apply (persisted:false)."""
    cfg = _cfg()
    client = _client(cfg, _Engine(), None)
    resp = client.post(
        "/api/desktop/providers", json={"synthesis": {"provider": "local"}}
    )
    body = resp.json()
    assert body["ok"] is True
    assert body["persisted"] is False
    assert cfg.models.synthesis.provider == "local"
