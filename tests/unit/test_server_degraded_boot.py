"""Regression: main() with an unreadable config serves degraded UI.

The degraded path (missing/invalid config.yaml) used to die in main()
with UnboundLocalError: stt_monitor was only bound inside the
gateway-ok branch but read unconditionally when building the status
provider. main() must serve the UI (degraded) instead of crashing.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_main_missing_config_serves_degraded(monkeypatch, tmp_path) -> None:
    """A missing config file yields a serving app, not an exception."""
    import uvicorn

    import dmd.server as srv

    captured: dict = {}

    def fake_run(app, **kwargs) -> None:
        captured["app"] = app

    monkeypatch.setattr(uvicorn, "run", fake_run)
    srv.main(str(tmp_path / "no-such-config.yaml"))  # must not raise
    client = TestClient(captured["app"])
    resp = client.get("/api/status")
    assert resp.status_code == 200
    resp = client.get("/api/desktop/status")
    assert resp.status_code == 200
    assert resp.json()["ok"] is False  # no config, honestly reported


def test_main_invalid_config_serves_degraded(monkeypatch, tmp_path) -> None:
    """Garbage YAML takes the same degraded path as a missing file."""
    import uvicorn

    import dmd.server as srv

    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("{{{not yaml", encoding="utf-8")
    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app))
    srv.main(str(cfg_path))  # must not raise
    client = TestClient(captured["app"])
    assert client.get("/api/status").status_code == 200
