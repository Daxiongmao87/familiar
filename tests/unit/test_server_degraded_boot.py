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


DEGRADED_CFG = """\
project:
  name: Degraded Campaign
discord:
  token: test-token
  guild_id: '700822745491046411'
  dm_user_id: '188660400722673664'
"""


def _degraded_client(monkeypatch, tmp_path, text: str = DEGRADED_CFG) -> TestClient:
    """Serve main() against a present-but-invalid config (degraded, cfg None).

    The file parses as YAML and carries discord credentials, but whole-file
    validation fails (required models section missing) — the stale desktop
    config shape that bricked the DM dropdown on 2026-09-19.
    """
    import uvicorn

    import dmd.server as srv

    cfg_path = tmp_path / "familiar-config.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app))
    srv.main(str(cfg_path))  # must not raise
    return TestClient(captured["app"])


def test_degraded_dm_endpoint_reads_saved_id(monkeypatch, tmp_path) -> None:
    """The saved DM id survives degraded mode (read from the file, not cfg)."""
    client = _degraded_client(monkeypatch, tmp_path)
    resp = client.get("/api/config/dm")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["dm_user_id"] == "188660400722673664"


def test_degraded_members_falls_back_to_file_creds(monkeypatch, tmp_path) -> None:
    """Guild members load from file credentials when cfg failed validation."""
    import httpx

    seen: dict = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return [
                {
                    "user": {"id": "1", "username": "dm", "global_name": "DM"},
                    "nick": None,
                }
            ]

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> bool:
            return False

        async def get(self, url, **kwargs):
            seen["url"] = url
            seen["auth"] = kwargs.get("headers", {}).get("Authorization")
            return _FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    client = _degraded_client(monkeypatch, tmp_path)
    resp = client.get("/api/guild/members")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["members"][0]["username"] == "dm"
    assert "700822745491046411" in seen["url"]
    assert seen["auth"] == "Bot test-token"


def test_degraded_members_without_discord_is_honest(monkeypatch, tmp_path) -> None:
    """No discord in the file reports unconfigured, not a bare no-config."""
    client = _degraded_client(monkeypatch, tmp_path, "project:\n  name: X\n")
    body = client.get("/api/guild/members").json()
    assert body["ok"] is False
    assert body["detail"] == "discord not configured"
    assert body["members"] == []


def test_config_health_reports_degraded(monkeypatch, tmp_path) -> None:
    """Degraded boot exposes the validation failure instead of staying silent."""
    client = _degraded_client(monkeypatch, tmp_path)
    body = client.get("/api/config/health").json()
    assert body["ok"] is True
    assert body["valid"] is False
    assert body["detail"]  # names the validation failure


def test_config_health_reports_valid(tmp_path) -> None:
    """A validating config reports healthy through the same endpoint."""
    import dmd.server as srv
    from dmd.config import load_config

    cfg_path = tmp_path / "good.yaml"
    cfg_path.write_text(
        "models:\n"
        "  synthesis:\n"
        "    base_url: http://127.0.0.1:8080/v1\n"
        "    model_id: test-model\n"
        "  stt:\n"
        "    stream_host: 127.0.0.1\n"
        "    stream_port: 43007\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    client = TestClient(srv.create_app(cfg=cfg, config_path=str(cfg_path)))
    body = client.get("/api/config/health").json()
    assert body == {"ok": True, "valid": True, "detail": ""}
