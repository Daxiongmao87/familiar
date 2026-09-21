"""Desktop provider-matrix smoke: real bridge + real Gateway, no mocks.

Spins the REAL Electron bridge (node, electron/main/bridge.js) with a
canned inference function and a REAL openjev-protocol JEV stub, then
exercises all four provider combinations through the REAL Gateway +
OpenjevGate + desktop API, plus a dev-backend boot check.

Usage: python3 scripts/desktop_e2e.py
Exit 0 when every check passes; prints a per-check report either way.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BRIDGE_JS = REPO / "electron" / "main" / "bridge.js"

NODE_STUB = """\
const { startBridge } = require(%s);
const seen = [];
startBridge({
  host: '127.0.0.1', port: %d, modelDir: %s,
  status: () => ({ modelLoaded: true, webgpu: { supported: true } }),
  infer: async (p) => {
    seen.push(p);
    const content = p.json ? JSON.stringify({ title: 'Stub card', body_md: 'ok' }) : 'stub says hi';
    return { content, usage: { prompt_tokens: 1 } };
  },
}).then((s) => console.log('BRIDGE_UP ' + s.address().port));
setTimeout(() => {}, 1 << 30);
"""


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_http(url: str, timeout_s: float = 15.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except OSError:
            time.sleep(0.2)
    return False


class Report:
    """Collects named check outcomes and prints a summary."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        """Record one check result."""
        self.rows.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    def passed(self) -> bool:
        """True when every recorded check passed."""
        return all(ok for _, ok, _ in self.rows)


def _cfg(synth: str, jev: str, bridge_url: str, jev_url: str) -> dict:
    return {
        "models": {
            "synthesis": {
                "provider": synth,
                "base_url": "http://remote-llm.test/v1",
                "model_id": "minicpm5-2b",
                "api_key": "sk-x",
            },
            "stt": {"stream_host": "127.0.0.1", "stream_port": 1},
        },
        "openjev": {"provider": jev, "base_url": jev_url + "-remote"},
        "desktop": {"enabled": True, "bridge_url": bridge_url, "jev_local_url": jev_url},
    }


async def _matrix(report: Report, bridge_url: str, jev_url: str) -> None:
    from dmd.config import load_config_dict
    from dmd.gateway import Gateway
    from dmd.openjev import OpenjevGate
    from dmd.providers import EndpointRouter

    for synth, jev in (("remote", "remote"), ("remote", "local"),
                       ("local", "remote"), ("local", "local")):
        cfg = load_config_dict(_cfg(synth, jev, bridge_url, jev_url))
        router = EndpointRouter(cfg)
        eff = router.resolve("synthesis").base_url
        want = bridge_url if synth == "local" else "http://remote-llm.test/v1"
        report.check(f"matrix {synth}/{jev}: synthesis routes", eff == want, eff)
        report.check(f"matrix {synth}/{jev}: jev routes",
                     router.jev_base_url() == (jev_url if jev == "local" else jev_url + "-remote"))
        if synth == "local":
            gw = Gateway(cfg, router=router)
            try:
                out = await gw.chat("synthesis", [{"role": "user", "content": "hi"}],
                                    json_schema={"type": "object"})
                report.check(f"matrix {synth}/{jev}: local chat JSON", out == {
                             "title": "Stub card", "body_md": "ok"}, repr(out)[:80])
            finally:
                await gw.aclose()
        if jev == "local":
            gate = OpenjevGate(router.jev_base_url(), threshold=0.5)
            try:
                dec = await gate.decide(["dm: Roll initiative.", "sam: I got a 19."])
                report.check(f"matrix {synth}/{jev}: local score flows",
                             dec.deploy is True and abs(dec.prob - 0.8) < 1e-9,
                             f"deploy={dec.deploy} p={dec.prob}")
            finally:
                await gate.aclose()


def _api_switch(report: Report, bridge_url: str, jev_url: str) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from dmd.config import load_config_dict
    from dmd.desktop_api import mount_desktop_api

    cfg = load_config_dict(_cfg("remote", "remote", bridge_url, jev_url))

    class Engine:
        def __init__(self) -> None:
            self.calls = 0

        def apply_providers(self) -> dict:
            self.calls += 1
            from dmd.providers import EndpointRouter

            router = EndpointRouter(cfg)
            return {"synthesis": router.resolve("synthesis").base_url,
                    "jev": router.jev_base_url()}

    engine = Engine()
    app = FastAPI()
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "config.yaml")
        Path(cfg_path).write_text("{}", encoding="utf-8")
        mount_desktop_api(app, cfg=cfg, engine=engine, gateway=None, config_path=cfg_path)
        client = TestClient(app)
        status = client.get("/api/desktop/status").json()
        report.check("desktop status shape", status["ok"] is True
                     and "sk-x" not in json.dumps(status))
        resp = client.post("/api/desktop/providers",
                           json={"synthesis": {"provider": "local"},
                                 "jev": {"provider": "local"}}).json()
        report.check("provider switch applies + persists",
                     resp["ok"] is True and resp["persisted"] is True
                     and resp["effective"]["synthesis"] == bridge_url
                     and engine.calls == 1, repr(resp["effective"]))


def _backend_boot(report: Report) -> None:
    venv_py = REPO / ".venv" / "bin" / "python"
    if not venv_py.exists():
        report.check("backend boot (dev)", False, "no .venv")
        return
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "e2e-config.yaml")
        Path(cfg_path).write_text(
            "project:\n  path: " + tmp + "\n  name: E2E\n"
            "models:\n"
            "  synthesis:\n    provider: remote\n"
            "    base_url: http://127.0.0.1:9/v1\n    model_id: m\n"
            "  stt:\n    stream_host: 127.0.0.1\n    stream_port: 1\n"
            "openjev:\n  base_url: http://127.0.0.1:9\n"
            "server:\n  host: 127.0.0.1\n  port: " + str(port) + "\n",
            encoding="utf-8",
        )
        env = dict(os.environ, PYTHONPATH=str(REPO))
        log_path = os.path.join(tmp, "backend.log")
        log_file = open(log_path, "wb")
        proc = subprocess.Popen(
            [str(venv_py), str(REPO / "dmd" / "server.py"), cfg_path],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            up = _wait_http(base + "/api/status", 60.0)
            detail = ""
            if not up:
                log_file.flush()
                try:
                    tail = Path(log_path).read_bytes().decode("utf-8", "replace")[-600:]
                    detail = "log tail: " + " | ".join(tail.splitlines()[-4:])
                except OSError:
                    detail = "no log"
            report.check("backend boots degraded (dead endpoints)", up, detail)
            if up:
                with urllib.request.urlopen(base + "/api/desktop/status", timeout=10) as resp:
                    status = json.loads(resp.read().decode())
                report.check("desktop status on live backend",
                             status["ok"] is True
                             and status["synthesis_reachable"] is False,
                             f"reachable={status['synthesis_reachable']}")
                with urllib.request.urlopen(base + "/", timeout=10) as resp:
                    html = resp.read().decode()
                report.check("live UI served", "Familiar" in html or "DM Copilot" in html)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                report.check("clean backend shutdown", False, "needed SIGKILL")
            else:
                report.check("clean backend shutdown", True)


def main() -> int:
    """Run the smoke matrix and report."""
    report = Report()
    if not BRIDGE_JS.exists():
        print("missing electron/main/bridge.js")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "model")
        os.makedirs(model_dir)
        Path(model_dir, "mlc-chat-config.json").write_text("{}", encoding="utf-8")
        bridge_port = _free_port()
        stub_path = os.path.join(tmp, "bridge_stub.js")
        Path(stub_path).write_text(
            NODE_STUB % (json.dumps(str(BRIDGE_JS)), bridge_port, json.dumps(model_dir)),
            encoding="utf-8",
        )
        bridge = subprocess.Popen(["node", stub_path], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
        jev_url = ""
        try:
            bridge_url = f"http://127.0.0.1:{bridge_port}"
            if not _wait_http(bridge_url + "/models"):
                report.check("node bridge boots", False, "no /models")
                return 1
            report.check("node bridge boots", True, bridge_url)
            # JEV stub speaking the openjev protocol (contract proven
            # against the real vendored server in test_sidecar_contract).
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            class Handler(BaseHTTPRequestHandler):
                def _send(self, code: int, payload: dict) -> None:
                    data = json.dumps(payload).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def do_GET(self) -> None:
                    if self.path == "/health":
                        self._send(200, {"status": "ok",
                                         "model": {"revision": "stub"}})
                    else:
                        self._send(404, {"error": "x"})

                def do_POST(self) -> None:
                    length = int(self.headers.get("Content-Length", "0"))
                    row = json.loads(self.rfile.read(length).decode())
                    ids = [o["id"] for o in row["options"]]
                    probs = [0.8 if i == "deploy" else 0.2 / max(1, len(ids) - 1)
                             if "deploy" in ids else 1.0 / len(ids) for i in ids]
                    total = sum(probs)
                    self._send(200, {"id": row["id"], "option_ids": ids,
                                     "probabilities": [p / total for p in probs]})

                def log_message(self, *args) -> None:
                    pass

            jev_port = _free_port()
            jev_url = f"http://127.0.0.1:{jev_port}"
            jev_srv = ThreadingHTTPServer(("127.0.0.1", jev_port), Handler)
            import threading

            thread = threading.Thread(target=jev_srv.serve_forever, daemon=True)
            thread.start()
            asyncio.run(_matrix(report, bridge_url, jev_url))
            _api_switch(report, bridge_url, jev_url)
            jev_srv.shutdown()
        finally:
            bridge.terminate()
            bridge.wait(timeout=10)
    _backend_boot(report)
    print(f"\n{'ALL CHECKS PASSED' if report.passed() else 'FAILURES PRESENT'}")
    return 0 if report.passed() else 1


if __name__ == "__main__":
    sys.exit(main())
