"""Setup wizard UI: drive the real setup page headless with a stub bridge.

Serves electron/renderer/setup/ over localhost HTTP, stubs
window.familiarSetup (the preload bridge) with a canned mixed-capability
plan (WebGPU missing, CUDA present), and clicks through all three steps:
provider forcing on step 1, credentials + STT choice on step 2, install
screen on step 3. Screenshots land in /tmp (ephemeral audit aid).
"""
from __future__ import annotations

import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from camoufox.sync_api import Camoufox

SETUP_DIR = Path(__file__).resolve().parent.parent.parent / "electron" / "renderer" / "setup"

PLAN = {
    "providers": {"synthesis": "remote", "jev": "local", "stt": "remote"},
    "hardware": {"supported": False, "reason": "no-adapter"},
    "cuda": {"available": True, "gpus": ["Test GPU"], "detail": "CUDA via 1 GPU(s)"},
    "hwVerdict": {
        "synthesisLocal": {"ok": False, "reason": "WebGPU unavailable (no-adapter)"},
        "jevLocal": {"ok": True, "reason": "CUDA via 1 GPU(s)"},
        "sttLocal": {"ok": True, "reason": "CUDA via 1 GPU(s)"},
    },
    "forcedRemote": [{"provider": "synthesis", "reason": "WebGPU unavailable (no-adapter)"}],
    "saved": {
        "synthesis": {"base_url": "", "model_id": "", "has_api_key": False},
        "jev": {"base_url": ""},
        "stt": {"mode": "remote", "stream_host": "", "stream_port": "43007"},
        "discord": {"has_token": False, "guild_id": "", "dm_user_id": ""},
    },
    "wanted": ["jev-model"],
    "totalBytes": 100,
    "doneBytes": 0,
    "files": 2,
    "components": [{"id": "jev-model", "label": "Qwen 4B AWQ JEV model", "bytes": 100}],
}

STUB_JS = (
    "window.__savedSetup = null;\n"
    "window.familiarSetup = {\n"
    f"  _plan: {json.dumps(PLAN)},\n"
    "  getPlan: async function () { return structuredClone(this._plan); },\n"
    "  setProviders: async function (modes) {\n"
    "    Object.assign(this._plan.providers, modes); return structuredClone(this._plan);\n"
    "  },\n"
    "  saveSetup: async function (setup) {\n"
    "    window.__savedSetup = setup;\n"
    "    if (setup.stt) this._plan.providers.stt = setup.stt.mode;\n"
    "    return structuredClone(this._plan);\n"
    "  },\n"
    "  startInstall: async function () { return { ok: true }; },\n"
    "  boot: async function () { return { ok: true }; },\n"
    "  openAnyway: async function () { return { ok: true }; },\n"
    "  onState: function (fn) { setTimeout(() => fn(structuredClone(this._plan)), 10); },\n"
    "  onProgress: function () {},\n"
    "  onBoot: function () {},\n"
    "  onError: function () {},\n"
    "};\n"
)


@pytest.mark.e2e
def test_setup_wizard_three_steps() -> None:
    """Click through providers -> credentials -> install with assertions."""
    shots = Path("/tmp/setup_wizard_shots")
    shots.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(SETUP_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with Camoufox(headless=True) as browser:
            page = browser.new_page(viewport={"width": 780, "height": 780})
            page.add_init_script(STUB_JS)
            page.goto(f"{base}/index.html", wait_until="networkidle", timeout=15000)
            page.wait_for_selector("#comp-list .comp", timeout=5000)

            # Step 1: forced banner, disabled local synthesis, JEV local kept.
            assert page.locator("#forced-banner").is_visible()
            assert "synthesis" in (page.locator("#forced-banner").inner_text() or "")
            assert page.locator('input[name="synth"][value="local"]').is_disabled()
            assert page.locator('input[name="jev"][value="local"]').is_checked()
            assert "CUDA" in (page.locator("#cuda-line").inner_text() or "")
            page.screenshot(path=str(shots / "s1_providers.png"))

            # Step 2: credentials + endpoints.
            page.click("#btn-next")
            page.wait_for_selector("#step2:not([hidden])", timeout=5000)
            assert page.locator("#sec-syn-remote").is_visible(), "synth remote fields shown"
            assert not page.locator("#sec-jev-remote").is_visible(), "jev local: no URL needed"
            page.locator("#in-disc-token").fill("tok-test")
            page.locator("#in-disc-guild").fill("111")
            page.locator("#in-syn-base").fill("http://llm:8080/v1")
            page.locator("#in-syn-model").fill("m-test")
            page.locator("#in-stt-host").fill("192.168.0.50")
            page.locator("#in-stt-port").fill("43007")
            page.screenshot(path=str(shots / "s2_credentials.png"))

            # Save & Continue: payload shape verified through the stub.
            page.click("#btn-next")
            page.wait_for_selector("#overall-sec:not([hidden])", timeout=5000)
            saved = page.evaluate("window.__savedSetup")
            assert saved["discord"] == {"token": "tok-test", "guild_id": "111"}, saved
            assert saved["synthesisRemote"]["base_url"] == "http://llm:8080/v1"
            assert saved["synthesisRemote"]["model_id"] == "m-test"
            assert "jevRemote" not in saved, "jev local: no remote payload"
            assert saved["stt"]["mode"] == "remote"
            assert saved["stt"]["stream_host"] == "192.168.0.50"
            assert saved["stt"]["stream_port"] == "43007"
            page.screenshot(path=str(shots / "s3_install.png"))
            page.close()
    finally:
        server.shutdown()
    for name in ("s1_providers.png", "s2_credentials.png", "s3_install.png"):
        assert (shots / name).stat().st_size > 5000, name
