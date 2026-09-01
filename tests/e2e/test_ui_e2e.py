"""Browser E2E: real UI in Camoufox against the full wired stack.

Screenshots are written to screenshots/ and asserted for render sanity;
DOM assertions carry the functional weight.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import httpx
import pytest

from mock_server import _stable_vec  # reuse the deterministic vector helper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHOTS = PROJECT_ROOT / "screenshots"


def _browser_launchable() -> bool:
    """True if this host can create the semlock a headless browser needs.

    Some containers/PID namespaces block named-semaphore creation
    (``/dev/shm`` restricted), in which case Camoufox/Playwright cannot launch.
    The data path is still verified by the non-browser tests; the browser
    test is skipped here rather than failing for an environment reason.
    """
    import multiprocessing

    try:
        multiprocessing.Semaphore(1)
        return True
    except PermissionError:
        return False


def _png_dims(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _assert_rendered(path: Path, min_w: int = 800, min_h: int = 400) -> None:
    data = path.read_bytes()
    w, h = _png_dims(data)
    assert (w, h) >= (min_w, min_h), f"screenshot too small: {w}x{h}"
    assert len(data) > 8_000, f"screenshot suspiciously empty ({len(data)} bytes): {path.name}"


@pytest.mark.e2e
def test_ui_end_to_end_with_screenshots(stack) -> None:
    from camoufox.sync_api import Camoufox

    if not _browser_launchable():
        pytest.skip("host cannot launch headless browser (semlock PermissionError)")

    SHOTS.mkdir(exist_ok=True)
    with Camoufox(headless=True) as browser:
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto(stack.ui_url, wait_until="networkidle", timeout=30_000)
        page.wait_for_selector("#conn-dot.on", timeout=15_000)
        page.screenshot(path=str(SHOTS / "e2e_01_initial.png"), full_page=True)

        console_log: list[str] = []
        page.on("console", lambda m: console_log.append(f"{m.type}: {m.text}"))
        page.on("requestfailed", lambda r: console_log.append(f"REQFAIL: {r.url} {r.failure}"))

        status = httpx.get(f"{stack.ui_url}/api/status", timeout=5).json()
        assert status["project"] == "E2E Campaign"
        assert status["indexed_docs"] == 3
        assert status["entities"] == 2

        resp = page.request.post(
            f"{stack.ui_url}/api/query",
            data={"text": "what do i find when i search the body"},
            timeout=10_000,
        )
        assert resp.status == 200, f"query POST failed: {resp.status} {resp.text()}"
        assert resp.json().get("ok") is True, f"query rejected: {resp.text()}"

        try:
            page.wait_for_selector('[data-testid="card"]', timeout=20_000)
        except Exception:
            raise AssertionError(
                "card never rendered; console log:\n" + "\n".join(console_log[-40:])
            )
        title = page.text_content('[data-testid="card"] .card-title') or ""
        assert "Loot Table" in title
        body_text = page.text_content('[data-testid="card"] .card-body') or ""
        assert "Investigation" in body_text
        page.screenshot(path=str(SHOTS / "e2e_02_card.png"), full_page=True)

        stack.bus.publish_sync(
            {"type": "transcript", "user_id": "111111111111111111", "text": "I search the body", "t": time.time()}
        )
        stack.bus.publish_sync(
            {"type": "transcript", "user_id": "222222222222222222", "text": "roll for perception", "t": time.time() + 1}
        )
        page.wait_for_selector('[data-testid="transcript-line"]', timeout=10_000)
        n_lines = page.locator('[data-testid="transcript-line"]').count()
        assert n_lines >= 2

        page.screenshot(path=str(SHOTS / "e2e_03_transcript.png"), full_page=True)

    for name in ("e2e_01_initial.png", "e2e_02_card.png", "e2e_03_transcript.png"):
        _assert_rendered(SHOTS / name)

    recorded = httpx.get(f"{stack.mock_url}/_mock/requests", timeout=5).json()["requests"]
    # v2: the card is produced by the agentic worker on the synthesis (smart)
    # role. That call is free-form JSON in the message content, so it carries
    # no response_format — assert the synthesis model was actually invoked.
    synth_calls = [
        r
        for r in recorded
        if r["path"] == "/v1/chat/completions"
        and r["body"].get("model") == "mock-synthesis"
    ]
    assert len(synth_calls) >= 1, f"expected at least one synthesis (agent) call; got {len(synth_calls)}"


@pytest.mark.e2e
def test_mock_embeddings_are_stable(stack) -> None:
    a = _stable_vec("I search the body")
    b = _stable_vec("I search the body")
    c = _stable_vec("different text entirely")
    assert a == b
    assert a != c


def test_stt_script_endpoint_feeds_transcriptions(stack) -> None:
    """Scripted phrase flows out of the mock STT endpoint via a fresh Gateway.

    Engine-level transcription+correction composition is pinned by the
    regression golden; here we pin the endpoint dialect itself.
    """
    import asyncio

    from dmd.config import load_config_dict
    from dmd.gateway import Gateway

    r = httpx.post(
        f"{stack.mock_url}/_mock/stt_script",
        json={"texts": ["I search the body"]},
        timeout=5,
    )
    assert r.json() == {"queued": 1}

    cfg = load_config_dict(
        {
            "models": {
                "synthesis": {"base_url": stack.mock_url, "model_id": "mock-synthesis"},
                "stt": {"base_url": stack.mock_url},
            }
        }
    )

    async def _run():
        gw = Gateway(cfg)
        try:
            return await gw.transcribe(b"\x00\x00" * 320)
        finally:
            await gw.aclose()

    assert asyncio.run(_run()) == "I search the body"
