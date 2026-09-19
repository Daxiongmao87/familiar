"""UX pass: drive the real UI in a browser against the full mock stack and
capture screenshots at each interaction state for visual inspection.

This is a verification aid (ephemeral). It exercises the same paths as the
functional e2e tests but focuses on how the rendered surface behaves, not just
DOM assertions. The teardown "event loop closed" error is a Playwright/Coufox
shutdown artifact; the captures and asserts all run before teardown.
"""

import time
from pathlib import Path

import pytest
from camoufox.sync_api import Camoufox


@pytest.mark.e2e
def test_ui_pass_captures(stack) -> None:
    """Render initial, card, transcript, and settings states; save PNGs."""
    SHOTS = Path("/tmp/ui_pass_shots")
    SHOTS.mkdir(parents=True, exist_ok=True)

    with Camoufox(headless=True) as b:
        page = b.new_page(viewport={"width": 1280, "height": 800})
        page.goto(f"{stack.ui_url}", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#conn-dot.on", timeout=15000)
        page.screenshot(path=str(SHOTS / "01_initial.png"), full_page=True)

        resp = page.request.post(
            f"{stack.ui_url}/api/query",
            data={"text": "what do i find when i search the body"},
            timeout=15000,
        )
        assert resp.status == 200, f"query POST failed: {resp.status} {resp.text()}"
        _wait_card(page)
        page.screenshot(path=str(SHOTS / "02_card.png"), full_page=True)

        stack.bus.publish_sync(
            {
                "type": "transcript",
                "user_id": "111111111111111111",
                "text": "I search the body",
                "t": time.time(),
            }
        )
        page.wait_for_selector('[data-testid="transcript-line"]', timeout=10000)
        page.screenshot(path=str(SHOTS / "03_transcript.png"), full_page=True)

        page.click("#settings-btn", timeout=5000)
        page.wait_for_selector("#settings-panel", timeout=5000)
        page.screenshot(path=str(SHOTS / "04_settings.png"), full_page=True)

        page.close()

    for name in ("01_initial.png", "02_card.png", "03_transcript.png", "04_settings.png"):
        p = SHOTS / name
        assert p.exists(), f"screenshot not captured: {name}"
        assert p.stat().st_size > 5000, (
            f"screenshot {name} suspiciously small: {p.stat().st_size} bytes"
        )


@pytest.mark.e2e
def test_ui_pass_mobile_and_dropdown(stack) -> None:
    """Verify responsive layout at mobile width and the empty DM dropdown."""
    SHOTS = Path("/tmp/ui_pass_shots")
    SHOTS.mkdir(parents=True, exist_ok=True)

    with Camoufox(headless=True) as b:
        page = b.new_page(viewport={"width": 390, "height": 844})
        page.goto(f"{stack.ui_url}", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#conn-dot.on", timeout=15000)
        page.screenshot(path=str(SHOTS / "m1_initial_mobile.png"), full_page=True)

        # DM dropdown: empty-state is correct when Discord is not configured
        # (members endpoint returns []). The dropdown must not error.
        page.locator("#dm-input").click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(SHOTS / "m2_dmdropdown.png"), full_page=True)

        assert page.locator("#dm-dropdown .dm-username").count() == 0, (
            "expected no members when Discord is not configured"
        )

        page.close()


def _wait_card(page, timeout=20000) -> None:
    deadline = time.monotonic() + timeout / 1000.0
    while time.monotonic() < deadline:
        if page.locator('[data-testid="card"]').count() > 0:
            return
        time.sleep(0.15)
    raise AssertionError("card never rendered")
