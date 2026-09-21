"""Settings save flow: providers apply live through the settings modal.

Opens the (previously unreachable) settings modal in a real browser,
changes the JEV endpoint + provider selects, saves, and asserts the
desktop status reflects the new routing without a restart.
"""
from __future__ import annotations

import pytest
from camoufox.sync_api import Camoufox


@pytest.mark.e2e
def test_settings_save_applies_providers_live(stack) -> None:
    """Provider fields save through /api/desktop/providers; status updates."""
    with Camoufox(headless=True) as browser:
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(f"{stack.ui_url}", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#conn-dot.on", timeout=15000)
        page.click("#settings-btn", timeout=5000)
        page.wait_for_selector("#settings-panel", timeout=5000)
        page.wait_for_function(
            "() => (document.getElementById('cfg-jev-base').value || '').length > 0",
            timeout=10000,
        )
        page.select_option("#cfg-syn-provider", "remote")
        page.select_option("#cfg-jev-provider", "remote")
        page.locator("#cfg-jev-base").fill("http://127.0.0.1:8299")
        page.click("#settings-save", timeout=5000)
        page.wait_for_selector("#settings-status", timeout=5000)
        page.wait_for_function(
            "() => (document.getElementById('settings-status').textContent || '')"
            ".includes('Providers applied live')",
            timeout=10000,
        )
        resp = page.request.get(f"{stack.ui_url}/api/desktop/status", timeout=10000)
        assert resp.status == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["jev"]["provider"] == "remote"
        assert body["jev"]["base_url"] == "http://127.0.0.1:8299"
        assert body["synthesis"]["provider"] == "remote"
        assert "e2e-key" not in resp.text(), "secrets never echoed"
        page.close()
