"""Live provisional transcript rendering through the real event bus."""

from pathlib import Path

from camoufox.sync_api import Camoufox


def test_partial_visible_before_final_and_replaced_after_attribution(stack):
    """Growing speech is visible before finalization without duplicate rows."""
    from playwright.sync_api import expect

    with Camoufox(headless=True) as browser:
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(stack.ui_url, wait_until="networkidle")
        page.wait_for_selector("#conn-dot.on")
        stack.bus.publish_sync({"type": "transcript_partial", "user_id": "browser_mixed",
                                "text": "I search"})
        partial = page.locator('[data-testid="transcript-partial"]')
        expect(partial).to_contain_text("I search", timeout=5000)
        expect(page.locator('[data-testid="transcript-line"]')).to_have_count(0)
        stack.bus.publish_sync({"type": "transcript_partial", "user_id": "browser_mixed",
                                "text": "I search the goblin corpse"})
        expect(partial).to_have_count(1)
        expect(partial).to_contain_text("I search the goblin corpse")
        shots = Path(__file__).resolve().parents[2] / "screenshots"
        shots.mkdir(exist_ok=True)
        page.screenshot(path=str(shots / "streaming-partial.png"), full_page=True)
        stack.bus.publish_sync({"type": "transcript_partial", "user_id": "browser_mixed",
                                "text": ""})
        stack.bus.publish_sync({"type": "transcript", "user_id": "player-1",
                                "name": "Mira", "text": "I search the goblin corpse"})
        expect(partial).to_have_count(0)
        expect(page.locator('[data-testid="transcript-line"]')).to_have_count(1)
        expect(page.locator('[data-testid="transcript-line"]')).to_contain_text("Mira")
        page.screenshot(path=str(shots / "streaming-final.png"), full_page=True)
