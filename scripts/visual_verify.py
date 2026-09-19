"""End-to-end visual verification of the live familiar UI + E2E card re-probe.

Launches a headless Playwright chromium against the live HTTPS service
(``https://127.0.0.1:8765/``), navigates there, waits for the DOM to settle,
and captures screenshots at desktop (1440px) and mobile (380px) viewports.
Also performs a post-restart E2E probe: POST /api/query, poll GET /api/cards
until a NON-error card appears, and assert the produced card has populated
body_md.

Self-signed cert is bypassed with ``--ignore-certificate-errors`` in the
launch args plus ``ignore_https_errors=True`` on the context. NOTE: neither
``httpsErrors`` nor ``accept_certs`` is a valid ``chromium.launch()`` kwarg in
Playwright Python (verified against the installed signature) — passing one
raises TypeError. The context-level ``ignore_https_errors`` is the supported
path.

Run:  python scripts/visual_verify.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

BASE = "https://127.0.0.1:8765"
OUT = Path("/tmp/visual_verify_screenshots")
QUERY = "A player wants to grapple an ogre that is drinking a potion. What contested rolls should I call?"


def e2e_probe(timeout_s: float = 60.0) -> dict | None:
    """POST a query, then poll GET /api/cards until a non-error card appears."""
    httpx.post(f"{BASE}/api/query", json={"text": QUERY}, timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            data = httpx.get(f"{BASE}/api/cards", timeout=10).json()
        except Exception:
            time.sleep(2)
            continue
        for c in data.get("cards", []):
            if c.get("kind") != "error":
                return c
        time.sleep(3)
    return None


def capture() -> list[str]:
    """Screenshot the live UI at desktop + mobile viewports."""
    captured: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--ignore-certificate-errors"])
        ctx = browser.new_context(ignore_https_errors=True)
        page = ctx.new_page()
        page.goto(f"{BASE}/", timeout=30000)
        page.wait_for_timeout(2000)
        for name, w, h in [("v1_desktop_1440", 1440, 900), ("v2_mobile_380", 380, 700)]:
            page.set_viewport_size({"width": w, "height": h})
            page.wait_for_timeout(800)
            path = str(OUT / f"{name}.png")
            page.screenshot(path=path)
            captured.append(path)
        browser.close()
    return captured


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True

    print("=== E2E re-probe (post-restart) ===")
    card = e2e_probe(timeout_s=60.0)
    if card is None:
        print("FAIL: e2e — no non-error card produced")
        ok = False
    else:
        print(f"PASS: e2e — card kind={card.get('kind')!r} title={card.get('title')!r}")
        body = card.get("body_md", "") or ""
        if not body.strip():
            print("FAIL: e2e — card body_md empty")
            ok = False
        else:
            print(f"PASS: e2e — body_md non-empty ({len(body)} chars)")

    print("=== Visual capture (chromium) ===")
    for path in capture():
        if Path(path).exists() and Path(path).stat().st_size > 0:
            print(f"PASS: screenshot {path} ({Path(path).stat().st_size} bytes)")
        else:
            print(f"FAIL: screenshot missing or empty {path}")
            ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
