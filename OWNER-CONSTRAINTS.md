# OWNER CONSTRAINTS — read before any change (Dax, 2026-09-05, repeatedly enforced)

These override SPEC.md where they differ. Newest wins. Do not "helpfully" revert them.

## The product in one sentence
Familiar is a voice-copilot that sits in the DM's ear during a live session and
hands them short, glanceable, at-the-table answers — loot tables, DCs, rulings
— in seconds, so they never have to stop the game, open a book, or read a
paragraph.

## Time-sensitivity is the core constraint
- Answers must arrive in ~2-4s. The DM is mid-game.
- Model endpoints (HARD): synthesis + fast roles -> http://192.168.0.220:8080/v1
  model `minicpm5-2b` — the RTX 3090 box, which was installed and
  benchmarked as the most performant. NOT local :8081/V100s. NOT the Arc.
- NEVER use qwen 27b (or any big slow model) for familiar generation — it
  violates the time-sensitivity constraint. minicpm5-2b on the 3090 is the call.

## Cards are glanceable, never essays
- < 700 chars body (hard cap enforced in `_normalize_card`).
- Lead with the answer. 3-6 short lines / bullets / small table. A RULING card
  is 1-3 lines: rule, DC/skill, done.
- At most ONE source link. No hedging, no meta-commentary about the prompt.
- UI clamps card bodies to ~6 lines (web/style.css .card-body max-height).

## Rules/ruling cards must be grounded in real source text
- Pre-ground through the bundled SearXNG BEFORE the model decodes: fire one
  web_search, pass the actual rule TEXT from the search results (content/
  snippet), and tell the model to write FROM it — not from memory, not from
  URLs alone.
- SearXNG is a module of this project: `.searxng-src/` (vendored),
  `.venv-searxng/`, `tools/searxng.sh start|stop|status|log`, bound to
  127.0.0.1:8888. No Docker. DDG HTML scrape is only the fallback when the
  instance is down.

## Agent features
- The in-app agents are the built-in WorkerAgent loop (dmd/agent.py) — not an
  external framework. Tools: retrieve, repo_read, web_search, web_fetch,
  run_tool.
- Auto-resolve: monitor may mark a card done only when it names a REAL active
  card id (get_cards is wired) and the verdict persists past resolve_grace_s.
