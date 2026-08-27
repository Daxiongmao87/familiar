# Changelog

## [Unreleased]

- Baseline commit: dmd/ package, web/ UI, test suite (unit/regression/e2e),
  SPEC.md, and tooling. Pre-1.0 foundation; no version tag yet.
  Non-release-affecting (initial import of existing work).
- Adopt project-bootstrap doctrine: created AGENTS.md (repo-root
  Repository Guidelines) and .omp/RULES.md (Hard Rules) from the canonical
  project-bootstrap skill text. Non-release-affecting (project governance
  files; no product behavior or contract change).

### Added
- **v2 agentic live path.** The live path is now workers that *do work* as it
  comes in, instead of a fixed embed→retrieve→synthesize pipeline:
  - `dmd/agent.py` — `WorkerAgent`: a bounded tool-looping agent (retrieve /
    repo_read / web_search / web_fetch / run_tool) that produces grounded cards
    and ephemeral scene notes from the world map, transcript, and tool results.
  - `dmd/monitor.py` — `TranscriptMonitor`: a cadenced, proactive monitor that
    reads the rolling transcript and surfaces scene notes or auto-marks cards
    done (never deletes).
  - `dmd/player_state.py` — per-player state store (SQLite; seeded from character
    sheets, updated on card-done) so the assistant knows who already has/knows what.
  - `dmd/world_map.py` — `build_world_map`: the init artifact that orients live
    agents (players, locations, NPC inventory, probed tools).
  - `dmd/pipeline.py` — `SessionEngine` routes triggers + manual queries through
    the agent with two tiers: durable kinds (`agent.card_kinds`, default
    loot/rules) → cards; the rest → ephemeral scene context. The fast lane
    (`detect_trigger`) picks the tier; monitor + fast classifier use the optional
    `fast` role.
- E2E test `tests/e2e/test_agentic_pipeline.py`: replays a scripted Discord
  session through the full pipeline (fast lane → tier routing → agent) and asserts
  grounded cards for loot triggers and grounded scene context for lore triggers.
- **Live settings editor.** Edit and save `config.yaml` from the running UI:
  - `dmd/server.py` — `GET /api/config` returns the effective config (pydantic
    defaults applied) with secrets masked; `POST /api/config` deep-merges the
    submitted config, keeps masked secrets, and reports which top-level sections
    changed (restart hint).
  - `web/` — settings modal (gear button): Campaign, Discord, Models
    (synthesis / fast / STT / embeddings), Agent, Orchestration, and STT-pipeline
    sections. Load populates effective values; blank secret fields keep the
    current value on save.
  - Residual risk: a save rewrites `config.yaml` via `yaml.safe_dump`, which
    strips any hand-written comments in the file. Values are preserved; comments
    are not. Non-release-affecting (pre-1.0; no version tag yet).

### Changed
- `dmd/config.py`: `AgentConfig` (tool-call budgets, `card_kinds`, cadence).
- `dmd/server.py`: builds per-player state + world map for the engine; starts
  and stops the transcript monitor on app start/stop.
- `dmd/types.py`: `Card` gains `status` (active/done) and `player_ids`.
- `tests/e2e/mock_server.py`: the mock's fast role now answers both the
  fast-lane classifier (loot/lore/other) and the agent's ephemeral-tier call.
- Regenerated `tests/regression/golden/replay_events.json` to the agentic card
  shape. Non-release-affecting (the v2 live path is pre-1.0; no version tag yet).
- `dmd/agent.py`: `WorkerAgent._task_message` now caps the rolling
  transcript to a recent 4000-char window instead of re-sending the whole
  buffer on every loop call. Prevents unbounded context growth (and per-call
  latency) over long sessions; the triggering utterance and world map are
  still sent in full. Verified live: grounded loot card in 9.1s through the
  full pipeline when the endpoint is responsive. Non-release-affecting
  (pre-1.0; no version tag yet).
