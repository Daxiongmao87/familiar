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
- **Empty scene-context guard.** `dmd/pipeline.py` no longer publishes an
  empty `scene_context` event when an ephemeral-tier agent produces no text
  (e.g. it times out under GPU contention). Before the fix a timed-out
  agent leaked a no-op scene note (`text: ""`). Regression test
  `tests/e2e/test_agentic_pipeline.py::test_empty_agent_output_produces_no_scene_context`
  pins the defect (red before the fix: an empty scene event is emitted;
  green after: none emitted). Patch (pre-1.0).
- **Subprocess teardown fix.** `dmd/tools_reg.py`: `register()` and `call()` previously called `proc.kill()` on timeout then returned, leaving the killed child transport unreaped. At teardown the transport's `__del__` ran against a closed loop and raised `RuntimeError: Event loop is closed`. Adding `await proc.wait()` after `proc.kill()` reaps the killed child while the loop is still alive — the standard asyncio close idiom. Patch (pre-1.0).
- **Browser test environment guard.** `tests/e2e/test_ui_e2e.py` now skips
  the headless-browser test when the host cannot create a named semaphore
  (`/dev/shm` restricted; common in PID-namespace containers) instead of
  failing. The non-browser UI tests (data path) and
  `scripts/verify_live_ui.py` verify the live-window contract (cards +
  transcript over HTTP + WS) without a browser. Test hardening; no behavior
  change. Non-release-affecting.
- **Live UI data-path verification.** `scripts/verify_live_ui.py` replicates
  the e2e stack (mock backend + real engine + real uvicorn app), connects a
  WebSocket subscriber, drives a manual query + transcript events, and
  asserts cards and transcript lines reach the client. Run to confirm the
  live window renders artifacts when a real browser isn't available.
  Verification tooling; non-release-affecting.

### Fixed
- **LLM inference-slot leak (Priority 0 hotfix; production defect,
  2026-09-05).** `dmd/gateway.py` omitted `max_tokens` whenever the caller
  passed `None`, so llama.cpp ran with `n_predict=-1`; a repeating model
  never released its slot, and every 180 s client read-timeout abandoned one
  slot hostage (11 leaked in lockstep this morning; all fleet tiny-model
  slots held). Now every chat request carries a bounded `max_tokens`
  resolved caller arg > endpoint config (`models.*.max_tokens`) > per-role
  default (`ROLE_DEFAULT_MAX_TOKENS`: fast 1024, synthesis 4096, vision
  2048); `dmd/monitor.py` and `dmd/triggers.py` — the two callers that fired
  without a cap — now pass 1024 explicitly, and `dmd/gateway.py` honors a
  per-endpoint `request_timeout_s` so an abandoned call cancels as a typed
  `GatewayError` instead of hanging on the client default. Guarded by
  `tests/unit/test_gateway_slot_guard.py` (red before, green after: body
  always contains `max_tokens` for gateway, trigger-classifier, and
  monitor-judge call sites) plus
  `tests/unit/test_triggers.py::test_detect_trigger_fast_lane_passes_max_tokens`.
  Two silent-`None` defects found in the same audit: `Gateway.transcribe`
  openai-dialect path never returned the transcription (fixed; proven by
  previously-red `tests/unit/test_gateway.py::test_transcribe_multipart_returns_text_field`)
  and `Gateway.stt_health` fell through to `None` on 5xx (fixed). Patch
  (pre-1.0).
- **Intake latency instrumentation (SPEC §14, owner-verified defect).** The
  per-user STT worker queue (from the v2 wip lane) now carries timestamps
  proving the original defect is gone: every utterance logs enqueue /
  worker-start / STT-done monotonic timestamps and publishes an
  `stt_latency` event (`queue_wait_ms`, `stt_ms`, `post_speech_ms`);
  `consume_source` measures per-chunk handler work and warns if a chunk
  ever takes >50 ms in the feed loop (the inline-await signature).
  `SessionEngine.intake_stats()` exposes the counters. Proven by
  `tests/unit/test_pipeline_latency.py` (4 tests: feed span with 0.4 s STT
  calls, cross-speaker independence — fast B finishes before slow A —
  per-user ordering, pool drain). Patch (pre-1.0).
