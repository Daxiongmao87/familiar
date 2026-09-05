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
- **STT request params from config (SPEC §2 zero-hardcoding).** The gateway
  hardcoded `diarize=false&align=false` on whisperx requests, suppressing the
  whisperx-server's own defaults and blocking §7a attribution. `SttRole` gains
  `diarize` (default true — feeds pyannote segments to attribution) and
  `align` (default false — word timestamps nothing consumes);
  `Gateway.transcribe_diarized()` returns `(text, speaker segments)`;
  `config.example.yaml` documents both. Tests:
  `tests/unit/test_gateway.py::test_whisperx_params_reflect_config_diarize_align`,
  `::test_transcribe_diarized_returns_speaker_segments`,
  `::test_stt_health_5xx_reports_unreachable`. Patch (pre-1.0).
- **§7a per-speaker attribution (ABSENT defect closed).** New
  `dmd/attribution.py` implements the owner-ordered join: pyannote diarized
  segment windows (WhisperX `diarize=true`, clip-relative seconds mapped
  through the utterance's monotonic start) x Discord-gateway
  `member_speaking_state_update` windows (`SpeakingTracker`), best-overlap
  wins, 0.15 s minimum, crosstalk grouped into one utterance per speaker
  (`group_by_speaker`), dominant-window fallback without segments
  (`attribute_whole`). Attribution can only refine identity, never invent it
  (no overlap -> source label). `dmd/types.py`: `Utterance` gains `name`;
  `dmd/pipeline.py`: `transcribe_pcm` returns attributed utterances and the
  `transcript` event carries `user_id` + display `name` (browser_mixed is no
  longer the label on live capture); `SessionEngine` accepts
  `speaking_tracker`. `dmd/speaking_tracker.py`: `overlaps_during`,
  display-name registry; `dmd/voice_presence.py` feeds names;
  `dmd/server.py:main` creates the tracker and injects it into the engine.
  §4 per-user identity rides this mechanism (DAVE per-user RTP stays deferred
  by design, §17). Proven by `tests/unit/test_attribution.py` (7) and
  `tests/unit/test_pipeline_attribution.py` (5: named segments, crosstalk,
  dominant fallback, tracker-absent identity passthrough for replay sources).
  Minor (pre-1.0).
- **§11 player-state proof + re-seed HP bug fix.**
  `tests/unit/test_player_state.py` (6) pins seeding from `characters/*.md`
  (frontmatter titles), mark-done and monitor-AI-observed updates writing
  through `record_card_done` (done_cards ref + idempotent, item merge into
  inventory). The tests caught a real defect: `seed()` wrote empty hp into
  `COALESCE(excluded.hp, players.hp)`, clobbering live HP on every re-seed —
  now empty hp arrives as NULL. Patch (pre-1.0).
- **§5 no-baked-rules audit pinned.** `tests/unit/test_no_baked_rules.py`
  statically guards shipped `dmd/` + `web/` sources against dice notation,
  DC/AC constants, and rule-table fragments (the grep audit passed clean;
  the test keeps it clean). Test hardening; non-release-affecting.
- **§15 session controls + §9 card lifecycle UI; §10 scene strip rendered.**
  Backend: `SessionEngine` gains `set_capture_paused` (audio dropped at the
  feed loop; half-open VAD utterances discarded via new
  `UtteranceSegmenter.drop_user` — nothing stitches across the pause) and
  `set_ooc` (transcript keeps flowing — the event log is the sole truth —
  while fast-lane triggers and the proactive monitor go silent); both publish
  `capture_state` / `ooc_state` events so every client stays in sync.
  Endpoints: `POST /api/capture`, `POST /api/ooc`, `POST /api/card/done`
  (mark-done = set aside, never delete), `GET /api/players` (badge names);
  `/api/status` reports `controls` + active `speakers`;
  `_card_to_dict` now carries `status` + `player_ids` (the card event was
  silently dropping both). Frontend: topbar Pause / OOC buttons (state-
  reflected, `body.capture-paused` dims the transcript); player badges on
  cards (display-name resolved, hue-stable, `shared` for unowned); cards
  render collapsed and expand on head click (pre-generated content); Done
  button moves a card into the set-aside `<details>` done area with live
  count; scene-context notes now render in their own strip with fade-and-drop
  decay (they were emitted but never displayed before). Visual verification:
  13 DOM assertions at 1440x900 and 380x800 (E1 named transcript, E2 badges,
  E3 collapsed, E4 expand-on-click + table, E5 done lifecycle incl. server
  state sync, E6 scene note, E7/E7c control round-trip, E8 mobile single
  column) + 7-point pixel audit (A1-A7) of `screenshots/v1…v6` — all pass.
  Captions: screenshots/ is gitignored; the captures themselves were not
  viewable by this agent (no image input) — owner should eyeball them.
  Tests: `tests/unit/test_session_controls.py` (8). Minor (pre-1.0; new
  endpoints/contracts).
- **Live-path defects from the 2026-09-05 restart verification (3 fixes).**
  (a) `stt_health.py` unpacked `Gateway.stt_health()`'s coroutine object
  synchronously: the probe task died at startup and `/api/status` reported
  STT `healthy: false` against a whisperx answering 200 — the monitor now
  awaits the coroutine, the probe loop survives failures, first-failure
  log-backoff no longer subtracts `None` from a float (second latent crash
  caught by the same tests), and the snapshot reports `None` (unknown) until
  the first probe instead of false-dead. Regression:
  `tests/unit/test_stt_health.py` (6), including a contract test against the
  real `Gateway` + MockTransport.
  (b) `orchestration.job_timeout_s` default 20.0 < `agent.agent_timeout_s`
  45.0: the pool silently killed every card-producing agent run on the live
  tiny endpoint (manual query -> no card, no event). Default raised to 60.0
  (documented invariant: > agent budget) and `server._make_pool` now wires
  `on_drop` -> `job_dropped` event so future drops surface instead of
  hanging the DM. Test: `tests/unit/test_session_controls.py::test_pool_drop_is_published_not_silent`.
  (c) WhisperX diarize costs ~2 s/utterance on the real endpoint (measured
  3.6 s with an empty tracker vs the §14 ~2 s ephemeral budget) but is only
  consumed by the §7a join — the pipeline now requests diarized segments
  only when the SpeakingTracker actually has speaking windows; with no
  gateway data the fast path (`transcribe`) runs. Test:
  `tests/unit/test_pipeline_attribution.py::test_diarize_requested_only_when_tracker_has_windows`.
  Patch (pre-1.0).
