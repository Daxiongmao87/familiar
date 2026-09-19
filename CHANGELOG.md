# Changelog

## [Unreleased]

- **Structured output for ephemeral note synthesis.** `JevWorker` now
  requests `title`/`subtitle`/`body_md` JSON under a strict schema for
  scene-note tier (card tier already had one) and renders the note
  from the components, falling back to raw text when unparseable.
  Verified live on minicpm5-2b with thinking disabled. Minor (worker
  contract change while pre-1.0).
- **Optional per-call thinking switch on `Gateway.chat`.** New
  `thinking: bool | None` parameter injects
  `chat_template_kwargs.enable_thinking` (the switch llama.cpp honors;
  a top-level flag is ignored). Omitted by default; explicit callers
  override config `extra_body`. Verified live on minicpm5-2b:
  `thinking=False` answers directly, default burns all tokens on
  reasoning. Minor (additive API while pre-1.0).
- **Replace directed-worker retrieval with evidence-judged path.** New
  `dmd/terms.py`: zero-LLM term collection (RAKE phrases fused with
  lexicon matched spans, mass cutoff). `JevWorker` now collects terms,
  ranks them in one JEV pass, searches both legs per term (campaign RAG
  + web), judges the retrieved evidence with a new JEV relevance verb
  (offline/online/both), and synthesizes from the kept legs. Removed the
  route-first branch picker, LLM-written per-round queries, and the
  sufficiency loop (`route()`, `sufficient()`, `ROUTE_*`, `SUFF_*`,
  `max_rounds`). No presumed locations: relevance is judged after the
  fact every trigger. Tests rewritten (`test_jevworker.py`,
  `test_terms.py`, relevance/rank in `test_openjev.py`); live run
  verified end to end. Minor (worker contract change while pre-1.0).
- **Lock JEV gate prompt to production baseline jev-gate-v39.**
  `dmd/openjev.py`: deploy/wait question and option descriptions set to the
  validated "useful information work" wording (deploy first, wait second;
  IDs stable; threshold default stays 0.50; state construction, debounce,
  and per-verdict p_deploy publishing unchanged). New
  `tests/regression/golden/jev_gate_v39.json` + live
  `tests/regression/test_jev_gate_golden.py`: all nine golden cases must
  classify correctly with separation gap ≥ 0.5 before any prompt, ordering,
  transcript, model/quant, or readout change is accepted. Verified live:
  9/9, gap 0.7854 reproducing the baseline. Minor (gate contract change
  while pre-1.0).

- Baseline commit: dmd/ package, web/ UI, test suite (unit/regression/e2e),
  SPEC.md, and tooling. Pre-1.0 foundation; no version tag yet.
  Non-release-affecting (initial import of existing work).
- Adopt project-bootstrap doctrine: created AGENTS.md (repo-root
  Repository Guidelines) and .omp/RULES.md (Hard Rules) from the canonical
  project-bootstrap skill text. Non-release-affecting (project governance
  files; no product behavior or contract change).

### Added
- **CR 3-minute timed replay fixture.**
  `tests/regression/golden/cr2e2_3h14m29s_crownsguard.json`: 45
  timestamped, speaker-labeled transcript events (C2E2 VOD 3:14:29–3:17:29,
  crownsguard confrontation: skill checks, advantage/help, attack rolls,
  damage) in `replay_events.json` shape for timed live-play replay through
  detection, retrieval, and card publishing. Patch (test-only, no behavior
  change).
- **Openjev trigger gate + timed transcript replay.**
  New `dmd/openjev.py`: binary deploy/wait gate over openjev-serve
  `/score` (no taxonomy at the gate; a second pass labels kind only on
  deploy), fail-closed to wait, behind `openjev.enabled` (default off —
  legacy path byte-identical). `SessionEngine` uses it in
  `handle_utterance` when enabled and adds an `openjev` detail key to
  `turn_latency` events only in that mode. New
  `dmd/sources/transcript_replay.py` + `tools/replay_transcript.py`:
  wall-clock-paced fixture injection into the real pipeline with model
  overrides as CLI flags. Tests: `tests/unit/test_openjev.py` (9:
  deploy/kind, threshold boundary, fail-closed transport/shape/timeout,
  window cap, health) and `tests/unit/test_transcript_replay.py` (6:
  load/validate, pacing, catch-up, fast mode). Live-verified: 45-entry
  CR fixture replays in 179.4s wall, 10 deploys, 8 cards, avg gate
  644ms. Minor (pre-1.0).
- **JEV-routed deterministic worker + debounce + verdict logging.**
  New `dmd/jevworker.py` (`JevWorker`, behind `openjev.directed_worker`):
  passive RAG, JEV route (offline/online/both, failover both), bounded
  retrieve/judge rounds per branch (LLM terms, system executes, JEV
  sufficiency, failover more), one synthesis call; synthesis role on
  every tier. `Debouncer` (per-kind, `debounce_s` default 30s) in
  `dmd/openjev.py`; `trigger_verdict` events for every gate evaluation
  (waits included) for recall tuning. Harness gains `--directed` and
  `--max-concurrent`. Tests: route/sufficiency/debouncer cases plus
  `tests/unit/test_jevworker.py` (6: offline/both routing, round loop
  and bound, terms fallback, ephemeral text). Live-verified at
  concurrency 1: 45 entries, 4 deploys, 6 debounced, 2 cards, 0
  timeouts — but 31–40s per card (sufficiency too strict, rounds always
  max out) and ruling bodies still invent DCs. Minor (pre-1.0).
- **Trigger taxonomy removed (loot/lore/rules/other).**
  `detect_trigger` returns bool; gate kind stage replaced by a JEV
  card/ephemeral tier verdict on deploy (failover card); `Debouncer` is
  a global window; `_tier_for_kind` and `AgentConfig.card_kinds` gone
  (legacy triggers default to card tier, recall bias); `_task_for_ctx`
  is generic evidence-shaped text; agent pre-grounding fires for every
  card; `turn_latency`/`trigger_verdict` events drop kind; web Card
  kinds setting removed. `Card.kind` (synthesis-chosen output label)
  and `Job.kind` are untouched. Tests rewritten (`test_triggers`,
  `test_openjev` tier/debouncer, `test_agent_grounding` always-ground,
  e2e lore→card with hermetic search stub and drain-before-close);
  golden file surgically updated (kind key dropped, loot tool_calls
  0→1) with the two pre-existing golden reds unchanged in signature.
  `config.yaml`'s `card_kinds` key is now ignored (left in place).
  Live-verified: 45 entries → 2 deploys (both ephemeral), 8 debounced,
  0 cards, 2 scene notes, 0 errors. Minor (pre-1.0).
- **JEV lab form (`tools/jevlab/`).** Single-page form (state, question,
  2–16 options, one-click presets for the two recorded JEV misses) scoring
  through a stdlib same-origin proxy to the local `:8199` endpoint
  (no CORS there; no WebLLM/WebGPU needed). Ships as a systemd unit
  (`jevlab.service`, port 8093). Patch (dev tooling, no product change).
- **Predictive-retrieval staging ("Predictive RAG", Priority-1 deliverable).**
  Anticipate instead of react: the transcript monitor's judge now also emits
  `situation` / `likely_next_events` / `predicted_entities` (fast role, capped
  max_tokens), and every tick that carries predictions forwards them to the
  engine even when `action='none'`. The engine prefetches each predicted
  entity's campaign excerpts into a bounded RAM LRU (`dmd/staging.py`
  `StagedContext`: TTL + `max_entries`, adopted and completed from the
  interrupted prior run) on tracked, throttled background tasks (≤2 in
  flight; never blocks monitor cadence or the answer path). When a real turn
  arrives, `SessionEngine._generate_card` looks the trigger's mentioned
  entities up against the cache (a pure in-memory read) and injects the hits
  as an advisory `PRE-STAGED CONTEXT` block into the worker agent's task
  message; a miss is a byte-identical no-op and never touches the embedder or
  the store on the answer path. The card records `meta.staged_for` when a
  staged hit fed it; every prediction is published as a `staging_predict`
  event and prefetch/inject decisions are logged so precision is tunable.
  Config: `StagingConfig` (enabled/ttl/max_entries/prefetch_k/max_predicted/
  max_inject_chars) + `config.example.yaml`. Live-verified wiring detail: the
  prediction fields are **required** in the judge schema — probed against the
  live fast role, a tiny model skips optional fields entirely (returns only
  `action`) while schema-required fields are endpoint-enforced and always
  emitted; optional predictions would mean staging never fires. Tests:
  `tests/unit/test_staging.py` (11: cache LRU/TTL/counters/lookup-tagging,
  block caps, monitor `on_predict` on `action='none'`, prediction →
  prefetch → injection round trip with `staged_for` recorded, cache-miss
  provably non-blocking, and the required-fields schema guard). Minor
  (pre-1.0).
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
- **VAD endpoint hangover tuned 700→500 ms (Priority-1 latency).** The
  voice→transcript budget starts at *speech-stop*, so the trailing-silence
  hangover is spent before STT even begins. `SttPipelineConfig.silence_ms`
  default drops to 500 ms (the brief's 400–600 ms window); `config.example.yaml`
  and the live `config.yaml` follow. This is a *value* of an already-configurable
  knob, not an endpoint swap. Trims 0.2 s off every utterance's endpointing;
  measured long-utterance voice→transcript 4.64 s (≤5 s). Default-value change
  pinned in `tests/unit/test_config.py::test_defaults_stt_pipeline_and_project`.
  Patch (pre-1.0).
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
- **Streaming-dialect batch safety-net routes to whisperx (`98dd875` follow-on).**
  The `streaming` STT dialect keeps the whisperx HTTP server (:8123) as the batch
  safety net, but only its `/transcribe` + `/health` routes exist there (the
  streaming server on :43007 is raw TCP, no HTTP routes). `dmd/gateway.py`
  `transcribe` and `stt_health` now treat
  `dialect in ("whisperx", "streaming")` identically, so the streaming
  live-path's batch fallback hits the real whisperx route instead of the OpenAI
  `/v1/audio/transcriptions` one. Tests: `tests/unit/test_gateway.py` (12,
  green). Patch (pre-1.0).
- **Cards never surfaced in the live UI (owner-verified defect, 2026-09-05).**
  Three compounding causes, all fixed:
  (1) **Collapsed-by-default presentation** — `web/app.js:buildCard` added
  `collapsed` to every *new* card and `.card.collapsed .card-body` is
  `display:none`, so freshly generated cards arrived hidden/unreadable,
  contradicting SPEC §1/§2/§8/§9 ("read, expand if long, mark done").
  Now a fresh card renders **expanded with title + body visible**; only a
  Done (set-aside) card is collapsed. The Done transition also stopped
  double-binding click toggles (title click now expands a done card once,
  not twice) and now opens the set-aside done area so a card moving to done
  doesn't silently vanish into a closed `<details>`.
  (2) **No REST view / no replay** — cards only existed as live WS pushes;
  there was no `GET /api/cards` (the route 404'd) and a freshly-loaded or
  reconnected DM window started empty forever. Added `GET /api/cards`
  (the REST view of the same `engine._active_cards` store mark-done reads,
  newest-first, active + done) and the UI now fetches it on WS connect and
  renders through the same `addCard` path, id-deduplicated against live
  pushes (fixes a double-render on active cards too).
  Verified live on :8760 (HTTPS): a real manual query produced a
  `skill_table` card that reached a `/ws` subscriber in 12.5 s **and**
  `GET /api/cards`; DOM audit at 1280 px and 380 px confirms fresh cards
  expanded/readable, done cards collapsed + is-done in the open done area,
  and done-title click re-expands (`screenshots/d1-d4_*`, ephemeral).
  Minor (pre-1.0; new additive REST endpoint + client replay for a live
  defect).

- **Fast-lane classifier dominated the transcript→answer budget (live defect,
  Priority-1 measurement, 2026-09-05).** `detect_trigger` asked the fast role
  (ling-3.0-tiny) for every utterance *before* the regex fallback. That
  endpoint is reasoning-first: a bare classification request emits ~180 hidden
  reasoning tokens and takes **17–24 s** (measured: `turn_latency`
  `detect_ms` = 23854 ms on the live baseline, and a direct timed POST to the
  fast role reproducing 17.9 s), and it still misfired — an unambiguous "we
  loot … body" came back `is_trigger: false`. So the voice path produced **no
  answer at all** for 2 of 3 probe utterances (transcript→answer = `-1`).
  Precedence is now deterministic-first: the keyword regex short-circuits
  instantly (no LLM, no network) for the search/loot/examine intents that
  dominate the fast lane, and the LLM is only a *bounded* tie-breaker for prose
  the regex can't see — `asyncio.wait_for` over a hard
  `LANE_CLASSIFY_TIMEOUT_S` (3 s) with `LANE_CLASSIFY_MAX_TOKENS` (96), falling
  back to the regex verdict on timeout/empty/misparse. Measured on the live
  service: `detect_ms` 23854 ms → **0.0 ms**; transcript→answer `-1/-1/5.3+` →
  **9.99 / 7.34 / 4.66 s** (all ≤15 s), and all three probe utterances now fire
  a grounded card. Tests: `tests/unit/test_triggers.py` (regex short-circuits
  without touching the LLM; a slow LLM is hard-bounded and falls back; malformed
  result falls back) and the retargeted
  `tests/unit/test_gateway_slot_guard.py::test_trigger_classifier_call_sends_max_tokens`;
  golden replay `detect_ms`/`lane_ms` normalized as wall-clock noise. Patch
  (pre-1.0).
- **`/api/init` was dead on arrival (live defect, 2026-09-05).**
  `_make_init_runner` in `dmd/server.py` called
  `dmd.init_pass.run_init(project_path=…, gw=…)` with nonexistent keywords
  (`path=`, `gateway=`, plus a `lexicon_entries=` parameter the function
  never had), so every live init died as a `TypeError` inside an unwatched
  background task — the running service had never indexed its campaign
  (0 docs, 0 entities, empty lexicon). Also: the async `progress_cb` was
  called synchronously by `_cb`, so no `init_progress` event ever reached
  the bus. Now the runner matches the real signature, progress is published
  through `EventBus.publish_sync`, the completion event carries the counts,
  failures log a traceback, and a successful init hot-swaps the live
  engine's lexicon via `SessionEngine.refresh_lexicon` (re-init takes effect
  without a restart). Proven by
  `tests/unit/test_server_init_wiring.py` (4 tests, red against the old
  wiring) and live: docs 7 / entities 15 now indexed on :8760. Patch
  (pre-1.0).
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
- **manual_query / trigger / monitor job submission no longer blocks its
  caller.** `SessionEngine._submit_fire` schedules pool jobs as detached
  tasks (with an entered-handshake so `pool.drain()` never races an
  un-queued submit). Previously `POST /api/query` awaited the pool future,
  so the HTTP request blocked for the whole agent run and the client hit a
  read timeout while the card eventually arrived on the bus; and a trigger
  found in one utterance delayed that user's next utterance by the full job.
  Cards arrive via `card` events regardless. Regression:
  `tests/unit/test_session_controls.py::test_manual_query_does_not_block_on_agent_run`.
  The replay golden re-ordered accordingly (transcript #2 now precedes the
  first card — reviewed diff, pure async-arrival reordering). Patch (pre-1.0).
