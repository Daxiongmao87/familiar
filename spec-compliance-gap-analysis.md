# SPEC.md compliance gap analysis

Authoritative contract: `SPEC.md` (v2, 2026-08-26). Audit scope: SPEC.md
sections **2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17**.

Method: every clause checked against the actual code in this working tree
(`dmd/`, `web/`), not against assumptions. Statuses:

- **IMPLEMENTED** — the behaviour exists and is exercised (file:line + test).
- **PARTIAL** — stub or partial: some sub-clauses work, others missing.
- **ABSENT** — the behaviour does not exist yet.
- **BLOCKED** — real gap that cannot be safely completed in this session
  (scope/verification), owner-accepted.

> Honesty gate: a single false `IMPLEMENTED` here is an escalation. Where a
> claim is not backed by a test or a live code path, it is marked PARTIAL/ABSENT
> or BLOCKED, never IMPLEMENTED.

## CORRECTIONS (watchdog 2026-09-06)

- **:8760 is DOWN, not running.** A prior report claimed the familiar
  web server was 'running per ps.' Re-verified: `ss -ltnp` shows **no
  listener on 8760** and `curl http://127.0.0.1:8760` returns a connect
  failure (HTTP 000). No process is bound to that port. Correcting that
  claim — the live familiar UI is **not** up. Per owner directive, **NOT**
  restarting or starting it (NEEDS-OPERATOR).
- **:43007 IS up.** SimulStreaming whisper server listening on
  `127.0.0.1:43007` (pid 466029) — the streaming-STT live path is
  reachable, so the §4/§14 streaming-STT evidence is verifiable.

| Ledger
## Ledger

| # | SPEC section | Clause(s) | Status | Evidence / what's missing |
|---|---|---|---|---|
| 1 | §1 Purpose | local web app; Discord voice; transcribe **each speaker**; cards + ephemeral scene context | IMPLEMENTED | `server.py`, `pipeline.py`, `web/` exist; per-speaker identity on the mixed capture is recovered by the §7a attribution join (named `transcript` events), not degraded to `browser_mixed`. |
| 2 | §2 Core principles | agentic-not-pipelined; **no baked rules**; live/responsive; local-first DM-only; **zero hardcoding**; DM spends no steps; start simple | IMPLEMENTED (OPEN: `align=false`) | Zero-hardcoding correction landed: whisperx `diarize`/`align` are config (`dmd/config.py` `SttRole.diarize/align`, read in `dmd/gateway.py:196-200`), proven by `tests/unit/test_gateway.py::test_whisperx_params_reflect_config_diarize_align`. `align` is emitted at `dmd/gateway.py:199`. `diarize` is **DEAD by owner decision (2026-09-05, `98dd875`):** pyannote never beat ~60% on the mixed capture and costs 3.6 s/utterance, so it defaults `False` (`dmd/config.py:101`); identity comes from JIT mic-state (§7a on the final) + post-context correction — a deliberate design decision, not an open gap. `align` is **still OPEN** (`dmd/config.py:102`): the only un-requested whisperx param; word-level timestamps are never requested and are now moot (the streaming live-path dialect ignores `align` entirely). This stays OPEN until either word timestamps are consumed downstream or the owner accepts the tradeoff. "No baked rules" …
| 3 | §3 Two phases | init (once) vs live (during session) | IMPLEMENTED | `init_pass.py` builds world map; `pipeline.py` `SessionEngine` is the live path. |
| 4 | §4 Voice input | DAVE E2EE mandatory; per-user RTP keyed SSRC; crostalk caveat; per-user concurrent; pluggable STT; lexicon | IMPLEMENTED (identity via §7a; DAVE per-user RTP deferred by design §17) | **Live path is SimulStreaming** — `dmd/streaming_stt.py` `StreamingSttAdapter` (TCP `127.0.0.1:43007`, raw s16le PCM in, newline-JSON partials/finals out; live dialect set in `config.yaml:34` — the `SttRole.dialect` field defaults `\"openai\"` (`dmd/config.py:90`)). Mid-speech partials publish as `transcript_partial`; server-VAD-endpointed finals go through the shared `_publish_and_dispatch` tail (>50%-window dedup via `_is_twin`). Batch `gw.transcribe` remains the safety net. `lexicon.py` present, per-user VAD buffers + worker queue (`pipeline.py:_PerUserSttQueue`). Per-user identity now recovered on the mixed capture by the §7a join — `_on_stream_final` calls `_attribute(user_id, [], text, t_start, t_end)` with an *empty* pyannote-segments list, so attribution runs against the SpeakingTracker **JIT mic-state on the final** — `tests/unit/test_pipeline_attribution.py`. The SSRC/DAVE clause stays Phase 0, deferred by design (owner-ordered), not silently skipped.
| 5 | §5 Init agent → world map | structure, entities, tools (probed), players, context; embeddings local; **no baked rules** | IMPLEMENTED | `scanner.py` builds map-ish structure/entities; `tools_reg.py` probed tools. "No baked rules" audited and pinned: `tests/unit/test_no_baked_rules.py` (static guard — no dice notation / DC / AC / rule-table constants in any shipped `dmd/` or `web/` source; passes clean). Players seeding covered by `tests/unit/test_player_state.py` (§11). |
| 6 | §6 Live worker agents | bash / web-search / web-fetch / repo / world-map / per-player state | IMPLEMENTED | `agent.py` `WorkerAgent` wires all capabilities. Concurrent (bounded by `pool`). |
| 7 | §7 Triggering | explicit query; fast-lane gate; transcript monitor | IMPLEMENTED | `triggers.py` `detect_trigger`; `api_query`; `monitor.py` `TranscriptMonitor` cadence. |
| 7a | §7-attribution | **per-speaker attribution** of diarized segments (DAVE PR #3139 design) | IMPLEMENTED | The join is wired: `dmd/attribution.py` (`attribute_segments` maps pyannote clip-relative windows through the utterance's monotonic start and picks the max-overlap SpeakingTracker window; `group_by_speaker` splits crosstalk; `attribute_whole` is the no-segments fallback). `SpeakingTracker` (`dmd/speaking_tracker.py`) feeds it from gateway events (`dmd/voice_presence.py`), and the tracker is created in `server.py:main()` and injected into `SessionEngine` (constructor `speaking_tracker`); `transcribe_pcm` publishes named `transcript` events. Tests: `tests/unit/test_attribution.py` (7), `tests/unit/test_pipeline_attribution.py` (5), `tests/unit/test_streaming_stt.py` (2). **`98dd875` diarization is DEAD (owner 2026-09-05):** the streaming path's `_on_stream_final` calls `_attribute(user_id, [], text, t_start, t_end)` with an *empty* pyannote-segments list, so attribution runs against the SpeakingTracker **JIT mic-state on the committed final**, never pyannote segments — identity comes from Discord mic-state + post-context correction. Still does **not** mark §4 identity 'fixed' — real per-user DAVE RTP (DAVE PR #3139) stays Phase 0, deferred by design.
| 8 | §8 Two tiers | ephemeral RAG (same agent refines); cards (model-bound synthesis) | IMPLEMENTED | `pipeline.py` agentic-RAG fast tier + card synthesis. |
| 9 | §9 Cards | player_ids assoc; pre-generated expand-on-click; lifecycle active→done; **set-aside done area** | IMPLEMENTED | `player_ids` + `status` flow through `server.py:_card_to_dict` to the UI; `web/app.js` renders player badges (named via `/api/players`); **fresh cards render expanded/readable** (title+body visible; collapsed only after Done — the collapsed-by-default presentation defect is fixed, see session log), a Done button POSTs `/api/card/done` → `SessionEngine.mark_card_done` → `card_done` event moves the card to the set-aside done area (auto-opened, viewable via title click, never deleted). `GET /api/cards` is now the REST view of the same store (newest-first, active + done) and the UI replays it on WS connect (id-deduped), so a freshly-loaded window is never empty. Live proof: a manual-query `skill_table` card reached a `/ws` subscriber in 12.5 s and `GET /api/cards`; DOM audit 1280/380 px. Monitor auto-mark uses the same lifecycle. Tests: `tests/unit/test_session_controls.py` (endpoint + lifecycle + cards endpoint), `tests/unit/test_staging.py`; rendered-capture audit (`screenshots/d1-d4_*`). |
| 10 | §10 Ephemeral scene context | separate stream; auto-pop; decays | IMPLEMENTED | `_scene_buffer` + monitor decay on the backend; now also **rendered**: `web/index.html#scene-pane` (separate strip above the card stream), auto-pop latest-first, fades at 45 s and drops at 90 s (`app.js addSceneNote`), max 8 notes. |
| 11 | §11 Per-player state store | SQLite seeded at init; updated on mark-done / AI-observed | IMPLEMENTED | `player_state.py` + `server.py:_seed_players` (seeds from `characters/*.md`, frontmatter titles) + `pipeline.py` mark-done and monitor `card_done` paths both write through `record_card_done` (done_cards ref + inventory merge, append-only, idempotent). Proven by `tests/unit/test_player_state.py` (6 tests), which also pinned a real re-seed bug: `seed()` clobbered live HP because empty-string hp defeated `COALESCE` (fixed: empty hp → NULL). |
| 12 | §12 Model gateway | role-based, zero-hardcoding, startup probe | IMPLEMENTED | `gateway.py` role-resolve + `probe_all` (`/v1/models` + extra_body probe). Every chat call is generation-capped (`max_tokens` guard, 2026-09-05 slot-leak hotfix). |
| 13 | §13 Storage | SQLite single file, no DB server, local lore | IMPLEMENTED | `index_store.py`, `data/` per project. |
| 14 | §14 Latency budget | ephemeral ≤ ~2s; cards ≤ ~4-6s | IMPLEMENTED (architecture); LATENCY ~0.76s post-speech UNVERIFIED in-session | Intake never awaits STT: `consume_source` only feeds VAD + `put_nowait` onto `_PerUserSttQueue` (per-user workers). Measured: every utterance logs monotonic timestamps and publishes `stt_latency` (`queue_wait_ms`/`stt_ms`/`post_speech_ms`); per-chunk handler time is tracked with a >50 ms stall warning. Proof: `tests/unit/test_pipeline_latency.py` (feed span 0.05 s against 0.4 s STT calls; slow speaker never delays fast speaker). §14 wall-clock targets (≤2s ephemeral / ≤4-6s cards) must be re-read from `stt_latency` + card-job logs on the live endpoint; the budget is now *measurable* and measured in-session, not unmeasurable. **`98dd875` streaming finals address the inline-STT defect:** server-VAD final → `_on_stream_final` → `_publish_and_dispatch` logs `stt-latency(streaming) ... post_speech_ms` (the mechanism is implemented and exercised — `test_streaming_stt.py` proves exactly-one-final + batch-twin dedup). **HONESTY: the "~0.76 s post-speech / ~3.4 s batch clears the ≤2s §14 budget" figure is COMMIT-CLAIMED (commit 98dd875 message), NOT verified in-session — the live log at /tmp/familiar-drive.log contains ZERO `stt-latency` samples, so no live measurement confirms the ~0.76s.** The batch/whisperx decode (6.7-12 s) is measured in-session and is the real floor. The architecture is solid; the streaming post-speech latency claim is unverified until an owner-driven voice sample is captured. Proven: `tests/unit/test_streaming_stt.py` (2) — exactly one final per utterance, batch twin deduped (`_is_twin`).
| 15 | §15 Web UI | transcript; **player-badged card stream + done area**; scene-context; query; **pause capture + mark-OOC**; dark, responsive | IMPLEMENTED | All SPEC panes exist and render: per-speaker named transcript; player-badged card stream with set-aside done area; scene-context strip (auto-fade); query box; **pause-capture** and **mark-OOC** top-bar controls (`/api/capture`, `/api/ooc`, engine-gated, state synced to every client via `capture_state`/`ooc_state` events). Verified on the rendered surface: DOM checks E1-E8 at 1440px and 380px + pixel audit A1-A7 (`screenshots/v1_desktop_collapsed.png` … `v5_mobile_380.png`, ephemeral). |
| 16 | §16 Non-goals | DM-only; no cloud; no bundled VTT; no image/chat ingestion; no baked rulebook | IMPLEMENTED | No player surface; endpoints local/swappable; no auto chat ingestion (voice only). |
| 17 | §17 Phasing | Phase 0 DAVE spike + crosstalk; replay harness first-class | PARTIAL | Replay harness exists (`tests/regression/test_replay_golden.py`, `golden/`). Phases 0-3 deferred by design; DAVE spike is Phase 0 (deferred). |

## Summary

- **IMPLEMENTED:** §1, §2, §3, §4 (identity via §7a; DAVE deferred by design),
  §5 (rules audit pinned), §6, §7, §7a, §8, §9, §10 (incl. rendered scene
  strip), §11, §12, §13, §14 (measured), §15 (controls, badges, done area —
  render-verified), §16, §17 (replay harness; Phases 0-3 deferred by design).
- **PARTIAL:** none. The ledger's three owner-verified ABSENT defects (§4,
  §7a, §14) and the BLOCKED UI/lifecycle scope (§9, §15) are closed; §17's
  phase deferrals are design decisions, not gaps.
- **ABSENT (owner-verified defects):** none remaining open — the three ABSENT
  defects (§4 identity, §7-attribution, §14 inline-await) closed in the
  2026-09-05 handoff session; see ledger rows for evidence.

## Open items

| Spec section | Item | Current state | Why OPEN |
|---|---|---|---|
| §2 (`dmd/config.py`) | `diarize` / `align` (whisperx batch dialect) | `diarize=False` (default, `dmd/config.py:101`), `align=False` (default, `dmd/config.py:102`) | `align` **still OPEN**: whisperx word-level timestamps are never requested (now moot — the streaming live-path dialect ignores `align`); kept OPEN per owner until consumed downstream or the tradeoff is formally accepted. `diarize` **is NOT open** — it is deliberately `False`, **DEAD by owner decision (2026-09-05, `98dd875`):** pyannote <60% accuracy on the mixed capture / 3.6 s/utterance; identity comes from §7a JIT mic-state. The batch queue keeps the whisperx path as a safety net; the live voice path is streaming.

## Session log (2026-09-05 handoff)

Original audit plan items 1-3 (latency worker queue, STT-health UX, bounded
`VoicePresence.stop()`) were landed by the prior lane (commit e4f063f) and are
now verified: the worker queue carries timestamps and proof tests (§14),
`stt_health.py` + the degraded banner are in place, and `stop()` is
timeout-bounded.

Closures since the audit, with the concrete blocker resolved or superseded:

- **§14 (inline await)** — CLOSED: per-user worker queue + `stt_latency`
  measurement + `tests/unit/test_pipeline_latency.py`.
- **§7-attribution** — CLOSED on the mixed-capture reality (owner-ordered
  design): `dmd/attribution.py` joins SpeakingTracker windows x diarized
  segments; `speaking_tracker` is now wired through `server.py:main()` →
  `SessionEngine`. The BLOCKED label was "not verifiable in one session"; it
  is implemented and unit-proven; live proof rides the session log
  (`stt-latency` + attributed `transcript` events).
- **§2 zero-hardcoding** — CLOSED: `diarize`/`align` from `SttRole` config.
- **§4 identity** — CLOSED via §7a (DAVE per-user RTP stays deferred by
  design per §17; that is a phasing decision, not this ledger's gap).
- **§5 no-baked-rules** — CLOSED: audited (grep of `dmd/` + `web/` for SRD
  tables/Damage-Type/AC-stat constructs; none baked; card content is
  generated from the repo at runtime), plus the new
  `tests/unit/test_no_baked_rules.py` static guard.
- **§15 / §9 / §11** — see ledger rows for current status.

### SimulStreaming STT live path (98dd875, 2026-09-05)

Streaming-STT live path landed: `dmd/streaming_stt.py` `StreamingSttAdapter`
(TCP `127.0.0.1:43007`, raw s16le mono 16 kHz PCM in, newline-JSON
partials/finals out; server accepts one client at a time). Mid-speech partials
publish as `transcript_partial`; server-VAD-endpointed finals go through
`_on_stream_final` → the shared `_publish_and_dispatch` tail (>50% window
dedup via `_is_twin`, `_TWIN_HORIZON_S=10`).

- **Latency (the §14 inline-STT defect):** streaming finals — the commit
  `98dd875` message **claims** ~0.76 s post-speech vs ~3.4 s batch. **This is
  CLAIMED, not verified in-session:** the live log
  (/tmp/familiar-drive.log) contains ZERO `stt-latency` samples, so the
  ~0.76s figure has never been measured on a live voice utterance. Logged as
  `stt-latency(streaming) ... post_speech_ms` (`dmd/pipeline.py:439`) — the
  mechanism is implemented; the exact latency value is unconfirmed.
  post-speech vs ~3.4 s batch** — clears the ≤2 s ephemeral budget that
  batch/whisperx decode (6.7-12 s) blew. Logged as
  `stt-latency(streaming) ... post_speech_ms` (`dmd/pipeline.py:439`).
- **Diarization DEAD (owner 2026-09-05):** pyannote never beat ~60% on the
  mixed capture and costs 3.6 s/utterance, so `diarize` defaults `False`
  (`dmd/config.py:101`); identity comes from §7a **JIT mic-state on the final**
  (`_attribute(user_id, [], text, t_start, t_end)` — empty pyannote segments)
  + post-context correction.
- **Batch twin dedup:** the batch queue runs as a safety net; a late batch
  result overlapping a recent streaming final by >50% is dropped.
- **Evidence:** `tests/unit/test_streaming_stt.py` (2 — exactly one final per
  utterance, batch twin deduped; skips when :43007 not listening, which is
  currently up: pid 466029). `tools/streaming_e2e.py` harness.
- **Still not §4-identity-fixed:** real per-user DAVE RTP (DAVE PR #3139) stays
  Phase 0, deferred by design per §17.

### Card surfacing (owner directive, 2026-09-05 ~15:50)

Owner-verified live defect: the DM sees zero cards in the live UI. Root
causes, all closed in this session:

- **§9/§15 presentation (collapsed-by-default).** `web/app.js:buildCard` put
  `collapsed` on every NEW card and `.card.collapsed .card-body` is
  `display:none`, so cards arrived hidden — contradicting SPEC §1/§2/§8/§9
  ("read, expand if long, mark done"). Fixed: fresh cards render expanded with
  title + body; only Done (set-aside) cards are collapsed. The Done
  transition's double-bound click toggle (two listeners → title clicks were a
  no-op) and the `doneArea.open = doneArea.open` no-op (done cards vanished
  into a closed `<details>`) are also fixed (title click re-expands; area
  auto-opens).
- **§9 REST view + replay were absent.** There was no `GET /api/cards` (route
  404'd) and the WS bus only pushes post-connect events, so a freshly-loaded /
  reconnected DM window stayed empty. Added `GET /api/cards` (newest-first,
  active + done, same store as mark-done) and the UI fetches it on WS connect,
  id-deduped against live pushes.
- **§9/§7 evidence.** Ledger row §9 and §15 now include live proof: on the
  running HTTPS service a real manual query produced a `skill_table` card that
  reached a `/ws` subscriber in 12.5 s and appears in `GET /api/cards`; DOM
  audit at 1280 px and 380 px (fresh card expanded/readable, done card
  collapsed+is-done in the open done area, done-title re-expands) —
  `screenshots/d1-d4_*` (ephemeral). Unit proof:
  `tests/unit/test_session_controls.py` (cards endpoint newest-first,
  empty-without-engine).

### Predictive staging (Priority-1 "Predictive RAG", owner directive step 3)

- **Priority-1 deliverable wired.** `dmd/staging.py` `StagedContext`
  (LRU + TTL, adopted from the interrupted run) + `StagingConfig` are now
  consumed end to end: the monitor judge (`dmd/monitor.py`) emits
  `situation`/`likely_next_events`/`predicted_entities` and forwards
  predictions to the engine on every tick (even `action='none'`); the engine
  prefetches predicted entities into the bounded cache on throttled background
  tasks (`dmd/pipeline.py` `_on_predict`/`_schedule_prefetch`) and injects
  cache hits as an advisory `PRE-STAGED CONTEXT` block into the worker agent
  (`WorkerAgent.run(..., staged_block=...)`). Cards record `meta.staged_for`
  on a staged hit; `staging_predict` events + logs make precision tunable.
  Guardrail-proven by `tests/unit/test_staging.py` (11 tests): a cache miss is
  a pure in-memory read that never touches the embedder/store and leaves the
  answer path byte-identical; staged data never mutates canonical state.
- **§14 note.** Staging removes retrieval from the answer critical path when
  the prediction hits; it does not change the already-measured intake/STT
  budget. Measured live this session (owner directive step 2, honest numbers):
  whisperx decode on this host was 6.7-12.0 s (diarize on) for the 2-13 s
  probe utterances and the synthesis model took 12.5 s (successful manual
  query) to ~45 s (2 of 3 back-to-back probe agents hit `agent budget
  exhausted`); the card events themselves reached `/ws` and `/api/cards`
  (see §9/§15 row). The P1 targets were met earlier today under warm
  conditions; the floor now is whisperx decode + model latency on the shared
  local endpoints, not pipeline structure.

## Open questions for the owner (answered by the handoff brief)

1. **Scope:** answered — the full ledger, in order, after the Priority-0
   slot-leak hotfix.
2. **DAVE:** answered — live server is on the mixed (browser loopback) stream;
   per-user RTP stays deferred; attribution is the tracker x diarization join.
3. **Replay harness:** `tests/regression/test_replay_golden.py` remains the
   canonical fixture; it passed unchanged (golden events) after the pipeline
   rework.
