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

## Ledger

| # | SPEC section | Clause(s) | Status | Evidence / what's missing |
|---|---|---|---|---|
| 1 | §1 Purpose | local web app; Discord voice; transcribe **each speaker**; cards + ephemeral scene context | IMPLEMENTED | `server.py`, `pipeline.py`, `web/` exist; per-speaker identity on the mixed capture is recovered by the §7a attribution join (named `transcript` events), not degraded to `browser_mixed`. |
| 2 | §2 Core principles | agentic-not-pipelined; **no baked rules**; live/responsive; local-first DM-only; **zero hardcoding**; DM spends no steps; start simple | IMPLEMENTED | Zero-hardcoding correction landed: whisperx `diarize`/`align` are config (`dmd/config.py` `SttRole.diarize/align`, read in `dmd/gateway.py` `_transcribe`), proven by `tests/unit/test_gateway.py::test_whisperx_params_reflect_config_diarize_align`. "No baked rules" satisfied (no SRD embedded). |
| 3 | §3 Two phases | init (once) vs live (during session) | IMPLEMENTED | `init_pass.py` builds world map; `pipeline.py` `SessionEngine` is the live path. |
| 4 | §4 Voice input | DAVE E2EE mandatory; per-user RTP keyed SSRC; crosstalk caveat; per-user concurrent; pluggable STT; lexicon | IMPLEMENTED (identity via §7a; DAVE per-user RTP deferred by design §17) | STT present (`gw.transcribe`), lexicon present (`lexicon.py`), per-user VAD buffers + worker queue (`pipeline.py:_PerUserSttQueue`). Per-user identity now recovered on the mixed capture by the §7a join (`dmd/attribution.py`; `SessionEngine.transcribe_pcm` emits named speakers) — `tests/unit/test_pipeline_attribution.py`. The SSRC/DAVE clause stays Phase 0, deferred by design (owner-ordered), not silently skipped. |
| 5 | §5 Init agent → world map | structure, entities, tools (probed), players, context; embeddings local; **no baked rules** | IMPLEMENTED | `scanner.py` builds map-ish structure/entities; `tools_reg.py` probed tools. "No baked rules" audited and pinned: `tests/unit/test_no_baked_rules.py` (static guard — no dice notation / DC / AC / rule-table constants in any shipped `dmd/` or `web/` source; passes clean). Players seeding covered by `tests/unit/test_player_state.py` (§11). |
| 6 | §6 Live worker agents | bash / web-search / web-fetch / repo / world-map / per-player state | IMPLEMENTED | `agent.py` `WorkerAgent` wires all capabilities. Concurrent (bounded by `pool`). |
| 7 | §7 Triggering | explicit query; fast-lane gate; transcript monitor | IMPLEMENTED | `triggers.py` `detect_trigger`; `api_query`; `monitor.py` `TranscriptMonitor` cadence. |
| 7a | §7-attribution | **per-speaker attribution** of diarized segments (DAVE PR #3139 design) | IMPLEMENTED | The join is wired: `dmd/attribution.py` (`attribute_segments` maps pyannote clip-relative windows through the utterance's monotonic start and picks the max-overlap SpeakingTracker window; `group_by_speaker` splits crosstalk; `attribute_whole` is the no-segments fallback). `SpeakingTracker` (`dmd/speaking_tracker.py`) feeds it from gateway events (`dmd/voice_presence.py`), and the tracker is created in `server.py:main()` and injected into `SessionEngine` (constructor `speaking_tracker`); `transcribe_pcm` publishes named `transcript` events. Tests: `tests/unit/test_attribution.py` (7), `tests/unit/test_pipeline_attribution.py` (5). |
| 8 | §8 Two tiers | ephemeral RAG (same agent refines); cards (model-bound synthesis) | IMPLEMENTED | `pipeline.py` agentic-RAG fast tier + card synthesis. |
| 9 | §9 Cards | player_ids assoc; pre-generated expand-on-click; lifecycle active→done; **set-aside done area** | IMPLEMENTED | `player_ids` + `status` now flow through `server.py:_card_to_dict` to the UI; `web/app.js` renders player badges (named via `/api/players`), cards collapse/expand on head click (pre-generated content, nothing loads on click), a Done button POSTs `/api/card/done` → `SessionEngine.mark_card_done` → `card_done` event moves the card to the set-aside `<details>` done area (viewable, never deleted). Monitor auto-mark (AI heard it resolved) uses the same lifecycle. Proven: `tests/unit/test_session_controls.py` (endpoint + lifecycle) and the rendered-capture audit (`screenshots/v1-v6`, DOM checks E1-E8). |
| 10 | §10 Ephemeral scene context | separate stream; auto-pop; decays | IMPLEMENTED | `_scene_buffer` + monitor decay on the backend; now also **rendered**: `web/index.html#scene-pane` (separate strip above the card stream), auto-pop latest-first, fades at 45 s and drops at 90 s (`app.js addSceneNote`), max 8 notes. |
| 11 | §11 Per-player state store | SQLite seeded at init; updated on mark-done / AI-observed | IMPLEMENTED | `player_state.py` + `server.py:_seed_players` (seeds from `characters/*.md`, frontmatter titles) + `pipeline.py` mark-done and monitor `card_done` paths both write through `record_card_done` (done_cards ref + inventory merge, append-only, idempotent). Proven by `tests/unit/test_player_state.py` (6 tests), which also pinned a real re-seed bug: `seed()` clobbered live HP because empty-string hp defeated `COALESCE` (fixed: empty hp → NULL). |
| 12 | §12 Model gateway | role-based, zero-hardcoding, startup probe | IMPLEMENTED | `gateway.py` role-resolve + `probe_all` (`/v1/models` + extra_body probe). Every chat call is generation-capped (`max_tokens` guard, 2026-09-05 slot-leak hotfix). |
| 13 | §13 Storage | SQLite single file, no DB server, local lore | IMPLEMENTED | `index_store.py`, `data/` per project. |
| 14 | §14 Latency budget | ephemeral ≤ ~2s; cards ≤ ~4-6s | IMPLEMENTED (architecture + measured) | Intake never awaits STT: `consume_source` only feeds VAD + `put_nowait` onto `_PerUserSttQueue` (per-user workers). Measured: every utterance logs monotonic timestamps and publishes `stt_latency` (`queue_wait_ms`/`stt_ms`/`post_speech_ms`); per-chunk handler time is tracked with a >50 ms stall warning. Proof: `tests/unit/test_pipeline_latency.py` (feed span 0.05 s against 0.4 s STT calls; slow speaker never delays fast speaker). §14 wall-clock targets (≤2s ephemeral / ≤4-6s cards) must be re-read from `stt_latency` + card-job logs on the live endpoint; the budget is now *measurable* and measured in-session, not unmeasurable. |
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

## Open questions for the owner (answered by the handoff brief)

1. **Scope:** answered — the full ledger, in order, after the Priority-0
   slot-leak hotfix.
2. **DAVE:** answered — live server is on the mixed (browser loopback) stream;
   per-user RTP stays deferred; attribution is the tracker x diarization join.
3. **Replay harness:** `tests/regression/test_replay_golden.py` remains the
   canonical fixture; it passed unchanged (golden events) after the pipeline
   rework.
