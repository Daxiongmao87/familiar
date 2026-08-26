# Voice Chat DM Assistant — Architecture Spec

Status: design frozen 2026-08-25, pre-implementation.
Origin: audit thread with the project owner; every decision below was explicitly made or accepted by them.

---

## 1. Purpose

A locally-run web application that listens to a Discord voice session while the owner DMs,
transcribes each speaker in real time with exact identity, and pushes contextually relevant,
generated artifacts (skill-check tables, location briefs, rules rulings) to a live UI window
fetched from the owner's worldbuilding repository.

Target experience: a player says "I search the body" → a corpse-specific skill-check table,
built from repo lore + party stats, appears within **2–4 seconds** of end-of-utterance.

## 2. Non-goals

- No player-facing surface. DM-only.
- No cloud dependency by default. All AI endpoints are user-configured and swappable.
- No hardcoded model names, folder conventions, or vendor-specific parameter vocabularies anywhere.
- No VTT integration in v1.
- No automatic Discord image/chat-text ingestion in v1 (voice only).

## 3. System architecture

```
                        ┌────────────────────────── pre-session ──────────────────────────┐
                        │                                                                  │
  project folder ──────►│ deterministic scan ─► doc graph + entity index                   │
  (picker in UI)        │        │                                                         │
                        │        ▼                                                         │
                        │  LLM enrichment (synthesis lane) ─► summaries, aliases           │
                        │        │                                                         │
                        │        ├──► lexicon artifact (hotwords + canonical names)        │
                        │        ├──► tool discovery: probe & register repo scripts        │
                        │        └──► index store (SQLite + sqlite-vec)                    │
                        └──────────────────────────────────────────────────────────────────┘

                        ┌────────────────────────── live session ─────────────────────────┐
                        │                                                                  │
  Discord VC ──────────►│ voice bot: per-user RTP streams (DAVE E2EE)                      │
                        │        │  opus/PCM per speaker                                   │
                        │        ▼                                                         │
                        │  utterance segmenter (VAD) ──► STT endpoint(s), parallel         │
                        │        │                                per-user                 │
                        │        ▼                                                         │
                        │  lexicon post-correction (always on)                             │
                        │        │                                                         │
                        │        ▼                                                         │
                        │  fast lane: trigger / IC-vs-OOC / query rewrite                  │
                        │        │                                                         │
                        │        ▼                                                         │
                        │  retrieval (hybrid: vector + keyword over index) ◄── tools       │
                        │        │            (cached character data etc.)                │
                        │        ▼                                                         │
                        │  synthesis pool (bounded async, priorities, staleness cancel)    │
                        │        │                                                         │
                        │        ▼                                                         │
                        │  WebSocket push ──► live UI window (transcript + cards)          │
                        └──────────────────────────────────────────────────────────────────┘
```

Two strictly separated paths. The init path may explore, run enrichment, execute probes.
The live path is pure retrieval → generation → push; it never explores the filesystem.

## 4. Components

### 4.1 Discord voice tap (bot)

**Decision:** a minimal bot joins the campaign voice channel and receives each user's audio as a
separate stream keyed SSRC → Discord user ID. Speaker identity is exact metadata, not inference.
No diarization ML anywhere in the primary path.

Constraints (MEASURED against current ecosystem, 2026-02→2026-08 sources):

- DAVE E2EE is mandatory for all Discord voice since 2026-03-01 (official docs:
  docs.discord.com/developers/topics/voice-connections). Bots must negotiate MLS via a DAVE lib.
- JS stack (`@discordjs/voice` + `@snazzah/davey`) had open receive-path defects through 2026-03:
  reconnect loops (#11419), decryption failures (#11445), silent ~34% packet loss during key
  transitions corrupting STT input. Fix PR #11449 existed; full resolution unverified.
- py-cord reported working on identical channels by one production user; maintainer called their
  implementation not-yet-stable elsewhere. Conflicting signals — do not assume either works.

Bot requirements: `selfDeaf: false`, `selfMute: true`, Connect permission. Streams arrive per
transmission (VAD/PTT).

**Step 0 spike (gates everything):** throwaway bot records 5 minutes from ≥2 speakers.
Success criteria: decodable audio per user, <2% packet loss across at least one join/leave
(key-transition event), stable over 30 min. On failure, fallback ladder:

1. Pin whichever library/version demonstrably passes the spike.
2. Hybrid: system loopback audio for STT + gateway speaking-events per user ID for attribution.
3. Streaming diarization + voice enrollment (last resort; degrades under crosstalk).

### 4.2 STT pipeline

- Pluggable endpoint speaking the OpenAI `/v1/audio/transcriptions` dialect (near-universally
  cloned). Chunked utterance requests, not vendor websocket streaming — portable, costs ~0.5s.
- Per-user streams transcribed independently and concurrently (overlapping speakers never mix).
- Lexicon hotwords injected where supported (layer 1, best-effort): faster-whisper `hotwords`,
  OpenAI-style `prompt` field.
- Layer 2, always on regardless of endpoint: fuzzy/phonetic post-correction of every transcript
  against the lexicon → canonical entity names feed retrieval directly.

### 4.3 Lexicon

Byproduct of the entity index — one extraction pass, two consumers (retrieval + STT).
Built by the init agent:

- alias grouping ("Vex", "Vex'ahlia", "the ranger" → one canonical entry),
- false-positive filtering (capitalized non-names, common-word collisions),
- typing and weighting by narrative centrality (names > spells > places > factions).

Versioned alongside the index snapshot; regenerated incrementally by the file watcher.
A session always runs against one consistent (index, lexicon) snapshot.

### 4.4 Project initialization

Structure-agnostic and agent-led. Zero conventions required; Obsidian-ish patterns
(frontmatter, wikilinks, tags, headings) are exploited when present.

Order of operations on "pick folder":

1. Deterministic scan: file tree, frontmatter, links, headings → document graph + raw entities.
2. LLM enrichment (synthesis lane): per-doc one-line summaries, alias extraction, entity typing.
3. Tool discovery (see 4.6).
4. Embed + persist index; emit lexicon; pin embedding-model identity into index metadata.

File watcher reindexes changed documents incrementally after init.

Embeddings: local by default (ONNX/fastembed class, ~100MB–1GB, CPU-sufficient; query embed
~5–20ms). Switching embedding providers invalidates vectors → app detects metadata mismatch and
forces re-embed rather than mixing spaces.

### 4.5 Retrieval

Hybrid (vector + keyword/BM25) with reranking, over the SQLite/sqlite-vec store.
Retrieval budgets are tight: synthesis latency is prompt-prefill-bound, so generous stuffing
directly hurts the 2–4s budget. Entity-linked transcripts let a spoken name retrieve its page
even when STT mangled it (post-correction output).

SRD 5.1 + SRD 5.2 (CC-BY licensed) embedded at first init; house rulings are ordinary repo docs.

### 4.6 Tool registry (repo-provided integrations)

The init agent discovers tools existing in the project (e.g., live character-sheet access
scripts). Registration requires a validation probe executed once at init:

- does it run,
- is output machine-parseable (JSON preferred; schema captured),
- timing recorded → timeout assigned.

Registered tools become runtime-callable functions with aggressive caching (character modifiers
do not change mid-scene; refresh on demand). The live path calls registered tools only — it
never executes arbitrary discovered scripts ad hoc.

### 4.7 Model gateway

Role-based, zero hardcoding. Roles map to configured endpoints; any slot can be omitted except
where noted.

```yaml
models:
  synthesis:                       # REQUIRED. card generation, init enrichment, scene summaries
    base_url: http://localhost:8081/v1
    api_key: ${LLM_KEY}            # optional; sent as Authorization: Bearer
    model_id: <user-configured>
    extra_body: { }                # free-form JSON injected verbatim into request body
  fast:                            # optional. per-utterance classification (see 4.8)
    ...
  vision:                          # optional. init-time captioning of images in repo
    ...
  stt:
    base_url: ...                  # OpenAI transcription dialect
    api_key: ...
    extra_body: { hotwords?: ... } # layer-1 biasing, best-effort
  embeddings:
    provider: local | endpoint
    model_id: ...                  # pinned into index metadata
```

Startup validation:

1. Probe `{base_url}/models` for each role; warn if configured `model_id` absent.
2. One 1-token completion per role carrying its `extra_body`; rejected params flagged before a
   session ever depends on them (reasoning-effort vocabulary is model-specific — OpenAI
   `reasoning_effort`, template kwargs like `enable_thinking` elsewhere — hence free-form
   passthrough, never an enum).
3. `/v1/models` listings do NOT advertise parameter support (verified against upstream behavior);
   capability is proven only by the probe.

Structured outputs: llama.cpp-class servers honor `response_format` JSON-schema when running
jinja chat templates. The app relies on it for card payloads, with defensive-parse fallback.

Reference deployment verified on this machine (MEASURED): nginx :8081 (Bearer-gated `/v1/*`)
→ llama.cpp router :8085 → three loaded instances incl. ornith-1.5-35b MoE (`--parallel 6`,
`--ubatch 2048`, jinja on) and a 4B utility model. App concurrency defaults must stay ≤ server
slot counts.

### 4.8 Fast lane vs synthesis lane

| | fast | synthesis |
|---|---|---|
| frequency | every transcript segment | few times/min |
| latency budget | tens of ms | 1–3s generation |
| tasks | trigger detection, IC/OOC filter, query rewrite, entity-link disambiguation | cards, briefs, rulings, init enrichment, scene summaries |
| failure cost | missed/noisy trigger | wrong info read mid-game |

Single-role deployments route everything through synthesis and remain fully functional;
fast is an optimization.

### 4.9 Job orchestration

All synthesis work is discrete jobs through a bounded async pool:

```yaml
orchestration:
  max_concurrent: 3          # ≤ backend slot count
  job_timeout_s: 20
```

- Out-of-order completion is normal: cards render ordered by context timestamp, not arrival.
- Staleness cancellation: jobs carry their context window; superseded jobs (scene moved on)
  are dropped before generation burns time.
- Two priority classes: manual DM queries preempt ambient auto-triggers. Nothing fancier.
- Shared scene-context prefixes across concurrent jobs benefit from backend prompt caching.

### 4.10 Web UI

One live window, WebSocket push, responsive to placement (second monitor, same screen, laptop).
Panes: rolling transcript (per-speaker attributed), card stream (latest-first), manual query box,
session controls (pause capture, mark-OOC, dismiss/pin card). Dark theme assumed.

## 5. Storage

SQLite single-file database per project folder: chunks + vectors (sqlite-vec), entity graph,
lexicon snapshots, tool registry, session transcripts, job log. No DB server. Audio/transcripts
stay local; lore text leaves the machine only as retrieved context inside prompts to endpoints
the user themselves configured.

## 6. Phasing

| Phase | Contents | Gate to advance |
|---|---|---|
| 0 | DAVE receive spike (4.1) | success criteria met on this rig |
| 1 | tap → STT → transcript pane; manual query → retrieval → card | end-to-end on a real session recording (replay harness) |
| 2 | fast-lane triggers → auto cards | precision/recall tuned on replayed past sessions; no ambient yet |
| 3 | ambient scene tracker (rolling summary, proactive cards) | phase-2 retrieval trust established |

Replay harness is a first-class deliverable: past recordings (owner already records all sessions)
are piped through the entire pipeline offline to tune triggers and measure retrieval quality
before anything is trusted live.

## 7. Latency budget (target ≤ 4s post-utterance)

| Stage | Budget |
|---|---|
| utterance close → STT chunk | ~0.5–1.0s |
| STT response | ~0.3–1.0s |
| fast-lane classify + rewrite | ~0.05–0.15s |
| retrieval + rerank | ~0.1–0.3s |
| synthesis first tokens → card render | ~1.0–2.5s (prefill-bound; keep contexts tight) |

## 8. Risks

| Risk | Mitigation |
|---|---|
| DAVE receive instability | step-0 spike gates everything; fallback ladder 4.1 |
| Homebrew noun mangling | lexicon layer-1 + layer-2 correction; canonical linking |
| Trigger false positives | explicit phrases only in phase 2; replay-tuned thresholds |
| Structured-output drift across servers | JSON-schema + defensive parse + startup probe |
| GPU/host contention | LLM lanes remote-or-slotted; embeddings CPU-local; OBS untouched (bot adds no desktop audio path) |
| Secret leakage | keys only in config/env; never logged, never committed |

## 9. Open items

None blocking. Build begins at Phase 0.
