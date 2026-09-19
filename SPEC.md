# Voice Chat DM Assistant — Architecture Spec (v2, agentic)

Status: agentic redesign, 2026-08-26. Supersedes v1 (frozen 2026-08-25).
v1 modeled the live path as a fixed *retrieve → synthesize* RAG pipeline with a
static tool registry and a baked 5e SRD. v2 makes the live path **agentic**, per
the owner's intent: live worker agents *do work* as it comes in, with general
tools (bash, web search, web fetch), operating from a reusable world map the init
agent builds once.

---

## 1. Purpose

A locally-run web app that listens to a Discord voice session while the owner DMs,
transcribes each speaker in real time with exact identity, and — as the scene
unfolds — feeds the DM the most accurate, relevant information at that moment. The
DM watches one window; relevant information arrives as **cards** and **ephemeral
scene context**; the DM reads, and marks things done.

Target: a player says "I search the corpse" → within a few seconds a
corpse-specific skill-check table (from repo lore + live character data + the
correct system's rules) appears.

## 2. Core principles (these override every implementation detail)

- **Agentic, not pipelined.** The live path is agents that *do work* as it comes
  in — not a fixed retrieve→stuff→synthesize pipeline. Retrieval, tools, and web
  access are *capabilities an agent reaches for*, not pipeline stages.
- **No baked rules.** Rules for any system (5e, Shadowdark, Pathfinder, Cyberpunk,
  whatever) are looked up live by an agent (web), never embedded at init. The repo
  is the world; the internet is the rulebook.
- **Live and responsive.** Information reflects the *current* scene and *live*
  data (character sheets, VTT, whatever the repo's tools expose), not a static
  snapshot.
- **Local-first, DM-only.** No cloud dependency; all AI endpoints user-configured
  and swappable. No player-facing surface.
- **Configurable, zero hardcoding.** No model names, folder conventions, or vendor
  vocabularies anywhere. Any DM's folder works.
- **The DM spends no steps.** The agent decides what's relevant, fetches it, and
  presents it ready-to-read. The DM's only actions: read, (expand if long), mark
  done. No drill-down generation, no multi-step, no filtering.
- **Start simple, add layers only when felt.** v1 ships cards (not buckets), a
  single fast dispatch, and a proactive transcript monitor. Extra abstraction is a later
  iteration, not a first cut.

## 3. Two strictly separated phases

- **Init (pre-session):** an init agent explores the worldbuilding folder once and
  produces a reusable **world map**. This is the only phase that explores the
  filesystem broadly.
- **Live (during session):** worker agents do work, triggered by the transcript.
  They operate from the world map + their tools and never re-explore the folder
  from scratch.

## 4. Voice input

- **Discord voice tap:** a bot joins the channel; per-user RTP streams keyed
  SSRC → Discord user ID. **Exact speaker identity is metadata, not diarization.**
  DAVE E2EE is mandatory; the receive spike (§17 Phase 0) and fallback ladder gate
  everything.
- **Crosstalk caveat (open investigation):** if the DAVE receive path is
  unstable and we fall back to system-loopback audio for STT, **overlapping
  speakers cannot be reliably attributed** — loopback is a single mixed stream.
  Best-effort attribution then comes from gateway speaking-events per user ID,
  which is degraded under crosstalk. This is worth investigating before relying on
  exact per-speaker identity in the fallback path.
- Per-user streams (when available) transcribed independently and concurrently
  (overlap never mixes).
- Pluggable **STT** endpoint (OpenAI `/v1/audio/transcriptions` dialect), chunked
  utterances, per-user hotwords best-effort.
- **Lexicon** (built by the init agent): canonical entity names + aliases →
  phonetic/fuzzy post-correction of every transcript, so a spoken name retrieves
  its page even when mangled. One build, two consumers (retrieval + STT).

## 5. Init agent → the world map

Runs once on "pick folder"; regenerates incrementally on file changes. Produces a
reusable **world map** — structured artifacts (markdown/JSON under a known project
location) that prime live agents so they don't re-discover. The map is
*orientation, not the answer*: it tells an agent what exists and how to reach it,
so the agent spends effort on the specific question, not exploration.

The map contains:

1. **Structure** — directory tree + what each area covers.
2. **Entities** — named things (characters, places, items, factions): canonical
   name, aliases, one-line summary, where it lives in the repo.
3. **Tools** — every script/skill/tool discovered in the folder (character-sheet
   access, VTT access, etc.), each with: purpose, exact invocation, output shape,
   probe status, timeout. Probed once at init; the live path calls only probed
  tools, never arbitrary scripts ad hoc.
4. **Players** — per player: name, character-sheet location, how to fetch live
   stats (which tool), current-state seed.
5. **Context** — narrative/organizational notes: how the campaign is structured,
   running threads.

Embeddings: local by default; entity vectors live in the index so an agent can use
retrieval as a capability. The provider is pinned into map metadata; switching it
forces a re-embed (never mix spaces).
**No rules are baked.** The map never embeds a rulebook (explicit correction to
v1's SRD embedding).

## 6. Live worker agents

A **worker agent** is a bounded unit of work, fired by a trigger (§7). It has:

- **bash** — run the repo's probed tools (character sheets, VTT access, scripts)
  via the map's tool entries.
- **web search** — look up anything not in the repo (rules for any system, general
  knowledge).
- **web fetch** — read a specific page/URL.
- **the repo** — read worldbuilding docs via the map's pointers.
- **the world map** — its starting orientation.
- **the per-player state store** — what's tracked about each player (§11).

An agent decides what it needs for the triggering moment, reaches for capabilities
as needed, and produces output. It does *not* follow a fixed retrieve→synthesize
sequence.

- **Async + concurrent:** agents run concurrently, bounded by a configurable
  `max_concurrent` (≤ backend slot count). Conversations overlap; work is never
  serialized.
- **Staleness:** a job carries its scene-context window; a superseded agent's
  output (scene moved on) is dropped, not rendered.

## 7. Triggering agent work (three sources)

1. **Explicit DM query** — the DM asks something in the query box. Highest
   priority; preempts ambient work.
2. **Fast lane (reactive, per-utterance gate).** A cheap, fast model classifies
   each transcript segment: is this worth a worker agent? Triggers: a question, a
   declared player action ("I loot the corpse"), a scene-relevant statement.
   Otherwise (small talk, OOC) → logged to the transcript, no agent fired. This is
   a *gate*, not a stage that does the work.
3. **Transcript monitor (proactive, async).** A background monitor continuously reads the rolling transcript (plus scene context and per-player state) on a cadence — not tied to a single utterance. **Based on its judgment of the transcript, it triggers a worker agent** when it decides something is worth surfacing without a direct trigger (a reminder, a looming consequence, a thread the party forgot). It's a monitoring mechanism, not a named role. It **tasks for both tiers** — ephemeral (§8) for light context, card (§8) for a durable card — whichever the moment needs.

Single-role deployments route the fast lane and monitor classification through
synthesis and remain fully functional.

## 8. Two output tiers

- **Ephemeral / informational → agentic RAG.** The worker agent runs agentic RAG
  for the triggering transcript portion: retrieves relevant context (map/index,
  tools, web as needed), then **the same agent refines and verifies the output
  against that transcript portion** so it's grounded in what actually just
  happened. **No separate post-RAG refinement LLM step** — the agent that retrieves
  also produces the refined final output. One agent, one pass. Fast; surfaces as
  ephemeral scene context.
- **Cards → agentic synthesis, model-bound.** Durable cards (loot table, NPC,
  location, ruling) built by an agent doing real synthesis — structured items,
  pulling live data where relevant. Latency = model speed/efficiency; accepted as
  the expensive tier.

"Agentic RAG" = the agent *uses* retrieval as a capability (decides what to fetch,
combines, refines, verifies). RAG survives as a tool an agent reaches for; what is
dead is RAG-as-the-whole-pipeline.

## 9. Cards (v1: no bucket layer)

A **card** is the unit of party state. v1 has no separate "bucket" abstraction —
a card can simply *contain* structured items (a loot table is one card with item
rows). If a container layer proves needed later, it's a later iteration.

- **Structured items (optional):** a card may carry item rows (e.g. loot: name,
  quantity, DC to find, DC to use, owner) rather than flat markdown, so items are
  addressable, not just text.
- **Player association:** `player_ids` (0 = ambient/shared, 1 = targeted, many =
  group). The card states *who it is about*.
- **Pre-generated, expand-on-click:** content is generated when the card is made,
  not on click. Click = expand/collapse for screen space; nothing loads, nothing
  generates.
- **Lifecycle: active → done (never deleted).** A card is *done* when its content
  is resolved, by either of:
  - the **DM marks it done** (dismiss), or
  - the **AI hears it resolved** in the transcript (e.g. the loot was claimed, the
    NPC was dealt with) and auto-marks it done.
  A done card is **set aside / moved below** the active stream — still viewable,
  never destroyed. This is how cards are "handled": deprecated, not deleted.
- **Kinds:** skill_table, loot, npc, location, ruling, info, transcript_notice,
  error.

## 10. Ephemeral scene context (separate stream)

A distinct section from cards. Auto-pops when an agent decides something is
scene-relevant; decays on staleness (configurable horizon). The DM never touches
it. Fed by the agentic-RAG tier (§8, fast).

## 11. Per-player state store

SQLite, per project. Seeded at init from the character-sheet tools (the map says
where each player's data is); updated as cards are marked done and as the AI
observes the transcript (loot given → recorded against the player; knowledge
gained → tracked). This is the "keep track of everyone" mechanism, and it feeds
back into what agents know — so an agent doesn't re-give Kael the key he already
has, and knows only one player saw the trap. It also powers the AI's "this card is
now resolved" detection (§9).

## 12. Model gateway

Role-based, zero hardcoding. Roles: **synthesis** (required; card synthesis, agent
work, init enrichment), **fast** (optional; the dispatch gate + monitor
classification), **stt**, **embeddings** (local). Any slot swappable via config.

- Custom base URLs + optional API keys (Bearer). Free-form `extra_body` passthrough
  for model-specific params — reasoning-effort vocabulary varies by model, so it's
  a user-provided string, never an enum.
- **Startup probe** per role: confirm the configured model exists and its
  `extra_body` params are accepted before a session depends on them. `/v1/models`
  listings don't advertise param support; capability is proven only by the probe.

## 13. Storage

SQLite single file per project: entity index + vectors, world map, probed tool
registry, per-player state, session transcripts, job log. No DB server.
Audio/transcripts stay local; lore leaves the machine only as context inside
prompts to user-configured endpoints.

## 14. Latency budget

- **Ephemeral (agentic RAG):** target ≤ ~2s — retrieval-speed + agent reasoning.
- **Cards (agentic synthesis):** model-bound; target ≤ ~4–6s, explicitly accepted
  as model-speed-dependent.
- The fast-lane gate + cadenced transcript monitor keep the expensive tier firing only when
  needed, so the common case stays fast.

## 15. Web UI

One live window, WebSocket push, responsive to placement (second monitor, same
screen, laptop). Dark theme.

Panes: rolling per-speaker transcript; **card stream** (latest-first,
expand / mark-done, player-badged, with a set-aside *done* area); **scene-context**
section (ephemeral, auto-fading); manual query box; session controls (pause
capture, mark-OOC).

## 16. Non-goals

- No player-facing surface (DM-only).
- No cloud dependency (endpoints local/swappable).
- **No bundled VTT** — but the repo's own VTT-access tools are fair game (agents
  have bash/web).
- No automatic Discord image/chat-text ingestion in v1 (voice only).
- **No baked rulebook** (any system's rules are fetched live).
- **No bucket layer in v1** (cards only; revisit if a container is actually
  needed).

## 17. Phasing

| Phase | Contents | Gate to advance |
|---|---|---|
| 0 | DAVE receive spike (§4); **investigate crosstalk attribution under loopback fallback** | per-user decodable audio, <2% packet loss across a join/leave, stable 30 min; crosstalk behavior characterized |
| 1 | tap → STT → transcript pane; **init agent → world map**; manual query → worker agent → card | end-to-end on a replay recording |
| 2 | fast-lane dispatch + **transcript monitor** → auto worker agents (cards + ephemeral) | precision/recall tuned on replayed sessions |
| 3 | per-player state store live (seed + update); AI auto mark-done; ambient scene-context decay | phase-2 dispatch trust established |

The **replay harness** is a first-class deliverable: past recordings (owner records
all sessions) are piped through the whole pipeline offline to tune dispatch and
measure retrieval quality before anything is trusted live.

## 18. Risks

| Risk | Mitigation |
|---|---|
| DAVE receive instability | step-0 spike gates everything; fallback ladder (§4) |
| **Crosstalk attribution under loopback** | investigate Phase 0; degrade to gateway speaking-events per user; flag low-confidence attributions in the transcript |
| Homebrew noun mangling | lexicon layer-1 + layer-2 post-correction; canonical linking |
| Agent over-fetching / slow | fast-lane gate + cadenced transcript monitor; per-agent tool-call budget; web timeouts |
| Rules need web access | fetched live by design; agents degrade to "I couldn't verify that rule" offline |
| Structured-output drift across servers | JSON-schema + defensive parse + startup probe |
| GPU/host contention | LLM lanes slotted; embeddings CPU-local; OBS untouched |
| Secret leakage | keys only in config/env; never logged, never committed |

## 19. Open items (to be nailed during the build, not blocking design)

- **DAVE crosstalk differentiation under loopback:** how far can per-speaker
  attribution go in the fallback path? (Phase 0 investigation.)
- **World-map artifact layout:** exact files/schemas per map section (§5).
- **Per-agent tool-call budget and web timeouts:** set from replay data (§18).
- **Fast-lane trigger phrasing + monitor cadence:** tuned on replay (§7).
- **Bucket layer:** revisit only if cards genuinely need a container (deferred).
