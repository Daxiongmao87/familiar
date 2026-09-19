# AGENTS.md — Repository Guidelines

## Project positioning

- Project: Voice Chat DM Assistant.
- Goal: A locally-run web application that listens to a Discord voice
  session while the DM session is active, transcribes each speaker in real
  time with exact identity, and pushes contextually relevant, gitignored
  artifacts (skill-check tables, location briefs, rules rulings) to a live
  UI window attached to the owner's worldbuilding repository. Serves the
  campaign owner (DM); "done" means the live UI renders real-time,
  identity-accurate artifacts from a real voice session on the local
  machine.
- Module boundaries and ownership (see `dmd/`): `server.py` (FastAPI app,
  WebSocket event bus, REST endpoints, entrypoint, and the live UI window),
  `voice_presence.py` (Discord DAVE E2EE voice presence), `sources/`
  (`discord_src.py`, `browser.py`, `replay.py`, `transcript_replay.py`),
  `vad.py` +
  `speaking_tracker.py` (utterance segmentation and per-speaker state),
  `pipeline.py` + `triggers.py` (orchestration and fast-lane triggers),
  `openjev.py` (openjev deploy/wait gate),
  `jevworker.py` (JEV-routed deterministic worker),
  `agent.py` (legacy worker-agent loop), `monitor.py` (proactive
  transcript monitor), `terms.py` (zero-LLM term collection),
  `staging.py` (predictive-retrieval staging), `streaming_stt.py` +
  `stt_health.py` (streaming STT client and endpoint health),
  `attribution.py` + `player_state.py` + `world_map.py` (speaker
  attribution, players, world context),
  `gateway.py` (role-based OpenAI-compatible model endpoints), `scanner.py`
  + `init_pass.py` (deterministic scan to doc graph and entity index),
  `lexicon.py` + `enrich.py` (lexicon artifact and enrichment lane),
  `index_store.py` + `embedder.py` (SQLite + sqlite-vec index and local
  embeddings), `orchestrator.py` (bounded async job orchestration),
  `tools_reg.py` (tool registry), `types.py`, and `config.py`.
- Non-goals: no cloud dependency (all AI endpoints local and swappable), no
  VTT integration in v1, no automated Discord image/chat ingestion in v1.

## Build, test, and development commands

- Build / install: `python -m venv .venv && . .venv/bin/activate && pip
  install -r requirements.txt` (no package.json, pyproject, Makefile, or CI
  yet — plain Python, dependencies in requirements.txt)
- Test: `pytest` (pytest.ini: testpaths=tests, asyncio_mode=auto; test files
  under tests/unit, tests/regression, tests/e2e)
- Run: `python dmd/server.py config.yaml` (main() defaults to config.yaml)
- Run the relevant verification before declaring any change done.

## Skills and tooling

- Use the `skills-marketplace` skill to find skills that help with the work
  in this project, and download what is needed autonomously — do not wait
  to be asked for a capability that the marketplace already provides.
- Ephemeral artifacts stay out of git: screenshots, captured test output,
  logs, and other verification evidence are gitignored, as are real
  configuration/environment files (.env, credentials, local overrides),
  whose templates are committed instead. Agent instruction files
  (AGENTS.md, .omp/, and siblings) are committed source. When a new kind
  of artifact appears, add it to .gitignore (see RULES.md, "Evidence and
  artifacts", "Configuration and environment", "Agent artifacts").


## Coding style and documentation

- Strict, explicit state modeling. Preserve existing module boundaries.
- Naming follows the conventions already in the codebase.
- Engineering fundamentals (RULES.md, "Engineering principles"): verify
  before you build on a prior — check installed versions, real behavior,
  and environment state instead of coding from memory; priors are neither
  conventions nor proof of modern practice — read conventions from this
  codebase, establish current practice from live sources; DRY —
  one authoritative home for each piece of knowledge; YAGNI — build what
  today's contract needs, nothing more; design big to small — architecture
  and module contracts before the code that fills them in.
- Patterns (RULES.md, "Design patterns and convention priority"):
  conventional patterns first; a custom pattern only with an inferred
  convention following the project -> industry -> aligned-new priority.
- Configuration (RULES.md, "Configuration and environment"):
  environment- and deployment-varying values are read from config, never
  hardcoded; real config/env files are gitignored, templates committed.
- In-code documentation is the default (RULES.md, "Documentation in code"):
  a file-level doc comment, plus doc comments on every module, class, and
  public function in the language's standard form. Where no convention is
  declared, pydoc/JSDoc style is the default.
- Size (RULES.md, "Size"): no file over 1000 lines (tests included); lines
  over 100 characters are discouraged — keep them rare and deliberate.

## Testing guidelines

- Tests prove behavior, not plumbing: state transitions, invariants, error
  paths, and the contract the code claims.
- Regression discipline (RULES.md, "Testing"): every bug fix carries a
  regression test that was red against the defect and green after the fix,
  landed with the fix, and pinned at the layer where the defect lived.
- MVP scope (RULES.md, "Testing"): core paths and primary failure modes.
  Edge cases are post-MVP and never block the MVP.
- Test results are ephemeral: capture them, use them, never commit them.

## UI and frontend verification

- For any UI or frontend change, capture screenshots of the actual surface
  and use them to audit and assess the implementation before declaring it
  done.
- Screenshots are ephemeral verification artifacts: never committed to
  source control.

## Commit and pull request guidelines

- Concise imperative commit messages, scoped to the change. Optional
  conventional prefixes: feat, fix, refactor, chore, docs, test.
- Pull requests describe impact, changed contracts, verification performed,
  and residual risks.

## Definition of done

A change is done when:

- the behavior works as specified and was verified by running it;
- the relevant tests pass;
- a bug fix includes a regression test that failed against the defect and
  passes unchanged after the fix (RULES.md, "Testing");
- RULES.md is respected: hard rules unviolated, any over-long line is
  deliberate, ephemeral artifacts uncommitted;
- residual risks are stated.

## Conventions

- Name test files `test_*.py`.
