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
  (`discord_src.py`, `browser.py`, `replay.py`), `vad.py` +
  `speaking_tracker.py` (utterance segmentation and per-speaker state),
  `pipeline.py` + `triggers.py` (orchestration and fast-lane triggers),
  `gateway.py` (role-based OpenAI-compatible model endpoints), `scanner.py`
  + `init_pass.py` (deterministic scan to doc graph and entity index),
  `lexicon.py` + `enrich.py` (lexicon artifact and enrichment lane),
  `index_store.py` + `embedder.py` (SQLite + sqlite-vec index and local
  embeddings), `orchestrator.py` (bounded async job orchestration),
  `tools_reg.py` (tool registry), `types.py`, and `config.py`.
- Governing reference: `SPEC.md` (architecture spec, design-frozen
  2026-08-25).
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
- Agent artifacts are gitignored: the project's .gitignore covers the
  artifacts agents produce (screenshots, captured test output, logs, other
  ephemeral verification evidence). When a new kind of artifact appears,
  add it to .gitignore (see RULES.md, "Evidence and artifacts").

## Coding style and documentation

- Strict, explicit state modeling. Preserve existing module boundaries.
- Naming follows the conventions already in the codebase.
- In-code documentation is the default (RULES.md, "Documentation in code"):
  a file-level doc comment, plus doc comments on every module, class, and
  public function in the language's standard form. Where no convention is
  declared, pydoc/JSDoc style is the default.
- Size (RULES.md, "Size"): no file over 1000 lines (tests included); lines
  over 100 characters are discouraged — keep them rare and deliberate.

## Testing guidelines

- Tests prove behavior, not plumbing: state transitions, invariants, error
  paths, and the contract the code claims.
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
- Every commit that changes behavior or contracts gets a changelog
  Unreleased entry naming its SemVer class (below), or an explicit
  non-release-affecting exclusion.
- Pull requests describe impact, changed contracts, verification performed,
  and residual risks.

## Versioning and changelog (SemVer 2.0)

Version authority has three parts; all three are required for a release:

1. The project manifest is the authoritative version string.
2. Annotated git tags named vX.Y.Z are immutable release snapshots.
3. CHANGELOG.md is the human release ledger: changes, commits,
   verification evidence, migration notes, residual risks.

A dirty working tree is never a released state.

### SemVer rules

Pre-1.0 (0.x.y):

- Minor, 0.(x+1).0: new capability, new API or contract, new user-visible
  behavior, persistence or migration behavior, or any breaking contract
  change while pre-1.0.
- Patch, 0.x.(y+1): backwards-compatible fix, documentation, test
  hardening, verification tooling, or safety fix that adds no capability
  and changes no contracts.

At and after 1.0.0 (standard SemVer 2.0.0):

- Major: breaking API, contract, persistence, or workflow change.
- Minor: backwards-compatible capability addition.
- Patch: backwards-compatible fix, documentation, test, or security
  hardening.

### Classification by change type

- New capability or additive contract: minor.
- Backwards-compatible fix: patch.
- Breaking change: minor pre-1.0, major post-1.0.
- Docs-only or test-only with no behavior change: patch, or no version
  impact with an explicit exclusion.
- State, persistence, or migration changes require replay or load evidence
  and a migration note.

### Changelog

- CHANGELOG.md keeps an [Unreleased] section plus one section per released
  version, newest first.
- Every post-release commit that affects product, contracts, or behavior
  gets an Unreleased entry: change summary, commit hash, SemVer class,
  verification state, residual risk — or an explicit non-release-affecting
  exclusion.
- "No unreleased changes" is valid only when every post-tag commit is
  recorded as non-release-affecting.

### Release workflow

1. Classify the change before implementation; name the provisional SemVer
   class.
2. Implement in scoped commits; verify (relevant tests, the full suite when
   feasible, runtime evidence when runtime behavior is claimed).
3. Bump the manifest version to the selected version.
4. Update CHANGELOG.md: version, date, included changes, commits,
   verification evidence, migration notes, residual risks.
5. Commit release metadata (chore(release): vX.Y.Z).
6. Create an annotated tag: git tag -a vX.Y.Z -m "Voice Chat DM Assistant vX.Y.Z".
7. Verify: git describe --tags, the tag points at the release commit, the
   manifest version matches the tag, the changelog has the matching
   section.
8. Push the commit and tag when a remote exists. If push fails, report the
   local tag and the concrete blocker; the local annotated tag is the local
   release authority.

### Prohibited substitutions

- A commit hash alone is not a version.
- Task completion is not a release without manifest, changelog, and tag.
- A dirty tree is never tagged.
- Smoke checks are not full-fidelity verification evidence.

## Definition of done

A change is done when:

- the behavior works as specified and was verified by running it;
- the relevant tests pass;
- a changelog Unreleased entry (or explicit exclusion) exists;
- RULES.md is respected: hard rules unviolated, any over-long line is
  deliberate, ephemeral artifacts uncommitted;
- residual risks are stated.
