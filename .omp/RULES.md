# RULES.md — Hard Rules

Rules for this project. The hard rules are non-negotiable; a violation is a
defect. A rule that lives only in habit gets broken silently; this file
keeps the rules visible at the point of work.

## Size

- No source file over 1000 lines. This includes test files: tests rot too.
  A file past 1000 lines has stopped being one thing. Split it along the
  seam that already exists, not by moving the last 200 lines somewhere
  else.
- Lines over 100 characters are discouraged. The limit is pressure, not a
  wall: wrap when the expression can be split naturally, and keep an
  over-long line only when breaking it would hurt clarity (long URL,
  unbreakable identifier, generated string). Each over-long line is a
  smell: expect it to draw a glance in review and to be revisited.

## Documentation in code

- Document in code, not only in prose docs. Every source file opens with a
  file-level doc comment. Every module, class, and public function carries a
  doc comment in the language's standard form.
- If this project has not declared a commenting convention, the language's
  standard doc form is the default. Pydoc-style docstrings and JSDoc blocks
  are the canonical examples of good in-code documentation.
- A doc comment states what a unit does, why it exists, and the invariants
  of its contract. It is not a line-by-line narration of the code.
- A public unit without a doc comment is incomplete work.

## Dependencies

- Minimal dependencies. The dependency set is the project's supply-chain and
  maintenance surface; every dependency is a decision, not a convenience.
- Adding a dependency requires a stated justification and a changelog entry.
- Prefer the standard library and existing project code over a new
  dependency.

## Engineering principles

- Verify, never assume: priors about libraries, APIs, tools, versions, and
  the environment are out of date by definition. Check the installed
  artifact, its source or docs, or run it before coding against what you
  "know". An unverified assumption is a bug you have not met yet.
- Priors are not conventions — and never proof of modern practice. What
  feels like "the standard way" is a training-corpus memory, frozen at
  corpus time: stale at best, deprecated by now at worst. Read a
  convention from this project's code; establish current practice from
  live docs and sources. Never invoke a prior as convention, and never
  invoke it as "the modern way".
- DRY: every piece of knowledge — validation rule, format string, constant,
  protocol detail — lives in exactly one place. When you notice yourself
  repeating it, extract the shared source. Duplication is a defect, not a
  style choice.
- YAGNI: build what the current requirement needs, nothing that "will be
  needed later." No speculative parameters, unused abstraction layers, or
  dead hooks. Speculative generality is maintenance debt paid on top of
  every real change.
- Architecture first: design big to small — system boundaries and
  components before modules, module contracts before functions. Know the
  shape of the thing before implementing its parts; let the parts fit the
  shape, not invent it.

## Design patterns and convention priority

- Use conventional, documented design patterns where they fit: their names
  carry meaning to every future reader, and a shared vocabulary beats a
  private one.
- Where conventional patterns are insufficient and a custom pattern is
  justified, infer the acceptable convention in this priority order:
  1. existing project conventions,
  2. industry conventions for this language and domain,
  3. a new convention that aligns with the existing context.
- A custom pattern is a change to the project's convention state: name it,
  document it where it is introduced, and say why the conventional options
  were insufficient.
- Every rung of the priority ladder is evidence, not memory. Project
  conventions come from reading this codebase; industry conventions from
  current documentation and ecosystem sources — industries move, and a
  prior preserves the older practice under the name of the modern one.
  A remembered, unchecked convention is an unverified prior.

## Configuration and environment

- Do not hardcode what belongs in configuration: endpoints, paths, ports,
  credentials, feature flags, tuning values, and anything else that can
  vary by environment or deployment come from config, never from source
  literals.
- Source code is identical across deployments; configuration varies. Keep
  the varying part out of the committed source tree.
- Configuration artifacts with real values (`.env`, local overrides,
  credential and key files) are gitignored. Commit a documented template
  (e.g. `.env.example`) instead, so every environment knows which keys
  exist without any secret being in the repository.
- Secrets never enter git in any file, template included.

## Agent artifacts

- Agent instruction and tooling artifacts are environment artifacts, not
  source code: `AGENTS.md`, `CLAUDE.md`, `.agents/`, `.omp/`, `.claude/`,
  `.opencode/`, `.cursor/` and similar are gitignored and never committed.
  They are binding on disk for every agent working in the project; the
  repository stays free of them.
- The project's `.gitignore` covers configuration, environment, and agent
  artifacts from the first commit.

## Versioning

- This project uses SemVer 2.0.0; the full policy is in AGENTS.md under
  "Versioning and changelog".
- A version exists only when the manifest version, an annotated git tag, and
  a changelog entry all agree.
- A commit hash alone is never a version.
- Never tag a dirty working tree.
- Never classify a change by convenience: classify by contract impact.

## Testing

- A test must be able to fail for the reason it claims. Before trusting a
  new test, break the thing it tests and watch it fail — a test that passes
  under both behaviors pins nothing.
- Never edit a test to make it pass without first stating that the test
  encoded the wrong expectation, and why.
- Every bug fix ships with a regression test in the same change. Write it
  first: watch it fail for the bug's exact mechanism, fix, then watch the
  same test pass unchanged. A fix that never went red is not verified.
- The regression test pins the mechanism, not the symptom: reintroducing
  the defect must make it fail. Verify its bite the same way as any new
  test — break the thing it guards and watch it fail.
- When a defect genuinely cannot be reproduced deterministically in a
  test, state the concrete reason and record the unguarded gap as residual
  risk; never drop the requirement silently.
- MVP scope: tests cover core paths and primary failure modes. Edge-case
  coverage is post-MVP work and never blocks MVP completion.
- Test results, captured output, and logs are ephemeral. Never source-
  control test results.

## Evidence and artifacts

- Screenshots for UI or frontend changes are highly encouraged. Take them
  and use them to audit and assess the implementation before declaring it
  done.
- Screenshots and captured outputs are verification evidence for a moment in
  time: ephemeral. Never commit them.
- Agent artifacts are gitignored. The project's .gitignore must cover the
   artifacts agents produce (screenshots, captured test output, logs, other
   ephemeral verification evidence), the agent instruction files (see
   "Agent artifacts"), and real configuration/environment files (see
   "Configuration and environment"); when a new kind of artifact appears,
   add it to .gitignore before it is ever committed.


## Working style

- Implement iteratively: smallest verifiable increment, verify, repeat.
  Never land a large unverified change in a single step.
- Report honestly: state what was verified by running, what was reasoned
  about, and what was not checked. Do not present an unverified change the
  way you present a tested one.
