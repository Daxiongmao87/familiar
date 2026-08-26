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
  ephemeral verification evidence); when a new kind of artifact appears,
  add it to .gitignore before it is ever committed.

## Working style

- Implement iteratively: smallest verifiable increment, verify, repeat.
  Never land a large unverified change in a single step.
- Report honestly: state what was verified by running, what was reasoned
  about, and what was not checked. Do not present an unverified change the
  way you present a tested one.
