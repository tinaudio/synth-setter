---
name: tdd-implementation
description: >-
  Test-driven implementation skill for writing code guided by tests. Use this skill whenever Claude
  is asked to implement, build, create, write, fix, or refactor code — especially when the task
  involves creating new functions, classes, modules, APIs, CLI tools, or features. Also trigger when
  the user says "implement", "build this", "write code for", "create a module", "add a feature",
  "fix this bug", or any request that results in production code being written. This skill ensures
  Claude writes tests FIRST (Red-Green-Refactor), tests behavior not implementation, and produces
  code with high confidence of correctness. For Python, use pytest and mutmut. Even if the user
  doesn't mention testing, this skill should be used for any non-trivial implementation task.
---

# Test-Driven Implementation

This skill governs how Claude writes code. The core principle: **write tests first, then implement**.
This applies to every implementation task — new features, bug fixes, refactors, and API design.

## Why Tests First

Writing tests before implementation forces you to think about the desired behavior, edge cases,
and public API before writing any code. It catches design problems early and produces code that
is testable by construction. Tests written after the fact tend to confirm what the code does
rather than what it should do.

## The Red-Green-Refactor Cycle

For every piece of functionality:

1. **Red**: Write a failing test that describes the desired behavior. Run it. Confirm it fails.
2. **Green**: Write the minimum code to make the test pass. Nothing more.
3. **Refactor**: Clean up both the implementation and the test code while keeping all tests green.

Work in small increments. Write one test at a time, make it pass, then write the next test.
Do not write a large batch of tests upfront and then implement everything at once.

## Planning Phase

Before writing any test or code, think through and briefly document:

1. **What behaviors** does this code need to exhibit? (Not what methods — what behaviors.)
2. **What are the edge cases?** Empty inputs, boundary values, error conditions, concurrent access.
3. **What is the public API?** What will callers see and use?
4. **What are the dependencies?** What needs to be faked or injected?

Share this plan with the user before proceeding.

## Core Testing Principles

These principles are adapted from Google's Tech on the Toilet series and industry best practices,
generalized for open-source development.

### Test Behavior, Not Implementation

This is the single most important testing principle. Tests should verify **what** code does,
not **how** it does it. A test should not break when you refactor internals without changing
observable behavior.

**Signs you're testing implementation:**

- Your test breaks when you rename a private method
- Your test verifies the order of internal function calls
- Your test mocks out collaborators and asserts they were called with specific arguments
  when you could instead verify the final result
- Your test mirrors the production code's logic

**Signs you're testing behavior:**

- Your test would still pass after a complete internal rewrite that preserves the same outputs
- Your test reads like a specification: "given X, when Y, then Z"
- Your test verifies observable outputs: return values, state changes, side effects, errors

### Keep Tests Focused

Each test should verify exactly one behavior or scenario. Do not test multiple scenarios in
a single test. This means:

- If a test fails, you know exactly what broke
- Adding new behaviors doesn't require modifying existing tests
- Each test's setup is minimal and obvious

### Prefer Testing Public APIs

Test through the public interface, not internal implementation classes. If your code has a
public `UserService` that delegates to an internal `UserValidator`, write tests against
`UserService`. The validator's correctness is verified indirectly. If the internal class is
complex enough to need its own tests, that's a sign it might deserve to be a public API itself.

### Don't Overuse Mocks

Prefer real objects and fakes over mocks. Mocks couple your tests to implementation details.
Use mocks primarily when:

- The real dependency is slow (network, database)
- You need to simulate error conditions
- The dependency has side effects you can't reverse in tests

When you do mock, mock at architectural boundaries, not between every pair of collaborating
classes. And never mock types you don't own — wrap them first if you must.

**Use pytest-mock over unittest.mock.** Prefer `pytest-mock` (`mocker` fixture) over
`unittest.mock.patch` decorators. The `mocker` fixture automatically cleans up after each
test, integrates with pytest's fixture system, and produces cleaner test code:

```python
# Prefer: pytest-mock
def test_retries_on_transient_http_error(mocker):
    mock_post = mocker.patch("pipeline.webhooks.requests.post")
    mock_post.side_effect = [ConnectionError("flaky"), Mock(status_code=200)]
    notify_completion(run_id="r-1")
    assert mock_post.call_count == 2

# Avoid: unittest.mock
@patch("pipeline.webhooks.requests.post")
def test_retries_on_transient_http_error(mock_post):
    ...
```

(For rclone/R2 code in this repo, do not patch `subprocess.check_call` —
use the `fake_r2_remote` fixture below.)

### Prefer State Testing Over Interaction Testing

Verify the result (state) rather than verifying that specific methods were called (interaction).
State testing is more resilient to refactoring.

```python
# Prefer: state testing
result = calculator.add(2, 3)
assert result == 5

# Avoid: interaction testing
calculator.add(2, 3)
mock_adder.process.assert_called_once_with(2, 3)
```

### Avoid Logic in Tests

Tests should not contain loops, conditionals, or complex calculations. Use hardcoded expected
values. If the test contains logic that mirrors the production code, a bug in the production
code will be replicated in the test, and both will agree on the wrong answer.

```python
# Bad: logic mirrors production code
expected = sum(price * discount for price in prices)
assert economy(prices, discount) == expected

# Good: hardcoded expected value
assert economy([50, 100, 25], 0.1) == 17.5
```

### Write Descriptive Test Names

Test names should communicate the scenario and expected outcome. A failing test's name alone
should tell you what went wrong. Use the pattern:
`test_<behavior>_<scenario>_<expected_outcome>` or similar.

```python
# Good
def test_withdraw_insufficient_funds_raises_error():
def test_transfer_between_accounts_updates_both_balances():

# Bad
def test_withdraw():
def test_transfer():
```

### Keep Cause and Effect Close

In each test, the action and the assertion should be close together. Avoid long setup sequences
that push the actual test logic far from the assertion. If you need extensive setup, use
fixtures or factory functions, but make sure the test body itself is clear about what's
being tested.

### Cleanly Create Test Data

Use builder patterns or factory functions to create test data. Each test should specify only
the fields relevant to the scenario being tested, with sensible defaults for everything else.
Tests should not rely on default values that are specified by a helper — if a value matters
to the test, set it explicitly in the test.

### Tests as Documentation

Well-written tests serve as living documentation. Each test should be simple enough that an
engineer can quickly grasp what behavior is being verified. Together, a module's tests should
read like a specification of that module's behavior.

### DAMP Over DRY in Tests

Tests should be **D**escriptive **A**nd **M**eaningful **P**hrases. It's OK to have some
repetition in tests if it makes each test self-contained and readable. Don't abstract away
test logic into deep helper hierarchies that require the reader to jump between files.

### Change-Detector Tests Are Harmful

If a test breaks every time you make a trivial change to production code, it's a change-detector
test. These tests verify implementation, not behavior. They slow down development and provide
false confidence. Delete them and write behavior-focused tests instead.

### Increase Test Fidelity by Reducing Mocks

The closer your test environment matches production, the more bugs it catches. Prefer:

1. Real implementations (fastest feedback, highest fidelity)
2. Fakes (in-memory database, local file system)
3. Mocks (lowest fidelity, use sparingly)

#### Project example: `fake_r2_remote` for rclone / R2 tests

In this repo, R2 / rclone code is one of the most-tempting candidates to mock
(`subprocess.check_call` is right there), and one of the worst — a `MagicMock`
on `_rclone_copy` proves nothing about whether the dispatch, upload path,
or rclone invocation actually works.

Prefer the `fake_r2_remote` fixture (`tests/pipeline/conftest.py`). It sets
`RCLONE_CONFIG_R2_TYPE=local` so the real `rclone` binary resolves `r2:` as
the local filesystem under `tmp_path`, then yields that path so the test can
assert state — "did the object materialize at `<root>/<bucket>/<key>`?" —
instead of introspecting a mock's call list.

```python
# Prefer: state-based, real rclone, fake remote
def test_uploads_spec_to_r2(spec, fake_r2_remote: Path):
    upload_spec(spec, "r2://intermediate-data/run-1/input_spec.json")

    landed = fake_r2_remote / "intermediate-data" / "run-1" / "input_spec.json"
    assert json.loads(landed.read_text())["task_name"] == spec.task_name

# Avoid: interaction-based, mocked subprocess
def test_uploads_spec_to_r2(mocker, spec):
    mock_run = mocker.patch("pipeline.spec_io.subprocess.check_call")
    upload_spec(spec, "r2://intermediate-data/run-1/input_spec.json")
    mock_run.assert_called_once()  # proves nothing about the URI, bytes, or rclone flags
```

When a test genuinely needs to simulate an rclone failure, prefer breaking
the real path (e.g. point `RCLONE_CONFIG_R2_TYPE` at a non-existent backend
to force a non-zero exit) over patching `_rclone_copy` — the rclone binary's
real failure semantics are part of what you're testing. See PR #1128 / #1136
for the migration pattern away from `_rclone_copy` mocks.

### Functional Core, Imperative Shell

Structure code so that business logic is pure (no I/O, no side effects) and I/O happens in a
thin outer shell. The pure core is trivially testable with unit tests. The shell needs
integration tests but is thin enough that those tests are manageable.

### Exercise Service Call Contracts in Tests

When your code calls external services, don't just mock the happy path. Write tests that
exercise the contract: correct inputs produce correct outputs, malformed inputs produce
sensible errors, timeouts are handled gracefully.

### Test UI by Interacting Like a User

When testing UI components, render the component and interact with it as a user would (clicking
buttons, filling forms, reading text). Don't test controller internals in isolation from the
rendering layer.

## Blocking Gates (hard stop)

The principles above are how you should write tests. The four gates below are
non-negotiable: if any fails, **stop and fix the suite before declaring the task
done** — do not report success, and mark the row `BLOCK` (not `⚠️`) in the
compliance report. They exist because an all-mock suite can pin the wiring of a
change while proving nothing about the behavior the change was supposed to
deliver: every test stays green even when the real code path is broken.

### Gate 1 — No mock-only coverage of the system under test (BLOCK)

If the change is a bug fix or a behavior change, **at least one committed test
must exercise the real behavior** — not a mock of the thing under test. Apply
this litmus test out loud:

> Would any committed test fail for the reason the bug exists / the behavior
> changed? If every test still passes when the real collaborator is replaced
> by a mock, that's a **BLOCK**.

Patching `subprocess.run` (or any entrypoint) and asserting the argv, or mocking
the collaborator and asserting it was *called*, is necessary-but-not-sufficient:
it pins the wiring, not the behavior. A bug like a dangling HDF5 virtual-dataset
read survives such a suite untouched, because the broken read is mocked out.
Prefer a fake or a real binary that drives the actual path — see the
`fake_r2_remote` example above, which runs the real `rclone` against a fake
remote and asserts the object materialized, instead of introspecting a mock's
call list.

### Gate 2 — No change-detector-only suites (BLOCK)

A suite whose *only* coverage asserts internal call sequences or exact argument
strings — and would break on a behavior-preserving refactor — is a **BLOCK**.
Such assertions are allowed only as **supplementary** contract pins layered on
top of a behavior test; they may never be the sole coverage for a change. See
"Change-Detector Tests Are Harmful" above for why these provide false
confidence.

### Gate 3 — End-to-end CLI coverage when warranted (BLOCK)

When the change adds or modifies a CLI entrypoint, command, or user-invocable
behavior, there must be **at least one test that drives the real entrypoint
end-to-end** with a tiny real config and asserts on the real produced
outputs — not a mocked subprocess. The pattern to imitate lives in synth-setter
`tests/test_train.py`:

- `test_train_fast_dev_run_tiny_model_tiny_data` calls the real `train(cfg)`
  entrypoint with `fast_dev_run=True` on a tiny fixture dataset.
- `test_train_eval_surge_xt` chains the real `train(cfg)` → `evaluate(cfg)`
  entrypoints and asserts on the produced checkpoint, metrics CSVs, and audio
  artifacts.

Imitate that shape: drive the actual entrypoint with tiny real data, chain
stages where the pipeline does, and assert on real artifacts.

**Scope / escape hatch.** If a true end-to-end run is genuinely infeasible in CI
(needs a GPU, the cluster, or secrets), the test may be marked/skipped with an
explicit reason (`@pytest.mark.slow`, `pytest.skip("needs GPU: <why>")`). But
the BLOCK still requires **either** the e2e test **or** an explicit, justified
unchecked `[ ]` item carrying the exact manual command to run it. Silent
omission is never acceptable.

### Gate 4 — No stub-fabricated assertions (BLOCK)

A test can exercise the real producer (a resolver, downloader, loader, builder)
and *still* prove nothing if its only assertion is on an artifact the **fake
itself fabricated** — that a returned path exists, an object has some type, or a
stub file is present — without ever feeding that artifact into its real consumer.
The assertion is circular: it confirms the fake did what the test told it to, not
that the producer's output is *usable*. When that is the sole coverage for a
behavior change, it is a **BLOCK**. The fix: pass the produced artifact through
its real consumer and assert on the downstream effect.

This is distinct from Gate 1 — there the system under test is mocked; here the
SUT runs for real, but the assertion targets the fake's own output instead of the
real downstream contract, so the suite stays green while the consumer is broken.

> Concrete miss: a `${wandb:...}` checkpoint-resolver test injected a fake `wandb`
> whose `download()` wrote the bytes `b"weights"` into `model.ckpt`, then asserted
> only `Path(cfg.ckpt_path).is_file()`. Those bytes are not a loadable checkpoint
> and are never loaded — the test passed while the real `eval` / `train` paths
> failed to load the artifact. An adequate test writes a *real* checkpoint into
> the fake's download dir and runs `evaluate(cfg)` so the model actually loads
> those weights and inference runs (which lands in Gate 3 territory).

## Unit Test Properties

Every unit test should exhibit these properties:

01. **Focused**: Narrow scope, validating individual behaviors
02. **Understandable**: An engineer can quickly grasp its purpose
03. **Maintainable**: Easy to update as code evolves
04. **Informative**: Clear failure messages that diagnose what went wrong
05. **Fast**: Executes quickly for continuous feedback
06. **Deterministic**: Always passes or fails the same way — no flakiness
07. **Resilient**: Doesn't break on unrelated code changes
08. **Isolated**: Outcome doesn't depend on other tests or test order
09. **Local**: No interaction with the external world (network, filesystem in unit tests)
10. **Without sleep**: No sleep statements — use deterministic synchronization

## Code Coverage Philosophy

Track coverage — its existence matters. But don't enforce rigid percentage targets.

- 60% is acceptable, 75% is commendable, 90% is exemplary
- Coverage measures lines executed, not behaviors verified — high coverage doesn't guarantee
  high-quality tests
- When coverage becomes a target instead of a measure, developers write tests to hit the number
  rather than to validate behavior
- Use judgment: some code deserves more coverage than others
- Favor mutation testing over raw coverage metrics (see below)

## PR Review Mode — Codecov-Guided Path Audit

When this skill is invoked by `repo-review-full` for an open PR, inspect the PR's
Codecov report before writing the review. Find the Codecov link in the PR checks,
conversation, body, or status details; follow the link far enough to identify
uncovered changed lines and the file/line ranges they belong to. If the report is
not available, say so in the review and continue with the diff and tests rather
than inventing coverage data.

Coverage is evidence for a path audit, not a percentage gate. For each uncovered
changed range, determine the behavior it represents: branch outcomes, validation
and error handling, boundary conditions, state transitions, persistence or
serialization, external-service contracts, concurrency, or numerical/data-shape
handling. Check existing tests before reporting a gap; a line can be covered
indirectly by a higher-fidelity behavior or integration test.

Classify an uncovered path as `BLOCK` when all of the following are true:

1. The path is reachable in normal use and affects correctness, data integrity,
   safety, or a user-visible result.
2. The PR changes or introduces the path, or makes its behavior materially
   different.
3. No test exercises the behavior, including an appropriate error, boundary, or
   integration case.

Use `WARN` for uncovered paths with lower correctness risk, testability tradeoffs,
or an existing test that covers the behavior but not the exact line. Do not block
solely on a low file/patch percentage, generated code, wiring-only code, or a
deliberately unreachable defensive branch. A single missing assertion is not a
coverage finding unless it leaves a correctness-sensitive behavior unverified.

Every Codecov finding must include the Codecov report link, the file and line,
the uncovered behavior or path, why it affects correctness (for `BLOCK`), and the
smallest behavior-focused test that would exercise it. Use the normal per-agent
report format so `repo-review-full` can post it inline:

```text
BLOCK: path/to/module.py:42 — Codecov marks the error branch uncovered; malformed
input reaches the success path without this validation, so the persisted result
can be incorrect. Add a test through the public entrypoint that submits malformed
input and asserts the rejected result. [Codecov](https://app.codecov.io/...)
```

If Codecov reports a missed line but the line is only an implementation detail of
a behavior already tested, record it under `What looks good` or omit it. The
review should identify missing behavior, not prescribe tests that merely execute
lines.

## Mutation Testing with mutmut (Python)

After writing tests, verify their effectiveness with mutation testing. Mutmut modifies your
source code (introduces mutations) and checks if your tests catch them. Surviving mutants
indicate weak tests.

```bash
# Run mutation testing
mutmut run --paths-to-mutate=src/

# View surviving mutants
mutmut results

# Inspect a specific mutant
mutmut show <id>
```

When mutants survive:

- Add tests that cover the missing behavior
- Don't write tests that just kill mutants mechanically — understand what behavior is untested
- Some mutants are equivalent (the mutation doesn't change behavior) — it's OK to skip these

For details on running mutmut and interpreting results, read `references/mutation-testing.md`.

## TotT Episode Catalog

For the full catalog of all 107+ Google Tech on the Toilet episodes — categorized by topic,
annotated with one-line takeaways, and with Google-specific episodes visually separated — see
`references/tott-catalog.md`. Consult it when encountering an unusual testing situation to see
if a relevant TotT episode exists.

## Python-Specific Guidelines

For Python projects, use **pytest** as the test framework.

Read `references/python-testing.md` for detailed pytest patterns, fixture usage, parametrize
patterns, and project structure conventions.

## Implementation Workflow

When given an implementation task:

1. **Plan**: Identify behaviors, edge cases, public API, dependencies
2. **Set up test infrastructure**: Create test files, install pytest, configure mutmut
3. **Red**: Write the first failing test for the simplest behavior
4. **Green**: Write minimum code to pass
5. **Refactor**: Clean up, extract helpers if needed
6. **Repeat**: Next behavior, next test, until all planned behaviors are covered
7. **Edge cases**: Add tests for error handling, boundary conditions
8. **Mutation testing**: Run mutmut, kill surviving mutants with additional behavior tests
9. **Review**: Verify tests read as documentation, no change-detectors, no logic in tests

## Safe Defaults and Defensive Design

When implementing code that takes configuration or flags:

- Default to the safest option (e.g., dry-run mode on by default)
- Require explicit opt-in for destructive operations
- Require environment-specific settings to be explicitly provided

## Code Organization for Testability

- Arrange code to communicate data flow: declare variables close to first use, group related
  operations, make the data flow through a function read linearly
- Separate pure logic from I/O (functional core, imperative shell)
- Inject dependencies rather than creating them internally
- Prefer composition over inheritance for testability

## TotT Catalog

A complete catalog of all published Google Tech on the Toilet episodes is in
`references/tott-catalog.md`, organized by topic with links and Google-specificity markers.
Consult it when facing an unusual testing situation — there may be a relevant episode.

## Post-Implementation TDD Compliance Report

After completing any implementation task, produce a short compliance report. This helps
the user see how the TDD principles were applied and identify any gaps. The report has
three parts: a blocking-gates table, a checklist table, and a brief narrative.

### Blocking Gates Table

These four rows are hard gates (see "Blocking Gates (hard stop)" above). They are
scored `PASS` or `BLOCK`, except Gate 3 (`N/A` when the change touches no
CLI/entrypoint or other user-invocable behavior) and Gate 4 (`N/A` when the change's
tests involve no fake/stub-produced artifact, so there is nothing to drive through a
real consumer). **If any applicable row is `BLOCK`, the task is not done**: fix the
suite and re-run before reporting success. Gate 3 may also be satisfied by an explicit,
justified unchecked `[ ]` item carrying the exact manual command when a real e2e run is
infeasible in CI.

| Gate | Rule                                                                             | Status         | Evidence                             |
| ---- | -------------------------------------------------------------------------------- | -------------- | ------------------------------------ |
| G1   | A committed test exercises the real behavior (not a mock of the SUT)             | PASS/BLOCK     | Which test; what real path it drives |
| G2   | Coverage is not change-detector-only (call/argv asserts are supplementary)       | PASS/BLOCK     | Behavior test backing the assertions |
| G3   | Real-entrypoint e2e test when warranted (or justified `[ ]` with manual command) | PASS/BLOCK/N/A | The e2e test, or why N/A             |
| G4   | No stub-fabricated assertions (produced artifact is driven through its real consumer) | PASS/BLOCK/N/A | The consumer test, or why N/A        |

### Checklist Table

| #   | Principle                                                | Applied? | Evidence   |
| --- | -------------------------------------------------------- | -------- | ---------- |
| 1   | Tests written before implementation (Red-Green-Refactor) | ✅/⚠️/❌ | Brief note |
| 2   | Tests verify behavior, not implementation                | ✅/⚠️/❌ | Brief note |
| 3   | Each test is focused on one scenario                     | ✅/⚠️/❌ | Brief note |
| 4   | Tests use public APIs, not internals                     | ✅/⚠️/❌ | Brief note |
| 5   | No overuse of mocks; fakes preferred                     | ✅/⚠️/❌ | Brief note |
| 6   | State testing preferred over interaction testing         | ✅/⚠️/❌ | Brief note |
| 7   | No logic in tests (hardcoded expected values)            | ✅/⚠️/❌ | Brief note |
| 8   | Descriptive test names                                   | ✅/⚠️/❌ | Brief note |
| 9   | Cause and effect kept close                              | ✅/⚠️/❌ | Brief note |
| 10  | Clean test data (builders/factories)                     | ✅/⚠️/❌ | Brief note |
| 11  | Tests serve as documentation                             | ✅/⚠️/❌ | Brief note |
| 12  | DAMP over DRY in test code                               | ✅/⚠️/❌ | Brief note |
| 13  | No change-detector tests                                 | ✅/⚠️/❌ | Brief note |
| 14  | Functional core / imperative shell separation            | ✅/⚠️/❌ | Brief note |
| 15  | Safe defaults for configuration                          | ✅/⚠️/❌ | Brief note |
| 16  | Mutation testing run (Python: mutmut)                    | ✅/⚠️/❌ | Brief note |

The blocking-gates table is scored `PASS`/`BLOCK` (Gates 3 and 4 may be `N/A`). For the
checklist table, use ✅ for fully applied, ⚠️ for partially applied with explanation,
❌ for not applied (with justification — some principles may not be relevant to a
given task).

### Narrative Summary

After the table, write 2-4 sentences covering:

- Which principles had the biggest impact on this implementation
- Any principles that were deliberately skipped and why
- Mutation testing results if applicable (mutants killed / total, any notable survivors)
- Suggestions for the user on areas that could benefit from additional testing
