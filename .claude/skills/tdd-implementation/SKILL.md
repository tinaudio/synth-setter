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
def test_upload_retries_on_failure(mocker):
    mock_rclone = mocker.patch("pipeline.storage.subprocess.run")
    mock_rclone.side_effect = [subprocess.CalledProcessError(1, "rclone"), None]
    upload_shard(shard_path)
    assert mock_rclone.call_count == 2

# Avoid: unittest.mock
@patch("pipeline.storage.subprocess.run")
def test_upload_retries_on_failure(mock_rclone):
    ...
```

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
two parts: a checklist table and a brief narrative.

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

Use ✅ for fully applied, ⚠️ for partially applied with explanation, ❌ for not applied
(with justification — some principles may not be relevant to a given task).

### Narrative Summary

After the table, write 2-4 sentences covering:

- Which principles had the biggest impact on this implementation
- Any principles that were deliberately skipped and why
- Mutation testing results if applicable (mutants killed / total, any notable survivors)
- Suggestions for the user on areas that could benefit from additional testing
