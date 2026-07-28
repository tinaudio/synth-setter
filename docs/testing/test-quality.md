# Test quality

What a test in this repo must earn its place by doing, and the failure modes
audited out of the suite. Read this before adding a test file.

Companion docs: [mutmut.md](mutmut.md) (mutation testing), the `ml-test` skill
(model/training-specific patterns).

## The bar

A test earns its place when **it can fail for exactly one interesting reason**,
and that reason is a behavior someone depends on.

Before writing one, answer: _what production bug does this catch?_ If the answer
is "a config value changed" or "someone renamed a function", the test is a change
detector — it will cost more in maintenance than it returns in signal.

## Anti-patterns

### 1. Testing the test scaffolding

The sharpest smell: a test whose subject is a helper defined in the same test
file. The helper has no production callers, so a failure means the test file
disagrees with itself.

```python
# tests/infra/test_pr_review_model_routing.py — removed
def _assert_process_terminated(pid: int, *, timeout: float = 1) -> None:
    """Local assertion helper used by other tests in this file."""
    ...


def test_assert_process_terminated_live_pid_fails() -> None:
    """Reject a descendant that is still executing."""
    child_pid = os.fork()  # forks a real process, in the fast suite,
    if child_pid == 0:     # to test an assertion helper
        time.sleep(30)
        os._exit(0)
    ...
```

If a test helper is complex enough to need tests, it is production code — move it
to `src/` or `tests/helpers/` and test it there. Otherwise let its consumers cover it.

### 2. Freezing config into a literal

A tuple in a test that mirrors a checked-in YAML/JSON file asserts only that two
files match. Every intentional config edit must be made twice, and the test
catches no behavior.

```python
# Change detector: mirrors agent config, breaks on every deliberate model bump.
_ROLE_MODELS = {
    "pr-review-worker-deep": {"claude": ("sonnet", "high"), ...},
}
```

Assert the **property** that matters instead — every role resolves to a
registered model, effort is within the allowed set — and read the config as data.

### 3. The allowlist treadmill

When a comparison test grows an exclusion list that every subsequent PR appends
to, the test has inverted: the allowlist is now the specification, and the
assertion is whatever is left.

The baseline-config comparator's `ACCEPTED_DIFFS` allowlist reached 24 entries
citing 10 PRs, and excluded the entire `training`, `evaluation`, and `r2`
subtrees — so it no longer guarded the thing it was written to guard.

Prefer a **golden file** the PR author updates in the same diff: intentional
changes show up as a reviewable snapshot change, not an allowlist entry.

### 4. Asserting a mock returns what you told it to

```python
loader.return_value = sentinel
assert run(loader) == sentinel  # asserts unittest.mock works
```

Assert the effect on the system under test, not the round-trip.

Verifying a call **is** correct when the call is the contract — a CLI translating
`argv` into keyword arguments is legitimately tested with `assert_called_once_with`.
The distinction is whether the boundary is the behavior.

### 5. Weak assertions on rich results

`assert result is not None` / `assert isinstance(x, dict)` pass for almost every
wrong answer. Assert values, shapes, dtypes, and ranges.

## Patterns worth copying

These are real tests in this repo. Model new work on them.

**Retry semantics + secret redaction + log contract in one focused test** —
`tests/pipeline/data/test_lance_materialize.py`:

```python
def test_resolve_txid_version_transient_version_list_retries_without_leaking(...):
    ...
    assert resolved_version == 1
    assert attempts == 2                      # retried exactly once
    assert [(l["operation"], l["attempt"], l["max_attempts"]) for l in retry_logs] == [
        ("version_list", 1, 3)
    ]
    assert "top-secret" not in repr(logs)     # no credential leak on the failure path
```

Every assertion can fail independently and each names a real requirement.

**Numeric behavior with real bounds** — `tests/models/test_cnn.py`:

```python
def test_log_mel_frontend_predictions_stay_in_normalized_parameter_range() -> None:
    predictions = _log_mel_model()(torch.randn(16, 4_410))
    assert torch.all((0 <= predictions) & (predictions <= 1))
```

Use `torch.testing.assert_close` for float comparison; assert ranges and shapes,
never bare truthiness.

**Crash-only smoke tests are fine** when the entrypoint raising _is_ the
assertion — `train(cfg)` under `fast_dev_run` needs no trailing `assert`. Say so
in the docstring so the next reader doesn't "fix" it.

**Invariants over enumeration** — `tests/_meta/` asserts structural rules about
the suite itself (e.g. entrypoint modules must not import Hydra initializers)
rather than listing files.

## Conventions

- Name: `test_<what>_<condition>_<expected>`.
- One reason to fail per test; use `@pytest.mark.parametrize` for cases that
  differ only in data.
- `@pytest.mark.slow` for slow tests; keep the `make test-fast` loop fast.
- Prefer real objects over mocks. Mock at process/network boundaries only.
- Docstring states the contract, not the mechanics.
