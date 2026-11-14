"""Tests for bounded retry of live-registry CLI invocations."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

import pytest

from tests.helpers.transient_cli import (
    is_transient_registry_failure,
    run_until_stable,
    skip_on_transient_registry_failure,
)

_FETCH_RESET = "TypeError: fetch failed\n  [cause]: Error: read ECONNRESET"
_LOCK_MISMATCH = "artifact lock mismatch for asb2m10/dexed@0.9.8"


def _completed(
    returncode: int, stderr: str = "", stdout: str = ""
) -> subprocess.CompletedProcess[str]:
    """Build a finished process result with the given status and output.

    :param returncode: Process exit status.
    :param stderr: Captured standard error.
    :param stdout: Captured standard output.
    :returns: Completed process standing in for a CLI invocation.
    """
    return subprocess.CompletedProcess(
        args=["studiorack"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _replaying(
    *results: subprocess.CompletedProcess[str],
) -> tuple[Callable[[], subprocess.CompletedProcess[str]], list[int]]:
    """Build a run callable replaying fixed results and a list recording its calls.

    :param *results: Results to return in order; the last one repeats.
    :returns: The run callable and the list it appends one entry per call to.
    """
    calls: list[int] = []

    def run() -> subprocess.CompletedProcess[str]:
        calls.append(len(calls))
        return results[min(len(calls) - 1, len(results) - 1)]

    return run, calls


def test_is_transient_registry_failure_marker_on_successful_run_is_not_transient() -> None:
    """A zero exit is never transient, even when its output mentions a reset."""
    assert is_transient_registry_failure(_completed(0, stdout=_FETCH_RESET)) is False


def test_is_transient_registry_failure_lock_mismatch_is_not_transient() -> None:
    """A deterministic lock rejection is a real failure, not a network fault."""
    assert is_transient_registry_failure(_completed(1, stderr=_LOCK_MISMATCH)) is False


def test_run_until_stable_transient_failure_then_success_returns_the_success() -> None:
    """A reset on the first attempt is retried and the successful retry is returned."""
    run, calls = _replaying(
        _completed(1, stderr=_FETCH_RESET), _completed(0, stdout='"installed": true')
    )

    completed = run_until_stable(run, attempts=3, backoff_s=0.0)

    assert completed.returncode == 0
    assert len(calls) == 2


def test_run_until_stable_persistent_reset_stops_at_the_attempt_budget() -> None:
    """A sustained outage is bounded by the attempt budget and returns the last result."""
    run, calls = _replaying(_completed(1, stderr=_FETCH_RESET))

    completed = run_until_stable(run, attempts=3, backoff_s=0.0)

    assert completed.returncode == 1
    assert len(calls) == 3


def test_run_until_stable_deterministic_failure_is_not_retried() -> None:
    """A lock rejection is returned on its first attempt so the test still fails fast."""
    run, calls = _replaying(_completed(1, stderr=_LOCK_MISMATCH))

    completed = run_until_stable(run, attempts=3, backoff_s=0.0)

    assert completed.returncode == 1
    assert len(calls) == 1


def test_run_until_stable_zero_attempts_rejects_the_budget() -> None:
    """A non-positive budget is a caller error rather than a silent no-run."""
    run, _ = _replaying(_completed(0))

    with pytest.raises(ValueError, match="attempts must be at least one"):
        run_until_stable(run, attempts=0, backoff_s=0.0)


def test_skip_on_transient_registry_failure_exhausted_reset_skips() -> None:
    """An exhausted outage is reported as an external-service skip, not a failure."""
    with pytest.raises(pytest.skip.Exception, match="registry unreachable"):
        skip_on_transient_registry_failure(_completed(1, stderr=_FETCH_RESET))


def test_skip_on_transient_registry_failure_lock_mismatch_continues() -> None:
    """A deterministic rejection reaches the test's own assertions."""
    skip_on_transient_registry_failure(_completed(1, stderr=_LOCK_MISMATCH))
