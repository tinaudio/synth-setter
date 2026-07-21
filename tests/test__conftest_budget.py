"""Tests for the session wall-clock budget enforced by tests/conftest.py."""

import os
import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import pytest

from tests.conftest import (
    _session_budget_seconds,
    pytest_sessionfinish,
    pytest_sessionstart,
)

_BUDGET_ENV = "PYTEST_SESSION_BUDGET_SECONDS"


class TestSessionBudgetSeconds:
    """Parsing of the budget env var: fail-open on anything not a positive number."""

    def test_unset_env_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No env var → no budget.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.delenv(_BUDGET_ENV, raising=False)
        assert _session_budget_seconds() is None

    def test_positive_value_returns_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A positive number parses as the budget in seconds.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv(_BUDGET_ENV, "600")
        assert _session_budget_seconds() == 600.0

    def test_non_numeric_value_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A malformed value is ignored, not fatal.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv(_BUDGET_ENV, "ten minutes")
        assert _session_budget_seconds() is None

    def test_zero_or_negative_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zero and negative budgets disable enforcement instead of always failing.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv(_BUDGET_ENV, "0")
        assert _session_budget_seconds() is None
        monkeypatch.setenv(_BUDGET_ENV, "-5")
        assert _session_budget_seconds() is None


def _fake_session(*, worker: bool = False) -> pytest.Session:
    """Build a minimal session double for the budget hooks.

    :param worker: Whether to mimic an xdist worker (has ``workerinput``).
    :returns: Object with the attributes the hooks touch, cast to ``Session``.
    """
    config = SimpleNamespace()
    if worker:
        config.workerinput = {}
    return cast("pytest.Session", SimpleNamespace(config=config, exitstatus=0))


class TestSessionBudgetEnforcement:
    """The sessionfinish hook flips a passing exit only when the budget is blown."""

    def test_over_budget_passing_session_flips_exitstatus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Elapsed > budget on a green session → exitstatus becomes nonzero.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv(_BUDGET_ENV, "10")
        clock = iter([100.0, 200.0])
        monkeypatch.setattr("tests.conftest.time.monotonic", lambda: next(clock))
        session = _fake_session()
        pytest_sessionstart(session)
        pytest_sessionfinish(session, exitstatus=0)
        assert session.exitstatus == 1

    def test_under_budget_session_keeps_exitstatus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Elapsed < budget leaves the session untouched.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv(_BUDGET_ENV, "1000")
        clock = iter([100.0, 200.0])
        monkeypatch.setattr("tests.conftest.time.monotonic", lambda: next(clock))
        session = _fake_session()
        pytest_sessionstart(session)
        pytest_sessionfinish(session, exitstatus=0)
        assert session.exitstatus == 0

    def test_over_budget_failing_session_preserves_real_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blown budget never rewrites an already-failing exit status.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv(_BUDGET_ENV, "10")
        clock = iter([100.0, 200.0])
        monkeypatch.setattr("tests.conftest.time.monotonic", lambda: next(clock))
        session = _fake_session()
        session.exitstatus = 2
        pytest_sessionstart(session)
        pytest_sessionfinish(session, exitstatus=2)
        assert session.exitstatus == 2

    def test_worker_session_never_enforces_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Xdist workers skip enforcement — only the controller owns the wall clock.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setenv(_BUDGET_ENV, "10")
        clock = iter([100.0, 200.0])
        monkeypatch.setattr("tests.conftest.time.monotonic", lambda: next(clock))
        session = _fake_session(worker=True)
        pytest_sessionstart(session)
        pytest_sessionfinish(session, exitstatus=0)
        assert session.exitstatus == 0

    def test_no_budget_env_never_enforces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the env var the hooks are inert.

        :param monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.delenv(_BUDGET_ENV, raising=False)
        clock = iter([0.0, 1_000_000.0])
        monkeypatch.setattr("tests.conftest.time.monotonic", lambda: next(clock))
        session = _fake_session()
        pytest_sessionstart(session)
        pytest_sessionfinish(session, exitstatus=0)
        assert session.exitstatus == 0


@pytest.mark.slow
def test_real_pytest_run_over_budget_exits_nonzero_with_message() -> None:
    """End-to-end: a real pytest run through the real conftest fails a blown budget."""
    env = {**os.environ, _BUDGET_ENV: "0.000001"}
    result = subprocess.run(  # noqa: S603 — fixed argv, our own test suite
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test__conftest_cpu.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "session budget" in (result.stdout + result.stderr)
