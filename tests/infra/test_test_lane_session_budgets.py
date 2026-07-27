"""Test lanes carry wall-clock session budgets so slow regressions fail loudly.

The #2274 profile showed `make test-fast` silently degrading from ~80 s to
28+ minutes (one pathological test plus a starved worker clamp). Each lane
pins `PYTEST_SESSION_BUDGET_SECONDS` so the run itself fails when it blows
its budget instead of quietly crawling; enforcement lives in
``tests/conftest.py``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# make target -> wall-clock budget (seconds) its pytest invocation must carry.
_LANE_BUDGETS = {
    "test-fast": 600,
    "test-ci-unit": 1500,
    "test-ci-slow": 4500,
    "test-ci-nightly": 4800,
}


def _dry_run(target: str) -> str:
    """Return the expanded recipe ``make -n`` would execute for ``target``.

    :param target: Makefile target to expand.
    :returns: The dry-run command text with Makefile variables substituted.
    """
    make = shutil.which("make")
    assert make is not None, "make binary not found despite skipif guard"
    return subprocess.run(  # noqa: S603 — resolved make binary over an allowlisted target
        [make, "-n", target],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.mark.infra
@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.parametrize(("target", "budget"), sorted(_LANE_BUDGETS.items()))
def test_lane_pins_session_budget(target: str, budget: int) -> None:
    """Every pytest invocation in the lane carries its session budget.

    :param target: Makefile test lane under test.
    :param budget: Expected ``PYTEST_SESSION_BUDGET_SECONDS`` value.
    """
    expanded = _dry_run(target)
    pytest_lines = [line for line in expanded.splitlines() if "pytest" in line]
    assert pytest_lines, f"{target}: no pytest invocation found in dry run"
    for line in pytest_lines:
        assert f"PYTEST_SESSION_BUDGET_SECONDS={budget}" in line, (
            f"{target}: pytest invocation missing budget {budget}: {line}"
        )
