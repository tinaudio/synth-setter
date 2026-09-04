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
    "test-fast": 120,
    "test-medium": 600,
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
    expanded = _dry_run(target).replace("\\\n", " ")
    pytest_lines = [line for line in expanded.splitlines() if "pytest" in line]
    assert pytest_lines, f"{target}: no pytest invocation found in dry run"
    for line in pytest_lines:
        assert f"PYTEST_SESSION_BUDGET_SECONDS={budget}" in line, (
            f"{target}: pytest invocation missing budget {budget}: {line}"
        )


@pytest.mark.infra
@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_fast_lane_collects_only_curated_test_paths() -> None:
    """The strict lane avoids importing the full medium-suite test tree."""
    expanded = _dry_run("test-fast")
    selected_paths = {token for token in expanded.split() if token.startswith("tests/")}

    assert "tests/_meta" in selected_paths
    assert "tests/data/vst/test_param_spec.py" in selected_paths
    assert "tests/integration/test_parallel_shard_dispatch.py" in selected_paths
    assert "tests/models/test_cnn.py" in selected_paths
    assert '-m "not slow and not gpu and not mps and not requires_vst and not infra"' in expanded
    assert " tests " not in f" {expanded} "


@pytest.mark.infra
@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_medium_lane_preserves_previous_full_non_slow_selection() -> None:
    """The medium lane selects non-slow CPU tests without explicit test paths."""
    expanded = _dry_run("test-medium")

    assert '-m "not slow and not gpu and not mps and not requires_vst"' in expanded
    assert not any(token.startswith("tests/") for token in expanded.split())


@pytest.mark.infra
@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_fast_budget_target_prints_two_minute_limit() -> None:
    """The workflow-readable fast budget matches the lane configuration."""
    make = shutil.which("make")
    assert make is not None

    result = subprocess.run(  # noqa: S603 — resolved make binary and allowlisted target
        [make, "--no-print-directory", "-s", "fast-test-budget"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == "120\n"
