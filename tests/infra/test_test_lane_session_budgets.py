"""Test lanes carry wall-clock session budgets so slow regressions fail loudly.

The #2274 profile showed `make test-fast` silently degrading from ~80 s to
28+ minutes (one pathological test plus a starved worker clamp). Each lane
pins `PYTEST_SESSION_BUDGET_SECONDS` so the run itself fails when it blows
its budget instead of quietly crawling; enforcement lives in
``tests/conftest.py``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

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


@pytest.mark.infra
def test_ci_fast_tier_kill_timeout_leaves_the_session_budget_room_to_report() -> None:
    """CI's outer ``timeout`` is a hang backstop, not a second budget.

    The in-process check fires at ``pytest_sessionfinish``, while the outer
    timer also covers ``make`` startup, interpreter boot and collection. Given
    the same number of seconds the ``SIGKILL`` always lands first, costing the
    ``session budget exceeded`` diagnostic, the test report and coverage, and
    leaving only exit 137 — #3344.
    """
    workflow = yaml.safe_load(
        (_PROJECT_ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["run_tests_ubuntu"]["steps"]
    run = next(step["run"] for step in steps if "fast-test-budget" in step.get("run", ""))

    timeout_arg = re.search(r'timeout --signal=KILL "([^"]+)s"', run)
    assert timeout_arg is not None, f"fast-tier step no longer wraps make in timeout:\n{run}"
    multiplier = re.fullmatch(r"\$\(\(budget_seconds \* (\d+)\)\)", timeout_arg.group(1))
    assert multiplier is not None, (
        f"kill timeout must scale from the lane budget, got {timeout_arg.group(1)!r}"
    )
    assert int(multiplier.group(1)) > 1, "kill timeout must exceed the in-process session budget"


def _jobs_running_budgeted_lanes() -> list[tuple[str, str, str, int]]:
    """Return every workflow job that invokes a budgeted lane.

    :returns: ``(workflow file name, job id, make target, lane budget seconds)``
        for each job whose steps run one of :data:`_LANE_BUDGETS`' targets.
    """
    found: list[tuple[str, str, str, int]] = []
    for path in sorted((_PROJECT_ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, job in (workflow.get("jobs") or {}).items():
            runs = " ".join(step.get("run", "") for step in job.get("steps") or [])
            for target, budget in _LANE_BUDGETS.items():
                if re.search(rf"make\s+{re.escape(target)}(?!\S)", runs):
                    found.append((path.name, job_id, target, budget))
    return found


@pytest.mark.infra
def test_budgeted_lane_jobs_cap_above_their_own_session_budget() -> None:
    """A job's ``timeout-minutes`` must outlast the lane budget it runs.

    Below it the runner cancels the job before ``pytest_sessionfinish`` can
    report, so the lane produces no verdict at all — no ``FAILED`` lines, no
    summary, only ``cancelled``. The nightly spent six nights that way — #3653.
    """
    jobs = _jobs_running_budgeted_lanes()
    assert jobs, "no workflow job invokes a budgeted lane — the scan is broken"

    for workflow_name, job_id, target, budget in jobs:
        workflow = yaml.safe_load(
            (_PROJECT_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
        )
        cap = workflow["jobs"][job_id].get("timeout-minutes")
        assert cap is not None, f"{workflow_name}:{job_id} runs {target} with no timeout-minutes"
        assert cap * 60 > budget, (
            f"{workflow_name}:{job_id} caps at {cap}m but {target} budgets "
            f"{budget}s ({budget / 60:.0f}m) — the job dies before it can report"
        )
