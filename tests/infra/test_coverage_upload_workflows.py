"""Coverage-producing test jobs upload reports even when pytest fails."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml
from workflow_fixtures import load_workflow

_UPLOAD_STEP_NAME = "Upload coverage to Codecov"
_EXPECTED_CONDITION = "always() && hashFiles('coverage.xml') != ''"


def _job_steps(project_root: Path, job_name: str) -> list[dict[str, object]]:
    """Return the steps for ``job_name`` in the main unit-test workflow.

    :param project_root: Repo root supplied by the infra test fixtures.
    :param job_name: Workflow job whose steps to return.
    :returns: The job's ordered step mappings.
    """
    workflow = load_workflow(project_root, "test.yml")
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    return cast(list[dict[str, object]], jobs[job_name]["steps"])


@pytest.mark.infra
@pytest.mark.parametrize("job_name", ["run_tests_ubuntu", "run_tests_macos"])
def test_unit_test_job_uploads_partial_coverage_after_pytest_failure(
    project_root: Path, job_name: str
) -> None:
    """A failing suite still publishes its current-commit coverage instead of carryforward data.

    :param project_root: Repo root supplied by the infra test fixtures.
    :param job_name: Unit-test workflow job under test.
    """
    upload_step = next(
        (
            step
            for step in _job_steps(project_root, job_name)
            if step.get("name") == _UPLOAD_STEP_NAME
        ),
        None,
    )

    assert upload_step is not None, f"Step {_UPLOAD_STEP_NAME!r} not found in job {job_name!r}"
    assert upload_step.get("if") == _EXPECTED_CONDITION


@pytest.mark.infra
def test_dataset_reopen_job_uploads_production_path_coverage(project_root: Path) -> None:
    """The real-R2 lane contributes reopen coverage skipped by unit-test runners.

    :param project_root: Repo root supplied by the infra test fixtures.
    """
    workflow = load_workflow(project_root, "test-dataset-reopen.yml")
    job = cast(dict[str, dict[str, object]], workflow["jobs"])["reopen-production-e2e"]
    steps = cast(list[dict[str, object]], job["steps"])
    test_step = next(
        step for step in steps if step.get("name") == "Run dataset reopen production E2E"
    )
    upload_step = next(step for step in steps if step.get("name") == _UPLOAD_STEP_NAME)

    assert "--cov=src --cov-branch --cov-report=xml" in cast(str, test_step["run"])
    assert upload_step == {
        "name": _UPLOAD_STEP_NAME,
        "if": _EXPECTED_CONDITION,
        "uses": "./.github/actions/upload-coverage",
        "with": {"flags": "integration-r2-reopen", "token": "${{ secrets.CODECOV_TOKEN }}"},
    }
    codecov_config = cast(
        dict[str, object], yaml.safe_load((project_root / "codecov.yml").read_text())
    )
    flag_management = cast(dict[str, object], codecov_config["flag_management"])
    flags = cast(list[dict[str, object]], flag_management["individual_flags"])
    assert any(flag.get("name") == "integration-r2-reopen" for flag in flags)
