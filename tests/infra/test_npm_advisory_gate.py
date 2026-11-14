"""A scheduled lane reports npm advisories in every checked-in lockfile."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

AUDIT_JOB = "npm-advisories"


def _audited_dirs(project_root: Path) -> set[str]:
    """Return the directories the nightly audit job covers.

    :param project_root: Repository root.
    :returns: Values of the job's ``lockfile-dir`` matrix axis.
    """
    jobs = cast(dict[str, object], load_workflow(project_root, "nightly.yml")["jobs"])
    job = cast(dict[str, object], jobs[AUDIT_JOB])
    matrix = cast(dict[str, list[str]], cast(dict[str, object], job["strategy"])["matrix"])
    return set(matrix["lockfile-dir"])


@pytest.mark.infra
def test_nightly_audits_the_lockfile_independently_of_the_full_suite(project_root: Path) -> None:
    """The audit runs on its own job, so the capped suite cannot mask it.

    :param project_root: Session fixture from ``tests/infra/conftest.py``.
    """
    jobs = cast(dict[str, object], load_workflow(project_root, "nightly.yml")["jobs"])
    job = cast(dict[str, object], jobs[AUDIT_JOB])

    assert "needs" not in job
    steps = cast(list[dict[str, object]], job["steps"])
    audits = [step for step in steps if cast(str, step.get("run") or "").startswith("npm audit")]
    assert len(audits) == 1
    assert "--audit-level=high" in cast(str, audits[0]["run"])


@pytest.mark.infra
def test_audit_covers_every_checked_in_lockfile(project_root: Path) -> None:
    """A lockfile added later stays unaudited until its directory joins the matrix.

    :param project_root: Session fixture from ``tests/infra/conftest.py``.
    """
    lockfiles = [
        path
        for path in project_root.rglob("package-lock.json")
        if "node_modules" not in path.parts
    ]
    assert lockfiles, "no checked-in npm lockfiles found"

    covered = {str(path.parent.relative_to(project_root)) for path in lockfiles}
    assert covered == _audited_dirs(project_root)


@pytest.mark.infra
def test_no_merge_lane_audits_the_lockfile(project_root: Path) -> None:
    """``npm audit`` reaches the registry, so no merge-gating lane may depend on it.

    :param project_root: Session fixture from ``tests/infra/conftest.py``.
    """
    workflows = project_root / ".github/workflows"
    auditing = {
        path.name
        for path in sorted(workflows.glob("*.y*ml"))
        if "npm audit" in path.read_text(encoding="utf-8")
    }

    assert auditing == {"nightly.yml"}
