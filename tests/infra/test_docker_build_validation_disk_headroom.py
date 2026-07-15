"""The Docker build jobs reclaim enough runner disk to export the image.

`docker-validate`'s `load: true` export writes the image twice more on top of the BuildKit cache,
so the runner's stock toolchain space is what keeps it off `no space left on device`. Both jobs pin
identical inline copies of the reclaim — a local composite would resolve against their foreign
checkout. See #1930.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

WORKFLOW_FILENAME = "docker-build-validation.yml"
FREE_DISK_STEP_NAME = "Free runner disk space"
BUILD_ACTION = "docker/build-push-action@"
BUILD_JOB_NAMES = ("docker-push", "docker-validate")

# Large preinstalled toolchains, unused by a Docker build; freeing them is what
# fits the export (#1930).
RECLAIMED_TOOLCHAIN_PATHS = (
    "/opt/az",
    "/opt/ghc",
    "/opt/hostedtoolcache/CodeQL",
    "/usr/local/.ghcup",
    "/usr/local/lib/android",
    "/usr/local/lib/node_modules",
    "/usr/local/share/powershell",
    "/usr/share/dotnet",
    "/usr/share/miniconda",
    "/usr/share/swift",
)


def _job_steps(project_root: Path, job_name: str) -> list[dict[str, object]]:
    """Return the steps of `job_name` in the Docker build workflow.

    :param project_root: Repo root; the workflow is read from
        ``<project_root>/.github/workflows/docker-build-validation.yml``.
    :param job_name: Job key within the workflow (e.g. ``docker-validate``).
    :returns: The job's parsed step list.
    """
    workflow = load_workflow(project_root, WORKFLOW_FILENAME)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    return cast(list[dict[str, object]], jobs[job_name]["steps"])


def _reclaim_body(project_root: Path, job_name: str) -> str:
    """Return the shell body of `job_name`'s disk-reclaim step.

    :param project_root: Root path of the repository under test.
    :param job_name: Job whose reclaim step is read.
    :returns: The step's `run` script.
    """
    steps = _job_steps(project_root, job_name)
    step = next(s for s in steps if s.get("name") == FREE_DISK_STEP_NAME)
    return cast(str, step["run"])


@pytest.mark.infra
@pytest.mark.parametrize("job_name", BUILD_JOB_NAMES)
@pytest.mark.parametrize("toolchain_path", RECLAIMED_TOOLCHAIN_PATHS)
def test_free_disk_space_reclaims_each_large_toolchain(
    project_root: Path, job_name: str, toolchain_path: str
) -> None:
    """Every large unused toolchain is reclaimed.

    :param project_root: Root path of the repository under test.
    :param job_name: Job whose reclaim step is checked.
    :param toolchain_path: Toolchain directory that must be removed.
    """
    assert toolchain_path in _reclaim_body(project_root, job_name)


@pytest.mark.infra
@pytest.mark.parametrize("job_name", BUILD_JOB_NAMES)
def test_free_disk_space_keeps_the_python_toolcache_setup_python_reuses(
    project_root: Path, job_name: str
) -> None:
    """The reclaim must not delete the toolcache `Set up Python` reuses.

    Removing `/opt/hostedtoolcache/Python` would force `actions/setup-python` to re-download the
    interpreter, trading disk for a network dependency the build does not need.

    :param project_root: Root path of the repository under test.
    :param job_name: Job whose reclaim step is checked.
    """
    assert "/opt/hostedtoolcache/Python" not in _reclaim_body(project_root, job_name)


@pytest.mark.infra
@pytest.mark.parametrize("job_name", BUILD_JOB_NAMES)
def test_free_disk_space_reports_every_filesystem_not_only_root(
    project_root: Path, job_name: str
) -> None:
    """Disk triage needs each mount's free space, not just `/`.

    A `no space left on device` failure is only actionable if the log shows which filesystem filled
    and what headroom the others had.

    :param project_root: Root path of the repository under test.
    :param job_name: Job whose reclaim step is checked.
    """
    reclaim_lines = [line.strip() for line in _reclaim_body(project_root, job_name).splitlines()]

    assert "df -h" in reclaim_lines
    assert "df -h /" not in reclaim_lines


@pytest.mark.infra
@pytest.mark.parametrize("job_name", BUILD_JOB_NAMES)
def test_free_disk_space_reclaim_is_forced_and_failure_tolerant(
    project_root: Path, job_name: str
) -> None:
    """A path this runner image stops shipping must cost headroom, not the build.

    The step runs under `set -e`, so tolerating a stale entry rests on two
    things: `rm -rf` exits 0 on a missing path where a bare `rm` would not, and
    the `|| echo` keeps any residual failure non-fatal.

    :param project_root: Root path of the repository under test.
    :param job_name: Job whose reclaim step is checked.
    """
    run_body = _reclaim_body(project_root, job_name)

    assert "set -euo pipefail" in run_body
    assert "rm -rf" in run_body
    assert "toolchain cleanup failed, continuing" in run_body


@pytest.mark.infra
def test_both_build_jobs_reclaim_identically(project_root: Path) -> None:
    """Both jobs build the same image, so both need the same reclaim.

    A local composite would be the natural way to share this, but each job
    checks out a ref other than the workflow's own, and `uses: ./...` resolves
    against that workspace — so the copies are pinned equal instead.

    :param project_root: Root path of the repository under test.
    """
    validate_body = _reclaim_body(project_root, "docker-validate")
    push_body = _reclaim_body(project_root, "docker-push")

    assert validate_body == push_body


@pytest.mark.infra
@pytest.mark.parametrize("job_name", BUILD_JOB_NAMES)
def test_no_build_step_runs_before_the_disk_is_reclaimed(
    project_root: Path, job_name: str
) -> None:
    """Reclaiming disk after a build would not help that build.

    Checks every `build-push-action` step, not just the first: the `load: true` export is the step
    under disk pressure, and pinning only one step by name would keep passing if the builds were
    reordered.

    :param project_root: Root path of the repository under test.
    :param job_name: Job whose step order is checked.
    """
    steps = _job_steps(project_root, job_name)
    reclaim_index = next(
        i for i, step in enumerate(steps) if step.get("name") == FREE_DISK_STEP_NAME
    )
    build_indices = [
        i for i, step in enumerate(steps) if BUILD_ACTION in str(step.get("uses", ""))
    ]

    assert build_indices, f"{job_name} has no {BUILD_ACTION} step"
    assert reclaim_index < min(build_indices)


@pytest.mark.infra
@pytest.mark.parametrize("job_name", BUILD_JOB_NAMES)
def test_build_jobs_use_no_local_action_they_may_not_have_checked_out(
    project_root: Path, job_name: str
) -> None:
    """A local composite here breaks every ref that predates it.

    Both jobs check out a ref other than the one the workflow was read from —
    the PR head, or a dispatched `git_ref`. A `uses: ./...` step resolves from
    that workspace, so it fails on any ref that does not carry the action.

    :param project_root: Root path of the repository under test.
    :param job_name: Job whose steps are checked.
    """
    local_uses = [
        step.get("uses")
        for step in _job_steps(project_root, job_name)
        if str(step.get("uses", "")).startswith("./")
    ]

    assert local_uses == []
