"""Semantic Release serializes queued jobs against the live main tip."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict, cast

import pytest
import yaml

pytestmark = pytest.mark.infra

WorkflowStep = TypedDict(
    "WorkflowStep",
    {
        "continue-on-error": bool,
        "env": dict[str, str],
        "id": str,
        "if": str,
        "name": str,
        "run": str,
        "uses": str,
        "with": dict[str, str | int],
    },
    total=False,
)


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run Git with captured output and fail on nonzero status.

    :param *args: Test-owned Git operands passed without shell expansion.
    :param cwd: Isolated test repository, never the project checkout.
    :returns: Successful process with stdout and stderr retained as text.
    """
    return subprocess.run(  # noqa: S603 — resolved Git binary and test-controlled arguments
        [shutil.which("git") or "git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit(repository: Path, filename: str, message: str) -> str:
    """Create one conventional commit in an isolated repository.

    :param repository: Mutable test checkout.
    :param filename: Relative path used to make the commit non-empty.
    :param message: Subject consumed by Semantic Release.
    :returns: Full commit SHA used as a simulated push event.
    """
    (repository / filename).write_text(f"{message}\n")
    _git("add", filename, cwd=repository)
    _git("commit", "-m", message, cwd=repository)
    return _git("rev-parse", "HEAD", cwd=repository).stdout.strip()


def _release_steps(workflow: object) -> list[WorkflowStep]:
    """Validate the release-job shape before reading control-flow fields.

    :param workflow: Untrusted value returned by YAML parsing.
    :returns: String-keyed steps safe to inspect as ``WorkflowStep``.
    :raises TypeError: If the release job or steps have the wrong shape.
    """
    if not isinstance(workflow, dict):
        raise TypeError("release workflow must be a mapping")
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        raise TypeError("release workflow jobs must be a mapping")
    release = jobs.get("release")
    if not isinstance(release, dict):
        raise TypeError("release job must be a mapping")
    steps = release.get("steps")
    if not isinstance(steps, list) or not all(
        isinstance(step, dict) and all(isinstance(key, str) for key in step) for step in steps
    ):
        raise TypeError("release steps must be string-keyed mappings")
    return cast(list[WorkflowStep], steps)


def _checkout_ref(workflow: object) -> str | None:
    """Resolve the checkout ref that selects stale or live job behavior.

    :param workflow: Parsed workflow validated by :func:`_release_steps`.
    :returns: Explicit branch ref, or ``None`` for the triggering SHA.
    :raises TypeError: If checkout or its options have the wrong shape.
    """
    checkout = next(
        (
            step
            for step in _release_steps(workflow)
            if isinstance((uses := step.get("uses")), str) and uses.startswith("actions/checkout@")
        ),
        None,
    )
    if checkout is None:
        raise TypeError("release workflow must contain actions/checkout")
    options = checkout.get("with")
    if options is None:
        return None
    if not isinstance(options, dict):
        raise TypeError("checkout options must be a mapping")
    ref = options.get("ref")
    if ref is not None and not isinstance(ref, str):
        raise TypeError("checkout ref must be a string")
    return ref


def _step_by_id(steps: list[WorkflowStep], step_id: str) -> WorkflowStep:
    """Require a workflow step used by the release-retry contract.

    :param steps: Structurally validated release steps.
    :param step_id: Stable ID referenced by workflow expressions.
    :returns: Matching step whose fields are asserted by the test.
    :raises TypeError: If workflow wiring omitted the required step.
    """
    step = next((candidate for candidate in steps if candidate.get("id") == step_id), None)
    if step is None:
        raise TypeError(f"release workflow must contain step {step_id}")
    return step


def _start_queued_job(
    remote: Path,
    destination: Path,
    trigger_sha: str,
    *,
    checkout_ref: str | None,
) -> None:
    """Materialize the branch state seen when a queued job starts.

    :param remote: Bare origin shared by simulated jobs.
    :param destination: Fresh disposable checkout.
    :param trigger_sha: Commit carried by the original push event.
    :param checkout_ref: Live ref override; ``None`` reproduces stale checkout.
    """
    _git("clone", str(remote), str(destination), cwd=remote.parent)
    target = checkout_ref or trigger_sha
    _git("checkout", "-B", "main", target, cwd=destination)
    _git("config", "user.name", "Release Test", cwd=destination)
    _git("config", "user.email", "release-test@example.com", cwd=destination)


def _run_release(repository: Path) -> subprocess.CompletedProcess[str]:
    """Run the real version, commit, tag, and push path against fake origin.

    :param repository: Disposable release-job checkout.
    :returns: Process result retained so rejected pushes remain observable.
    :raises RuntimeError: If the required release CLI is unavailable.
    """
    semantic_release = shutil.which("semantic-release")
    if semantic_release is None:
        raise RuntimeError("semantic-release must be installed for release integration tests")
    return subprocess.run(  # noqa: S603 — resolved release binary and fixed arguments
        [
            semantic_release,
            "version",
            "--no-changelog",
            "--no-vcs-release",
            "--skip-build",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )


def _seed_release_remote(tmp_path: Path) -> tuple[Path, Path, str]:
    """Create tagged history with one release-worthy push event.

    :param tmp_path: Per-test storage boundary.
    :returns: Bare origin, mutable seed checkout, and queued trigger SHA.
    """
    remote = tmp_path / "origin.git"
    _git("init", "--bare", "--initial-branch=main", str(remote), cwd=tmp_path)

    seed = tmp_path / "seed"
    _git("clone", str(remote), str(seed), cwd=tmp_path)
    _git("config", "user.name", "Release Test", cwd=seed)
    _git("config", "user.email", "release-test@example.com", cwd=seed)
    (seed / "pyproject.toml").write_text(
        """[project]
name = "release-race"
version = "1.0.0"

[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
branch = "main"
commit_message = "chore(release): {version}"
"""
    )
    _git("add", "pyproject.toml", cwd=seed)
    _git("commit", "-m", "chore: seed release history", cwd=seed)
    _git("tag", "v1.0.0", cwd=seed)
    _git("push", "origin", "main", "v1.0.0", cwd=seed)

    trigger_sha = _commit(seed, "older.txt", "fix: queue older release job")
    _git("push", "origin", "main", cwd=seed)
    return remote, seed, trigger_sha


def _assert_single_release(remote: Path, expected_tag: str, ancestors: tuple[str, ...]) -> None:
    """Assert one new release preserves every intervening main commit.

    :param remote: Bare origin after all simulated jobs finish.
    :param expected_tag: Sole tag allowed in addition to the baseline.
    :param ancestors: Commits that the release tip must retain.
    """
    release_count = _git(
        "rev-list", "--count", "--grep=^chore(release):", "main", cwd=remote
    ).stdout.strip()
    assert release_count == "1"
    assert _git("tag", "--list", "v*", cwd=remote).stdout.splitlines() == [
        "v1.0.0",
        expected_tag,
    ]
    for ancestor in ancestors:
        _git("merge-base", "--is-ancestor", ancestor, "main", cwd=remote)


def _recovery_environment(repository: Path, release_base: str) -> dict[str, str]:
    """Opt a disposable checkout into destructive stale-state cleanup.

    :param repository: Simulated ``GITHUB_WORKSPACE`` boundary.
    :param release_base: Remote main tip observed before release generation.
    :returns: Environment satisfying every production safety guard.
    """
    return os.environ | {
        "GITHUB_ACTIONS": "true",
        "GITHUB_OUTPUT": str(repository / "github-output"),
        "GITHUB_WORKSPACE": str(repository),
        "RELEASE_BASE": release_base,
        "RELEASE_RECOVERY_ALLOWED": "true",
    }


def _run_recovery_script(
    project_root: Path, repository: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — repository-owned script and fixed environment
        [str(project_root / "scripts/ci/recover_stale_release.sh")],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _recover_stale_release(
    project_root: Path, repository: Path, release_base: str
) -> subprocess.CompletedProcess[str]:
    environment = _recovery_environment(repository, release_base)
    return _run_recovery_script(project_root, repository, environment)


def _finalize_stale_release(
    project_root: Path, repository: Path, release_base: str
) -> subprocess.CompletedProcess[str]:
    environment = _recovery_environment(repository, release_base)
    environment["FAIL_AFTER_RECOVERY"] = "true"
    return _run_recovery_script(project_root, repository, environment)


def test_release_workflow_wires_stale_failure_recovery_and_safe_retry(
    project_root: Path,
) -> None:
    """Pin the control flow that wraps the real release and recovery paths.

    :param project_root: Checkout containing the release workflow under test.
    """
    workflow = yaml.safe_load((project_root / ".github/workflows/release.yml").read_text())
    steps = _release_steps(workflow)
    release = _step_by_id(steps, "release")
    recovery = _step_by_id(steps, "recover-release")
    retry = _step_by_id(steps, "retry-release")
    final = _step_by_id(steps, "finalize-stale-release")
    enforcement = _step_by_id(steps, "enforce-release-outcome")

    assert release.get("continue-on-error") is True
    assert recovery.get("if") == "steps.release.outcome == 'failure'"
    assert retry.get("if") == "steps.recover-release.outcome == 'success'"
    assert retry.get("continue-on-error") is True
    assert final.get("if") == "steps.retry-release.outcome == 'failure'"
    assert enforcement.get("if") == "always()"

    recovery_environment = recovery.get("env")
    final_environment = final.get("env")
    assert isinstance(recovery_environment, dict)
    assert isinstance(final_environment, dict)
    assert recovery_environment["RELEASE_BASE"] == "${{ steps.release-base.outputs.base }}"
    assert recovery_environment["RELEASE_RECOVERY_ALLOWED"] == "true"
    assert final_environment["RELEASE_BASE"] == "${{ steps.recover-release.outputs.base }}"
    assert final_environment["FAIL_AFTER_RECOVERY"] == "true"
    enforcement_command = enforcement.get("run")
    assert isinstance(enforcement_command, str)
    assert 'FIRST_OUTCOME" == "success" || "$RETRY_OUTCOME" == "success' in enforcement_command


def test_serialized_release_jobs_refresh_to_live_main_without_duplicate_release(
    project_root: Path, tmp_path: Path
) -> None:
    """Preserve an intervening merge while two queued jobs release exactly once.

    :param project_root: Checkout containing the release workflow under test.
    :param tmp_path: Isolated directory for the temporary Git remote and jobs.
    """
    workflow = yaml.safe_load((project_root / ".github/workflows/release.yml").read_text())
    checkout_ref = _checkout_ref(workflow)
    remote, seed, older_trigger = _seed_release_remote(tmp_path)

    intervening_main = _commit(seed, "intervening.txt", "feat: advance main while job waits")
    _git("push", "origin", "main", cwd=seed)

    older_job = tmp_path / "older-job"
    _start_queued_job(remote, older_job, older_trigger, checkout_ref=checkout_ref)
    older_result = _run_release(older_job)
    assert older_result.returncode == 0, older_result.stdout + older_result.stderr

    newer_job = tmp_path / "newer-job"
    _start_queued_job(remote, newer_job, intervening_main, checkout_ref=checkout_ref)
    newer_result = _run_release(newer_job)
    assert newer_result.returncode == 0, newer_result.stdout + newer_result.stderr

    _assert_single_release(remote, "v1.1.0", (intervening_main,))

    release_output = (
        older_result.stdout + older_result.stderr + newer_result.stdout + newer_result.stderr
    )
    assert "non-fast-forward" not in release_output
    assert "[rejected]" not in release_output


def test_release_push_race_recovers_from_live_main_without_duplicate_release(
    project_root: Path, tmp_path: Path
) -> None:
    """Retry from live main when it advances after checkout but before release push.

    :param project_root: Checkout containing the recovery script under test.
    :param tmp_path: Isolated directory for the temporary Git remote and job.
    """
    remote, seed, trigger_sha = _seed_release_remote(tmp_path)
    release_job = tmp_path / "release-job"
    _start_queued_job(remote, release_job, trigger_sha, checkout_ref="main")
    release_base = _git("rev-parse", "HEAD", cwd=release_job).stdout.strip()

    intervening_main = _commit(seed, "intervening.txt", "feat: advance main during release")
    _git("push", "origin", "main", cwd=seed)

    stale_result = _run_release(release_job)
    assert stale_result.returncode != 0
    assert "[rejected]" in stale_result.stderr
    stale_artifact = release_job / "dist" / "stale.whl"
    stale_artifact.parent.mkdir()
    stale_artifact.touch()

    recovery = _recover_stale_release(project_root, release_job, release_base)
    assert recovery.returncode == 0, recovery.stdout + recovery.stderr
    assert _git("rev-parse", "HEAD", cwd=release_job).stdout.strip() == intervening_main
    assert not stale_artifact.exists()

    retry_result = _run_release(release_job)
    assert retry_result.returncode == 0, retry_result.stdout + retry_result.stderr
    _assert_single_release(remote, "v1.1.0", (intervening_main,))


def test_recovery_failure_before_release_commit_refreshes_to_live_main(
    project_root: Path, tmp_path: Path
) -> None:
    """Recover when an attempt fails before creating its release commit.

    :param project_root: Checkout containing the recovery script under test.
    :param tmp_path: Isolated directory for the temporary Git remote and job.
    """
    remote, seed, trigger_sha = _seed_release_remote(tmp_path)
    release_job = tmp_path / "release-job"
    _start_queued_job(remote, release_job, trigger_sha, checkout_ref="main")

    intervening_main = _commit(seed, "intervening.txt", "fix: advance main before commit")
    _git("push", "origin", "main", cwd=seed)
    recovery = _recover_stale_release(project_root, release_job, trigger_sha)

    assert recovery.returncode == 0, recovery.stdout + recovery.stderr
    assert _git("rev-parse", "HEAD", cwd=release_job).stdout.strip() == intervening_main


def test_recovery_tagged_base_refreshes_without_treating_tag_as_new_release(
    project_root: Path, tmp_path: Path
) -> None:
    """Ignore a pre-existing tag when no local release commit was created.

    :param project_root: Checkout containing the recovery script under test.
    :param tmp_path: Isolated directory for the temporary Git remote and job.
    """
    remote, _, live_main = _seed_release_remote(tmp_path)
    tagged_base = _git("rev-list", "-n", "1", "v1.0.0", cwd=remote).stdout.strip()
    release_job = tmp_path / "release-job"
    _start_queued_job(remote, release_job, tagged_base, checkout_ref=None)

    recovery = _recover_stale_release(project_root, release_job, tagged_base)

    assert recovery.returncode == 0, recovery.stdout + recovery.stderr
    assert _git("rev-parse", "HEAD", cwd=release_job).stdout.strip() == live_main
    assert _git("tag", "--list", "v*", cwd=release_job).stdout.splitlines() == ["v1.0.0"]


def test_second_release_push_race_exits_safely_for_latest_queued_job(
    project_root: Path, tmp_path: Path
) -> None:
    """Let the newest queued job release after both earlier attempts become stale.

    :param project_root: Checkout containing the recovery script under test.
    :param tmp_path: Isolated directory for the temporary Git remote and jobs.
    """
    remote, seed, trigger_sha = _seed_release_remote(tmp_path)
    release_job = tmp_path / "release-job"
    _start_queued_job(remote, release_job, trigger_sha, checkout_ref="main")
    first_base = _git("rev-parse", "HEAD", cwd=release_job).stdout.strip()

    first_advance = _commit(seed, "first.txt", "fix: advance main during first attempt")
    _git("push", "origin", "main", cwd=seed)
    assert _run_release(release_job).returncode != 0
    first_recovery = _recover_stale_release(project_root, release_job, first_base)
    assert first_recovery.returncode == 0, first_recovery.stdout + first_recovery.stderr

    second_advance = _commit(seed, "second.txt", "feat: advance main during retry")
    _git("push", "origin", "main", cwd=seed)
    assert _run_release(release_job).returncode != 0
    second_recovery = _finalize_stale_release(project_root, release_job, first_advance)
    assert second_recovery.returncode != 0
    assert "Main advanced twice" in second_recovery.stderr
    assert _git("rev-parse", "HEAD", cwd=release_job).stdout.strip() == second_advance

    newest_job = tmp_path / "newest-job"
    _start_queued_job(remote, newest_job, second_advance, checkout_ref="main")
    newest_result = _run_release(newest_job)
    assert newest_result.returncode == 0, newest_result.stdout + newest_result.stderr
    _assert_single_release(remote, "v1.1.0", (first_advance, second_advance))


def test_stale_recovery_published_release_refuses_duplicate_retry(
    project_root: Path, tmp_path: Path
) -> None:
    """Leave a published release untouched instead of retrying it.

    :param project_root: Checkout containing the recovery script under test.
    :param tmp_path: Isolated directory for the temporary Git remote and job.
    """
    remote, seed, trigger_sha = _seed_release_remote(tmp_path)
    release_job = tmp_path / "release-job"
    _start_queued_job(remote, release_job, trigger_sha, checkout_ref="main")

    release_result = _run_release(release_job)
    assert release_result.returncode == 0, release_result.stdout + release_result.stderr
    release_tip = _git("rev-list", "-n", "1", "v1.0.1", cwd=remote).stdout.strip()

    _commit(seed, "intervening.txt", "fix: replace main after published tag")
    _git("push", "--force", "origin", "main", cwd=seed)
    recovery = _recover_stale_release(project_root, release_job, trigger_sha)

    assert recovery.returncode != 0
    assert "Release tag v1.0.1 is already published" in recovery.stderr
    assert _git("rev-list", "-n", "1", "v1.0.1", cwd=remote).stdout.strip() == release_tip
    assert _git("tag", "--list", "v*", cwd=remote).stdout.splitlines() == ["v1.0.0", "v1.0.1"]
