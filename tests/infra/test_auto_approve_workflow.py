"""Regression checks for the auto-approve workflow's CI status policy."""

import shutil
import subprocess
from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/auto-approve.yml")


def test_auto_approve_rechecks_pr_updates(project_root: Path) -> None:
    """PR creation and pushes must evaluate the workflow from the PR branch.

    :param project_root: Repository root containing the workflow under test.
    """
    workflow = (project_root / WORKFLOW_PATH).read_text()

    pull_request_block = workflow.split("  pull_request:\n", maxsplit=1)[1].split(
        "  workflow_dispatch:", maxsplit=1
    )[0]

    assert "types: [opened, synchronize, ready_for_review]" in pull_request_block


def test_draft_prs_keep_auto_approve_status_neutral(project_root: Path) -> None:
    """Draft PRs must wait instead of publishing a failing auto-approve status.

    :param project_root: Repository root containing the workflow under test.
    """
    workflow = (project_root / WORKFLOW_PATH).read_text()
    draft_block = workflow.split("# --- Condition 1: PR is not a draft ---", maxsplit=1)[1].split(
        "# --- Condition 1b:", maxsplit=1
    )[0]

    assert 'echo "result=neutral" >> "$GITHUB_OUTPUT"' in draft_block
    assert 'echo "result=failure" >> "$GITHUB_OUTPUT"' not in draft_block


def test_cancelled_checks_keep_auto_approve_status_neutral(project_root: Path) -> None:
    """Cancelled checks must wait for a rerun instead of showing a red status.

    :param project_root: Repository root containing the workflow under test.
    :raises RuntimeError: When ``bash`` isn't available on the test runner.
    """
    workflow = (project_root / WORKFLOW_PATH).read_text()
    pending_line = next(
        line.strip() for line in workflow.splitlines() if line.strip().startswith("PENDING=")
    )
    failed_line = next(
        line.strip() for line in workflow.splitlines() if line.strip().startswith("FAILED=")
    )

    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash not found on PATH; cannot exercise workflow shell")
    result = subprocess.run(  # noqa: S603 — workflow shell is the behavior under test.
        [
            bash,
            "-c",
            f'{pending_line}\n{failed_line}\nprintf "%s %s\\n" "$PENDING" "$FAILED"',
        ],
        env={"CHECK_DATA": "completed cancelled VST slow tests"},
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout == "1 0\n"
