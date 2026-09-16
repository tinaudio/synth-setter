"""Regression checks for the auto-approve workflow's CI status policy."""

import json
import os
import shutil
import subprocess
from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/auto-approve.yml")
COPILOT_LOGIN = "copilot-pull-request-reviewer[bot]"
HEAD_SHA = "abc1234def567890"
QUOTA_DENIAL = (
    "Copilot was unable to review this pull request because the user who requested "
    "the review has reached their quota limit."
)


def _run_copilot_review_condition(
    project_root: Path, tmp_path: Path, reviews: list[dict[str, object]]
) -> str:
    workflow = (project_root / WORKFLOW_PATH).read_text()
    condition = workflow.split(
        "# --- Condition 3: Copilot has reviewed and all threads are resolved ---",
        maxsplit=1,
    )[1].split("          UNRESOLVED=", maxsplit=1)[0]
    condition = condition.replace("${{ github.repository_owner }}", '"testorg"')

    gh_stub = tmp_path / "gh"
    gh_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "filter=''\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ $1 == --jq ]]; then filter=$2; break; fi\n"
        "  shift\n"
        "done\n"
        'if [[ -n $filter ]]; then jq -c "$filter" <<<"$REVIEWS_JSON"; '
        "else printf '%s\\n' \"$REVIEWS_JSON\"; fi\n"
    )
    gh_stub.chmod(0o755)
    output_path = tmp_path / "github-output"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output_path),
        "HEAD_SHA": HEAD_SHA,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PR": "42",
        "REPO": "testorg/testrepo",
        "REVIEWS_JSON": json.dumps(reviews),
    }
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash not found on PATH; cannot exercise workflow shell")
    result = subprocess.run(  # noqa: S603 — workflow shell is the behavior under test.
        [bash, "-c", f'{condition}\necho "accepted=true" >> "$GITHUB_OUTPUT"'],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return output_path.read_text()


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


def test_copilot_quota_denial_on_head_does_not_satisfy_condition_3(
    project_root: Path, tmp_path: Path
) -> None:
    """A quota denial must not be mistaken for a completed Copilot review.

    :param project_root: Repository root containing the workflow under test.
    :param tmp_path: Temporary directory for the GitHub CLI stub and output.
    """
    output = _run_copilot_review_condition(
        project_root,
        tmp_path,
        [
            {
                "body": QUOTA_DENIAL,
                "commit_id": HEAD_SHA,
                "user": {"login": COPILOT_LOGIN},
            }
        ],
    )

    assert "result=neutral" in output
    assert "quota" in output.lower()
    assert "accepted=true" not in output


def test_copilot_genuine_review_on_head_satisfies_condition_3(
    project_root: Path, tmp_path: Path
) -> None:
    """A genuine review anchored to the current head must satisfy the review gate.

    :param project_root: Repository root containing the workflow under test.
    :param tmp_path: Temporary directory for the GitHub CLI stub and output.
    """
    output = _run_copilot_review_condition(
        project_root,
        tmp_path,
        [
            {
                "body": "Copilot reviewed 4 out of 4 changed files and generated no comments.",
                "commit_id": HEAD_SHA,
                "user": {"login": COPILOT_LOGIN},
            }
        ],
    )

    assert "accepted=true" in output


def test_copilot_stale_quota_denial_is_irrelevant_to_condition_3(
    project_root: Path, tmp_path: Path
) -> None:
    """A quota denial on an earlier commit must not classify the current head unavailable.

    :param project_root: Repository root containing the workflow under test.
    :param tmp_path: Temporary directory for the GitHub CLI stub and output.
    """
    output = _run_copilot_review_condition(
        project_root,
        tmp_path,
        [
            {
                "body": QUOTA_DENIAL,
                "commit_id": "def567890abc1234",
                "user": {"login": COPILOT_LOGIN},
            }
        ],
    )

    assert "result=neutral" in output
    assert "accepted=true" not in output
    assert "quota" not in output.lower()


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
