"""Regression checks for the auto-approve workflow's CI status policy."""

import shutil
import subprocess
from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/auto-approve.yml")


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
