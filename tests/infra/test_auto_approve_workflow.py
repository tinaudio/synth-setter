"""Regression checks for the auto-approve workflow's CI status policy."""

from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/auto-approve.yml")


def test_cancelled_checks_keep_auto_approve_status_neutral(project_root: Path) -> None:
    """Cancelled checks must wait for a rerun instead of showing a red status.

    :param project_root: Repository root containing the workflow under test.
    """
    workflow = (project_root / WORKFLOW_PATH).read_text()

    assert 'grep -c "^completed cancelled\\|^queued\\|^in_progress"' in workflow
    assert (
        'grep -v "^completed success\\|^completed skipped\\|^completed neutral\\|^completed cancelled\\|^queued\\|^in_progress"'
        in workflow
    )
