"""Git hook installation invariants for fresh worktrees."""

from __future__ import annotations

import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_install_git_hooks_target_installs_commit_and_push_hooks() -> None:
    """The shared installer registers both enforcement stages."""
    result = subprocess.run(  # noqa: S603
        ["make", "--dry-run", "install-git-hooks"],  # noqa: S607
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--hook-type pre-commit" in result.stdout
    assert "--hook-type pre-push" in result.stdout


def test_pre_push_stage_runs_all_pre_commit_hooks() -> None:
    """The pre-push stage blocks unless every tracked file passes pre-commit."""
    config = (_PROJECT_ROOT / ".pre-commit-config.yaml").read_text()

    assert "id: pre-push-all-files" in config
    assert "entry: uv run pre-commit run --all-files" in config
    assert "stages: [pre-push]" in config
