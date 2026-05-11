"""Invariant 3: representative workflows parse under `act -n` (dry-run/list mode).

`act --list` parses each workflow file the same way GitHub Actions does and
exits 0 if the YAML is structurally valid. We do NOT execute jobs — just
validate parsing. Skipped when `act` is not on PATH (the common local case);
the test exists for the infra-refactor loop, where `act` is installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPRESENTATIVE_WORKFLOWS = (
    "test.yml",
    "code-quality-pr.yaml",
    "stale.yml",
)

ACT_TIMEOUT_SECONDS = 30


def _run_act_list(workflow_path: Path, repo_root: Path) -> subprocess.CompletedProcess[str]:
    """Invoke `act --list` against a single workflow file and return the result."""
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["act", "-W", str(workflow_path), "--list"],  # noqa: S607 — act on PATH
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=ACT_TIMEOUT_SECONDS,
        check=False,
    )


@pytest.mark.infra
@pytest.mark.parametrize("workflow_filename", REPRESENTATIVE_WORKFLOWS)
def test_representative_workflow_parses_under_act_list_mode(
    project_root: Path,
    workflow_filename: str,
) -> None:
    """`act --list` parses the workflow YAML and exits 0."""
    if shutil.which("act") is None:
        pytest.skip("'act' binary not available on PATH")

    workflow_path = project_root / ".github" / "workflows" / workflow_filename
    assert workflow_path.is_file(), f"Missing representative workflow: {workflow_path}"

    result = _run_act_list(workflow_path, project_root)
    assert result.returncode == 0, (
        f"act --list failed for {workflow_filename!r} with exit {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
