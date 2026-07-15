"""Failure-diagnostic invariants for the Claude GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from workflow_fixtures import load_workflow

WORKFLOW_FILENAMES = ("claude.yml", "claude-repo-review-full.yml")


def _step_by_id(workflow: dict[str, Any], step_id: str) -> dict[str, Any]:
    """Return a workflow step by ID.

    :param workflow: Parsed GitHub Actions workflow.
    :param step_id: Step identifier to locate.
    :returns: Matching step mapping.
    """
    job = next(iter(workflow["jobs"].values()))
    return next(step for step in job["steps"] if step.get("id") == step_id)


@pytest.mark.infra
@pytest.mark.parametrize("workflow_filename", WORKFLOW_FILENAMES)
def test_claude_workflow_preserves_sanitized_failure_diagnostic(
    project_root: Path, workflow_filename: str
) -> None:
    """Claude workflows retain the terminal SDK result when execution fails.

    :param project_root: Repository root containing the workflows.
    :param workflow_filename: Claude workflow under test.
    """
    workflow = load_workflow(project_root, workflow_filename)
    claude_step = next(
        step
        for step in next(iter(workflow["jobs"].values()))["steps"]
        if str(step.get("uses", "")).startswith("anthropics/claude-code-action@")
    )
    diagnostic_step = _step_by_id(workflow, "capture-claude-failure")
    upload_step = _step_by_id(workflow, "upload-claude-failure")

    assert claude_step["with"]["claude_code_oauth_token"] == (
        "${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}"
    )
    assert "failure()" in diagnostic_step["if"]
    assert "execution_file" in diagnostic_step["if"]
    assert "select(.type == \"result\")" in diagnostic_step["run"]
    assert upload_step["if"] == (
        "${{ always() && steps.capture-claude-failure.outcome == 'success' }}"
    )
