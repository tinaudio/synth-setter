"""Failure-diagnostic invariants for the Claude GitHub Actions workflows."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_composite_action, load_workflow

WORKFLOW_FILENAMES = ("claude.yml", "claude-repo-review-full.yml")


def _steps(workflow: dict[str, object]) -> list[dict[str, object]]:
    """Return the sole job's steps.

    :param workflow: Parsed GitHub Actions workflow.
    :returns: Step mappings from the workflow's sole job.
    """
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    job = next(iter(jobs.values()))
    return cast(list[dict[str, object]], job["steps"])


@pytest.mark.infra
@pytest.mark.parametrize("workflow_filename", WORKFLOW_FILENAMES)
def test_claude_workflow_uploads_failure_with_oauth(
    project_root: Path, workflow_filename: str
) -> None:
    """Claude workflows keep OAuth and invoke the shared diagnostic action on failure.

    :param project_root: Repository root containing the workflows.
    :param workflow_filename: Claude workflow under test.
    """
    workflow = cast(dict[str, object], load_workflow(project_root, workflow_filename))
    steps = _steps(workflow)
    claude_step = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("anthropics/claude-code-action@")
    )
    upload_step = next(step for step in steps if step.get("id") == "upload-claude-failure")

    action_inputs = cast(dict[str, object], claude_step["with"])
    upload_inputs = cast(dict[str, object], upload_step["with"])
    assert str(action_inputs["claude_code_oauth_token"]).endswith("CLAUDE_CODE_OAUTH_TOKEN }}")
    assert upload_step["uses"] == "./.github/actions/upload-claude-failure"
    assert "failure()" in cast(str, upload_step["if"])
    assert "execution-file" in upload_inputs


@pytest.mark.infra
def test_claude_failure_action_excludes_worker_events(project_root: Path, tmp_path: Path) -> None:
    """The diagnostic contains the terminal error without tool output or secrets.

    :param project_root: Repository root containing the composite action.
    :param tmp_path: Temporary directory for the SDK log and diagnostic.
    """
    execution_file = tmp_path / "execution.json"
    execution_file.write_text(
        json.dumps(
            [
                {"type": "system", "subtype": "init", "model": "claude-sonnet-5"},
                {"type": "tool_result", "content": "secret-worker-output"},
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": True,
                    "result": "resource_exhausted: quota resets at 5 PM",
                    "duration_ms": 505,
                    "num_turns": 1,
                    "total_cost_usd": 0,
                },
            ]
        )
    )
    action = cast(dict[str, object], load_composite_action(project_root, "upload-claude-failure"))
    runs = cast(dict[str, object], action["runs"])
    capture_step = cast(list[dict[str, object]], runs["steps"])[0]
    env = os.environ | {
        "EXECUTION_FILE": str(execution_file),
        "RUNNER_TEMP": str(tmp_path),
    }

    subprocess.run(  # noqa: S603 — the repository-owned action body is the behavior under test.
        ["/bin/bash", "-c", cast(str, capture_step["run"])], env=env, check=True
    )
    diagnostic = json.loads((tmp_path / "claude-failure.json").read_text())
    assert diagnostic == [
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "resource_exhausted: quota resets at 5 PM",
            "errors": None,
            "duration_ms": 505,
            "num_turns": 1,
            "total_cost_usd": 0,
        }
    ]
    assert "secret-worker-output" not in json.dumps(diagnostic)
