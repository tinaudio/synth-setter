"""Failure-diagnostic invariants for the Claude GitHub Actions workflows."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_composite_action, load_workflow

WORKFLOWS = (
    ("claude.yml", "claude"),
    ("claude-repo-review-full.yml", "repo-review-full"),
)


def _steps(workflow: dict[str, object]) -> list[dict[str, object]]:
    """Return the sole job's steps.

    :param workflow: Parsed GitHub Actions workflow.
    :returns: Step mappings from the workflow's sole job.
    """
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    job = next(iter(jobs.values()))
    return cast(list[dict[str, object]], job["steps"])


@pytest.mark.infra
@pytest.mark.parametrize(("workflow_filename", "claude_step_id"), WORKFLOWS)
def test_claude_workflow_uploads_failure_with_oauth(
    project_root: Path, workflow_filename: str, claude_step_id: str
) -> None:
    """Claude workflows keep OAuth and invoke the shared diagnostic action on failure.

    :param project_root: Repository root containing the workflows.
    :param workflow_filename: Claude workflow under test.
    :param claude_step_id: Identifier of the workflow's Claude step.
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
    expected_auth_reference = "${{ secrets." + "CLAUDE_CODE_OAUTH_TOKEN }}"
    assert action_inputs["claude_code_oauth_token"] == expected_auth_reference
    assert upload_step["uses"] == "./.github/actions/upload-claude-failure"
    assert upload_step["if"] == (
        f"${{{{ failure() && steps.{claude_step_id}.outputs.execution_file != '' }}}}"
    )
    assert upload_inputs["execution-file"] == (
        f"${{{{ steps.{claude_step_id}.outputs.execution_file }}}}"
    )
    assert upload_inputs["artifact-name"] == (
        "claude-failure-${{ github.run_id }}-${{ github.run_attempt }}"
    )


def _capture_diagnostic(
    project_root: Path, tmp_path: Path, events: list[dict[str, object]]
) -> list[object]:
    """Run the repository-owned diagnostic capture step.

    :param project_root: Repository root containing the composite action.
    :param tmp_path: Temporary directory for the SDK log and diagnostic.
    :param events: Claude SDK events to capture.
    :returns: Parsed diagnostic artifact content.
    """
    execution_file = tmp_path / "execution.json"
    execution_file.write_text(json.dumps(events))
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
    return cast(list[object], json.loads((tmp_path / "claude-failure.json").read_text()))


@pytest.mark.infra
def test_claude_failure_action_classifies_quota_without_content(
    project_root: Path, tmp_path: Path
) -> None:
    """The diagnostic classifies quota failures without retaining event content.

    :param project_root: Repository root containing the composite action.
    :param tmp_path: Temporary directory for the SDK log and diagnostic.
    """
    diagnostic = _capture_diagnostic(
        project_root,
        tmp_path,
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
        ],
    )
    assert diagnostic == [
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "error_category": "resource_exhausted",
            "duration_ms": 505,
            "num_turns": 1,
            "total_cost_usd": 0,
        }
    ]
    serialized = json.dumps(diagnostic)
    assert "secret-worker-output" not in serialized
    assert "quota resets" not in serialized

    action = cast(dict[str, object], load_composite_action(project_root, "upload-claude-failure"))
    runs = cast(dict[str, object], action["runs"])
    upload_step = cast(list[dict[str, object]], runs["steps"])[1]
    upload_inputs = cast(dict[str, object], upload_step["with"])
    assert str(upload_step["uses"]).startswith("actions/upload-artifact@")
    assert upload_inputs["name"] == "${{ inputs.artifact-name }}"
    assert upload_inputs["path"] == "${{ runner.temp }}/claude-failure.json"


@pytest.mark.infra
def test_claude_failure_action_handles_empty_log(project_root: Path, tmp_path: Path) -> None:
    """An empty SDK log produces a valid empty diagnostic.

    :param project_root: Repository root containing the composite action.
    :param tmp_path: Temporary directory for the SDK log and diagnostic.
    """
    assert _capture_diagnostic(project_root, tmp_path, []) == []
