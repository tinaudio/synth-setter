"""Security and runtime contracts for the automatic Pi PR review workflow."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workflow_fixtures import load_workflow

WORKFLOW_FILENAME = "claude-repo-review-full.yml"


def _workflow(project_root: Path) -> dict[str, object]:
    """Load the automatic Pi review workflow.

    :param project_root: Repository root containing the workflow.
    :returns: Parsed workflow mapping.
    """
    return cast(dict[str, object], load_workflow(project_root, WORKFLOW_FILENAME))


def _job(project_root: Path) -> dict[str, object]:
    """Return the automatic review job.

    :param project_root: Repository root containing the workflow.
    :returns: Parsed review job mapping.
    """
    workflow = _workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    return jobs["repo-review-full"]


def _steps(project_root: Path) -> list[dict[str, object]]:
    """Return the automatic review steps.

    :param project_root: Repository root containing the workflow.
    :returns: Parsed step mappings.
    """
    return cast(list[dict[str, object]], _job(project_root)["steps"])


@pytest.mark.infra
def test_pi_review_workflow_triggers_for_pr_heads_and_cancels_stale_runs(
    project_root: Path,
) -> None:
    """PR updates and trusted mentions share one stale-run concurrency group.

    :param project_root: Repository root containing the workflow.
    """
    workflow = _workflow(project_root)
    triggers = cast(dict[str, dict[str, list[str]]], workflow["on"])

    assert "pull_request_target" not in triggers
    assert triggers["pull_request"]["types"] == [
        "opened",
        "synchronize",
        "ready_for_review",
        "reopened",
    ]
    assert triggers["issue_comment"]["types"] == ["created"]
    assert workflow["concurrency"] == {
        "group": (
            "pi-repo-review-full-${{ github.event.pull_request.number || "
            "(github.event.comment.user.login == 'ktinubu' && "
            "contains(github.event.comment.body, '@github-actions review') && "
            "github.event.issue.number) || github.run_id }}"
        ),
        "cancel-in-progress": True,
    }


@pytest.mark.infra
def test_pi_review_workflow_secrets_are_restricted_to_allowlisted_same_repo_prs(
    project_root: Path,
) -> None:
    """Only allowlisted same-repository PRs can reach review secrets.

    :param project_root: Repository root containing the workflow.
    """
    workflow = _workflow(project_root)
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    authorization_job = jobs["authorize-review"]
    job = _job(project_root)

    assert authorization_job["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }
    assert "@github-actions review" in str(authorization_job["if"])
    assert "github.event.issue.state == 'open'" in str(authorization_job["if"])
    assert "github.event.comment.user.login == 'ktinubu'" in str(authorization_job["if"])
    assert "github.event.pull_request.head.repo.full_name == github.repository" in str(
        authorization_job["if"]
    )
    authorization_steps = str(authorization_job["steps"])
    assert "is_trusted_pi_review_pr.py" in authorization_steps
    assert "should_run_pi_review.py" in authorization_steps
    assert job["needs"] == "authorize-review"
    assert job["if"] == "${{ needs.authorize-review.outputs.should_review == 'true' }}"
    assert job["permissions"] == {
        "actions": "read",
        "checks": "read",
        "contents": "read",
        "pull-requests": "write",
    }


@pytest.mark.infra
def test_pi_review_workflow_uses_pinned_private_inputs_without_persisted_credentials(
    project_root: Path,
) -> None:
    """Both source trees and all third-party actions are immutable inputs.

    :param project_root: Repository root containing the workflow.
    """
    steps = _steps(project_root)
    checkout_steps = [
        step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")
    ]

    assert len(checkout_steps) == 2
    for step in checkout_steps:
        assert len(str(step["uses"]).partition("@")[2]) == 40
        assert cast(dict[str, object], step["with"])["persist-credentials"] is False

    skills_checkout = next(
        step
        for step in checkout_steps
        if cast(dict[str, object], step["with"]).get("repository") == "tinaudio/skills"
    )
    skills_inputs = cast(dict[str, object], skills_checkout["with"])
    expected_pat_reference = "${{ secrets." + "GIT_PAT }}"
    assert skills_inputs["token"] == expected_pat_reference
    assert len(str(skills_inputs["ref"])) == 40
    assert skills_inputs["path"] == ".review-skills"

    for step in steps:
        action = str(step.get("uses", ""))
        if action and not action.startswith("./"):
            assert len(action.partition("@")[2].split()[0]) == 40


@pytest.mark.infra
def test_pi_review_workflow_restores_pi_auth_and_runs_canonical_launcher(
    project_root: Path,
) -> None:
    """The real Pi launcher receives the PR number and approved provider credentials.

    :param project_root: Repository root containing the workflow.
    """
    workflow_text = (project_root / ".github" / "workflows" / WORKFLOW_FILENAME).read_text()
    steps = _steps(project_root)
    auth_step = next(step for step in steps if step.get("name") == "Configure Pi authentication")
    install_step = next(step for step in steps if step.get("name") == "Install Pi review runtime")
    review_step = next(step for step in steps if step.get("name") == "Run Pi repo-review-full")
    auth_env = cast(dict[str, object], auth_step["env"])
    install_env = cast(dict[str, object], install_step["env"])
    review_env = cast(dict[str, object], review_step["env"])

    assert "env" not in _job(project_root)
    expected_agent_dir = "${{ runner.temp }}/pi-agent"
    assert auth_env["PI_CODING_AGENT_DIR"] == expected_agent_dir
    assert install_env["PI_CODING_AGENT_DIR"] == expected_agent_dir
    assert review_env["PI_CODING_AGENT_DIR"] == expected_agent_dir
    assert auth_env["PI_AUTH_JSON"] == "${{ secrets.PI_AUTH_JSON }}"
    assert "install -m 600" in str(auth_step["run"])
    assert "anthropics/" not in workflow_text
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in workflow_text
    expected_github_token_reference = "${{ github." + "token }}"
    assert review_env["GH_TOKEN"] == expected_github_token_reference
    assert review_env["PR_NUMBER"] == (
        "${{ github.event.pull_request.number || github.event.issue.number }}"
    )
    assert str(review_env["PI_REVIEW_SKILLS_ROOT"]).endswith(
        "/.review-skills/codex/synth-setter-skills"
    )
    assert review_env["CI"] == "true"
    assert review_env["PI_REVIEW_FOLLOW_UP_OWNERSHIP_WAIT_SECONDS"] == "900"
    assert str(review_step["run"]).strip() == (
        'bash agent/_shared/run_pi_review.sh repo-review-full --target "${PR_NUMBER}"'
    )


@pytest.mark.infra
def test_pi_review_workflow_pins_runtime_and_uploads_follow_up_audit(
    project_root: Path,
) -> None:
    """The reproducible runtime retains foreground and synchronous follow-up evidence.

    :param project_root: Repository root containing the workflow.
    """
    steps = _steps(project_root)
    install_step = next(step for step in steps if step.get("name") == "Install Pi review runtime")
    upload_step = next(step for step in steps if step.get("name") == "Upload Pi review audit")
    install_command = str(install_step["run"])
    upload_inputs = cast(dict[str, object], upload_step["with"])

    assert "@earendil-works/pi-coding-agent@0.84.4" in install_command
    assert "@tintinweb/pi-subagents@0.14.1" in install_command
    assert "pydantic==2.13.4" in install_command
    assert "sh==2.2.2" in install_command
    assert upload_step["if"] == "${{ always() }}"
    assert upload_inputs["path"] == ".agent-reviews/"
    assert upload_inputs["include-hidden-files"] is True
    assert upload_inputs["if-no-files-found"] == "ignore"
    assert upload_inputs["retention-days"] == 7
