"""Behavior tests for deterministic Pi no-comments report rendering."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import sh

from agent._shared.pi_review_render import (
    RenderContext,
    ReviewPayload,
    render_markdown,
    render_payload,
    resolve_context,
)


def test_render_markdown_groups_findings_and_preserves_audit() -> None:
    """Render review data without asking the host model to reconstruct Markdown."""
    payload = {
        "pr_number": 2174,
        "repo": "tinaudio/synth-setter",
        "review_body": "Lead.\n\n## Pi review audit\n\nAudit row.",
        "findings": [
            {
                "path": "src/example.py",
                "line": 42,
                "body": "**[correctness:block]** Broken boundary.",
            },
            {
                "path": "src/example.py",
                "line": 55,
                "body": "**[python-style:warn]** Unclear name.",
            },
        ],
    }
    context = RenderContext(
        target="PR #2174",
        head_sha="a" * 40,
        head_ref="fix/review",
        upstream_sha="a" * 40,
        worktree_state="clean",
        unchanged_count=0,
        skill_count=2,
        next_step="Run /repo-review-full 2174 to post these findings.",
    )

    report = render_markdown(payload, context=context)

    assert "## Pi review audit" in report
    assert report.count("### `src/example.py`") == 1
    assert "**L42** — **[correctness:block]** Broken boundary." in report
    assert "1 BLOCK, 1 WARN across 2 skills" in report
    assert "Reviewed at: " + "a" * 40 in report


def _init_git_repo(path: Path) -> str:
    """Create one committed repository and return its HEAD.

    :param path: Empty repository directory.
    :returns: Full commit SHA.
    """
    git = sh.Command("git")
    git("init", "-q", path)
    git("-C", path, "config", "user.email", "test@example.com")
    git("-C", path, "config", "user.name", "Test")
    (path / "tracked.txt").write_text("content\n")
    git("-C", path, "add", "tracked.txt")
    git("-C", path, "commit", "-qm", "test: initialize")
    return str(git("-C", path, "rev-parse", "HEAD")).strip()


def test_resolve_context_reviewed_head_drift_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse to stamp a sentinel for a commit workers did not review.

    :param tmp_path: Temporary Git repository.
    :param monkeypatch: Changes the current directory to that repository.
    """
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    payload = ReviewPayload.model_validate_json(
        '{"pr_number":1,"repo":"owner/repo","review_body":"Audit.","findings":[]}'
    )

    with pytest.raises(ValueError, match="Reviewed HEAD changed before delivery"):
        resolve_context(
            payload,
            reviewed_head="f" * 40,
            target="PR #1",
            skill_count=1,
            next_step="Done.",
        )


def test_resolve_context_corrupt_progress_treated_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep advisory progress corruption from blocking review delivery.

    :param tmp_path: Temporary Git repository.
    :param monkeypatch: Changes the current directory to that repository.
    """
    head = _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    branch = str(sh.Command("git")("branch", "--show-current")).strip()
    key = hashlib.sha256(f"{branch}\n".encode()).hexdigest()
    progress = tmp_path / f".agent-reviews/repo-review-full-no-comments-progress.{key}.txt"
    progress.parent.mkdir()
    progress.write_text("not-an-integer corrupt\n")
    payload = ReviewPayload.model_validate_json(
        '{"pr_number":1,"repo":"owner/repo","review_body":"Audit.","findings":[]}'
    )

    context = resolve_context(
        payload,
        reviewed_head=head,
        target="PR #1",
        skill_count=1,
        next_step="Done.",
    )

    assert context.unchanged_count == 0


def test_renderer_cli_real_process_writes_report(tmp_path: Path) -> None:
    """Execute the documented script-path entrypoint used by the host.

    :param tmp_path: Temporary payload and sentinel directory.
    """
    payload_path = tmp_path / "findings.json"
    payload_path.write_text(
        json.dumps(
            {
                "pr_number": 2174,
                "repo": "tinaudio/synth-setter",
                "review_body": "No findings.\n\n## Pi review audit\n\nAudit row.",
                "findings": [],
            }
        )
    )
    output_path = tmp_path / f"repo-review-full-no-comments.{'c' * 40}.md"
    script = Path(__file__).resolve().parents[2] / "agent/_shared/pi_review_render.py"

    result = sh.Command(sys.executable)(
        script,
        "--payload",
        payload_path,
        "--target",
        "PR #2174",
        "--reviewed-head",
        str(sh.Command("git")("rev-parse", "HEAD")).strip(),
        "--skill-count",
        "1",
        "--next-step",
        "Done.",
        "--output",
        output_path,
        "--remove-payload",
        _cwd=Path(__file__).resolve().parents[2],
    )

    assert output_path.read_text() in str(result)
    assert f"Sentinel: {output_path}" in str(result)
    assert not payload_path.exists()


def test_render_payload_writes_canonical_sentinel_and_removes_exact_input(
    tmp_path: Path,
) -> None:
    """Drive payload validation through the real file producer and consumer.

    :param tmp_path: Temporary report directory.
    """
    payload_path = tmp_path / "findings.json"
    payload_path.write_text(
        json.dumps(
            {
                "pr_number": 2174,
                "repo": "tinaudio/synth-setter",
                "review_body": "No findings.\n\n## Pi review audit\n\nAudit row.",
                "findings": [],
            }
        )
    )
    output_path = tmp_path / f"repo-review-full-no-comments.{'b' * 40}.md"
    context = RenderContext(
        target="PR #2174",
        head_sha="b" * 40,
        head_ref="fix/review",
        upstream_sha="b" * 40,
        worktree_state="clean",
        unchanged_count=0,
        skill_count=1,
        next_step="Done.",
    )

    report = render_payload(
        payload_path,
        output_path=output_path,
        context=context,
        remove_payload=True,
    )

    assert output_path.read_text() == report
    assert "## Pi review audit" in report
    assert not payload_path.exists()
