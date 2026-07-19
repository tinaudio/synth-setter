"""Behavior tests for deterministic Pi no-comments report rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sh

from agent._shared.pi_review_render import RenderContext, render_markdown, render_payload


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
