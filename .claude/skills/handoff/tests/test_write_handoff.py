"""Unit tests for handoff/helpers/write_handoff.py — pure rendering only."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(SKILL_ROOT / "helpers"))

from helpers import discover_state as ds  # noqa: E402
from helpers import write_handoff as wh  # noqa: E402


def _minimal_context(**overrides) -> wh.HandoffContext:
    chain = ds.Chain(
        tracking_issue=882,
        repo="tinaudio/synth-setter",
        parent_phase=72,
        task_prefix="Task 5",
        plan_prs=[
            ds.ChainPR(
                id="PR-5",
                title="refactor(pipeline): relocate pipeline → src/pipeline",
                source_commits=["abc1234"],
            ),
            ds.ChainPR(
                id="PR-6",
                title="internal-fix(pipeline): post-relocation doc sweep",
                source_commits=["def5678"],
                depends_on=["PR-5"],
            ),
        ],
    )
    base = wh.HandoffContext(
        repo="tinaudio/synth-setter",
        tracking_issue=882,
        chain=chain,
        prior_handoffs=[],
        done_since=[],
        in_flight=[],
        phase_numbering=ds.PhaseTaskNumbering(
            phase_number=72, task_prefix="Task 5", used_numbers=[1, 2, 3]
        ),
        worktrees=[],
        now_utc="2026-05-11 22:00 UTC",
        surprises=[],
        anti_patterns=[],
        handoff_comment_url=None,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_render_comment_renders_first_handoff_header() -> None:
    """Render comment renders first handoff header."""
    out = wh.render(_minimal_context(), "comment.md.j2")
    assert "## Handoff update — chain progress as of 2026-05-11 22:00 UTC" in out
    assert "First handoff on tracking issue #882" in out


def test_render_comment_lists_remaining_chain_table() -> None:
    """Render comment lists remaining chain table."""
    out = wh.render(_minimal_context(), "comment.md.j2")
    assert "| PR-5 | pending" in out
    assert "| PR-6 | pending" in out
    assert "Click to expand the 2-row table" in out


def test_render_comment_includes_done_since_when_merged() -> None:
    """Render comment includes done since when merged."""
    ctx = _minimal_context(
        done_since=[
            ds.ClosedPR(
                number=999,
                title="feat(x): foo (#999)",
                merged_at="2026-05-12T00:00:00Z",
                closes_issues=[908],
            ),
        ]
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "| #999 | `feat(x): foo (#999)` | #908 |" in out


def test_render_comment_omits_done_table_when_empty() -> None:
    """Render comment omits done table when empty."""
    out = wh.render(_minimal_context(), "comment.md.j2")
    assert "_No new merges referencing this tracking issue since the prior handoff._" in out


def test_render_comment_in_flight_has_next_action_when_failing() -> None:
    """Render comment in flight has next action when failing."""
    ctx = _minimal_context(
        in_flight=[
            ds.InFlightPR(
                number=942,
                title="internal-feat(vst): render config CLI",
                branch="pr-4",
                head_oid="899449e0",
                head_committer_date="2026-05-11T20:00:00Z",
                mergeable="MERGEABLE",
                review_decision="REVIEW_REQUIRED",
                checks_state="failing",
                unresolved_threads=0,
                new_copilot_comments_since_push=0,
            )
        ]
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "PR #942" in out
    assert "Fix the failing checks" in out


def test_render_comment_in_flight_next_action_ready_to_merge_when_clean() -> None:
    """Render comment in flight next action ready to merge when clean."""
    ctx = _minimal_context(
        in_flight=[
            ds.InFlightPR(
                number=942,
                title="internal-feat(vst): foo",
                branch="b",
                head_oid="abc",
                head_committer_date="2026-05-11T20:00:00Z",
                mergeable="MERGEABLE",
                review_decision="APPROVED",
                checks_state="passing",
                unresolved_threads=0,
                new_copilot_comments_since_push=0,
            )
        ]
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "Ready to merge" in out


def test_render_comment_in_flight_next_action_rebase_when_conflicting() -> None:
    """In-flight next-action wording when the branch conflicts with base."""
    ctx = _minimal_context(
        in_flight=[
            ds.InFlightPR(
                number=942,
                title="internal-feat(vst): foo",
                branch="b",
                head_oid="abc",
                head_committer_date="",
                mergeable="CONFLICTING",
                review_decision="REVIEW_REQUIRED",
                checks_state="passing",
                unresolved_threads=0,
                new_copilot_comments_since_push=0,
            )
        ]
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "Rebase or merge base, resolve conflicts" in out


def test_render_comment_in_flight_next_action_address_threads_when_unresolved() -> None:
    """In-flight next-action wording when there are open review threads."""
    ctx = _minimal_context(
        in_flight=[
            ds.InFlightPR(
                number=942,
                title="internal-feat(vst): foo",
                branch="b",
                head_oid="abc",
                head_committer_date="",
                mergeable="MERGEABLE",
                review_decision="REVIEW_REQUIRED",
                checks_state="passing",
                unresolved_threads=3,
                new_copilot_comments_since_push=2,
            )
        ]
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "Address the 3 unresolved thread(s) + 2 new Copilot comment(s)" in out
    assert "/pr-review-resolver" in out


def test_render_comment_prior_handoff_breadcrumbs() -> None:
    """Render comment prior handoff breadcrumbs."""
    ctx = _minimal_context(
        prior_handoffs=[
            ds.PriorHandoff(
                comment_id=1,
                url="https://example.com/h1",
                created_at="2026-05-11T19:50:00Z",
                body="## Handoff update — A",
            ),
            ds.PriorHandoff(
                comment_id=2,
                url="https://example.com/h2",
                created_at="2026-05-11T03:18:00Z",
                body="## Handoff update — B",
            ),
        ]
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "Predecessors:" in out
    assert "https://example.com/h1" in out
    assert "https://example.com/h2" in out


def test_render_comment_lists_safe_to_remove_worktrees() -> None:
    """Render comment lists safe to remove worktrees."""
    ctx = _minimal_context(
        worktrees=[
            ds.WorktreeEntry(
                path="/wt/a",
                branch="feat-x",
                head="abc1234",
                associated_pr_number=101,
                associated_pr_state="MERGED",
                safe_to_remove=True,
            ),
            ds.WorktreeEntry(
                path="/wt/b",
                branch="feat-y",
                head="def5678",
                associated_pr_number=102,
                associated_pr_state="OPEN",
                safe_to_remove=False,
            ),
        ]
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "safe to `git worktree remove`" in out
    assert "/wt/a" in out
    assert "/wt/b" in out


def test_render_comment_renders_surprises_with_category_lead() -> None:
    """Render comment renders surprises with category lead."""
    ctx = _minimal_context(
        surprises=[
            wh.Surprise(
                category="Copilot race", note="Maintainer self-push raced with parallel PR plan."
            ),
        ]
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "- **Copilot race:**" in out


def test_render_prompt_includes_first_commands_block() -> None:
    """Render prompt includes first commands block."""
    ctx = _minimal_context(
        in_flight=[
            ds.InFlightPR(
                number=942,
                title="t",
                branch="b",
                head_oid="abc",
                head_committer_date="",
                mergeable="MERGEABLE",
                review_decision="REVIEW_REQUIRED",
                checks_state="passing",
                unresolved_threads=0,
                new_copilot_comments_since_push=0,
            )
        ]
    )
    out = wh.render(ctx, "prompt.md.j2")
    assert "git fetch origin --prune" in out
    assert "gh pr view 942" in out
    assert "git worktree list" in out


def test_render_prompt_default_action_drives_in_flight_first() -> None:
    """Render prompt default action drives in flight first."""
    ctx = _minimal_context(
        in_flight=[
            ds.InFlightPR(
                number=942,
                title="t",
                branch="b",
                head_oid="abc",
                head_committer_date="",
                mergeable="MERGEABLE",
                review_decision="REVIEW_REQUIRED",
                checks_state="passing",
                unresolved_threads=0,
                new_copilot_comments_since_push=0,
            )
        ]
    )
    out = wh.render(ctx, "prompt.md.j2")
    assert "Default: drive the in-flight PR(s) above to merge" in out


def test_render_prompt_mandatory_skills_cycle_listed() -> None:
    """Render prompt mandatory skills cycle listed."""
    out = wh.render(_minimal_context(), "prompt.md.j2")
    assert "/tdd-implementation" in out
    assert "/code-health" in out
    assert "/simplify" in out


def test_render_prompt_anti_patterns_render_each_line() -> None:
    """Render prompt anti patterns render each line."""
    ctx = _minimal_context(anti_patterns=["Don't compose @hydra.main outside the entrypoint."])
    out = wh.render(ctx, "prompt.md.j2")
    assert "Don't compose @hydra.main outside the entrypoint." in out


def test_render_ascii_graph_linear_chain() -> None:
    """Render ascii graph linear chain."""
    rows = [
        ds.ChainPR(id="PR-5", title="t"),
        ds.ChainPR(id="PR-6", title="t", depends_on=["PR-5"]),
        ds.ChainPR(id="PR-7", title="t", depends_on=["PR-6"]),
    ]
    out = wh.render_ascii_graph(rows)
    lines = out.splitlines()
    assert "PR-5  (root)" in lines[0]
    assert "PR-6  ← PR-5" in lines[1]
    assert "PR-7  ← PR-6" in lines[2]


def test_render_ascii_graph_fanout() -> None:
    """Render ascii graph fanout."""
    rows = [
        ds.ChainPR(id="PR-5", title="t"),
        ds.ChainPR(id="PR-6", title="t", depends_on=["PR-5"]),
        ds.ChainPR(id="PR-7", title="t", depends_on=["PR-5"]),
        ds.ChainPR(id="PR-8", title="t", depends_on=["PR-5"]),
    ]
    out = wh.render_ascii_graph(rows)
    assert "PR-5  (root)" in out
    assert "PR-6  ← PR-5" in out
    assert "PR-7  ← PR-5" in out
    assert "PR-8  ← PR-5" in out


def test_render_ascii_graph_empty_says_no_remaining() -> None:
    """Render ascii graph empty says no remaining."""
    assert wh.render_ascii_graph([]) == "(no remaining PRs)"


def test_check_idempotency_guard_blocks_recent_handoff() -> None:
    """Check idempotency guard blocks recent handoff."""
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    handoffs = [
        ds.PriorHandoff(comment_id=1, url="u", created_at=recent, body="## Handoff update — A")
    ]
    msg = wh.check_idempotency_guard(handoffs)
    assert msg is not None
    assert "5min ago" in msg


def test_check_idempotency_guard_allows_old_handoff() -> None:
    """Check idempotency guard allows old handoff."""
    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    handoffs = [
        ds.PriorHandoff(comment_id=1, url="u", created_at=old, body="## Handoff update — A")
    ]
    assert wh.check_idempotency_guard(handoffs) is None


def test_check_idempotency_guard_no_prior_returns_none() -> None:
    """Check idempotency guard no prior returns none."""
    assert wh.check_idempotency_guard([]) is None


def test_parse_surprise_splits_category_and_note() -> None:
    """Parse surprise splits category and note."""
    s = wh.parse_surprise("Copilot race: maintainer self-push raced with parallel PR plan")
    assert s.category == "Copilot race"
    assert s.note == "maintainer self-push raced with parallel PR plan"


def test_parse_surprise_with_no_colon_falls_back_to_other() -> None:
    """Parse surprise with no colon falls back to other."""
    s = wh.parse_surprise("just a free-form note")
    assert s.category == "Other"
    assert s.note == "just a free-form note"


def test_parse_args_rejects_comment_only_and_prompt_only_together() -> None:
    """`--comment-only` and `--prompt-only` are mutually exclusive — argparse must error."""
    with pytest.raises(SystemExit):
        wh.parse_args(["--comment-only", "--prompt-only"])


def test_default_handoffs_dir_is_anchored_under_dot_claude() -> None:
    """Save path must resolve relative to the `.claude/` tree, not CWD."""
    assert wh.DEFAULT_HANDOFFS_DIR.is_absolute()
    assert wh.DEFAULT_HANDOFFS_DIR.name == "handoffs"
    assert wh.DEFAULT_HANDOFFS_DIR.parent.name == ".claude"


def test_save_prompt_locally_writes_under_dot_claude_regardless_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running from a sandbox CWD must not create a stray `.claude/handoffs/` there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wh, "DEFAULT_HANDOFFS_DIR", tmp_path / "out")
    saved = wh.save_prompt_locally("# body", "2026-05-12 10:00 UTC")
    assert saved.parent == tmp_path / "out"
    assert saved.read_text() == "# body"
    assert not (tmp_path / ".claude").exists()


def test_render_comment_chain_complete_does_not_emit_empty_bold_list() -> None:
    """When every plan PR is merged, the Task-numbering section must not render `****`."""
    chain = ds.Chain(
        tracking_issue=882,
        repo="tinaudio/synth-setter",
        parent_phase=72,
        task_prefix="Task 5",
        plan_prs=[ds.ChainPR(id="PR-5", title="t", status="merged", pr_number=900)],
    )
    ctx = wh.HandoffContext(
        repo="tinaudio/synth-setter",
        tracking_issue=882,
        chain=chain,
        prior_handoffs=[],
        done_since=[],
        in_flight=[],
        phase_numbering=ds.PhaseTaskNumbering(
            phase_number=72, task_prefix="Task 5", used_numbers=[1, 2, 3]
        ),
        worktrees=[],
        now_utc="2026-05-11 22:00 UTC",
        surprises=[],
        anti_patterns=[],
        handoff_comment_url=None,
    )
    out = wh.render(ctx, "comment.md.j2")
    assert "****" not in out
    assert "chain is complete" in out
    assert "Task 5.4" in out


def test_render_comment_honors_linked_issue_parent_issue_over_phase() -> None:
    """`linked_issue.parent_issue` overrides `chain.parent_phase` in the remaining table."""
    chain = ds.Chain(
        tracking_issue=882,
        repo="tinaudio/synth-setter",
        parent_phase=72,
        task_prefix="Task 5",
        plan_prs=[
            ds.ChainPR(
                id="PR-15",
                title="ci(workflows): parallel WDS run",
                linked_issue=ds.LinkedIssue(strategy="create_under_phase", parent_issue=874),
            )
        ],
    )
    ctx = _minimal_context(chain=chain)
    out = wh.render(ctx, "comment.md.j2")
    assert "create under #874" in out
    assert "create under #72" not in out


def test_render_prompt_honors_linked_issue_parent_issue_over_phase() -> None:
    """Prompt's `Link to issue` line prefers `parent_issue` over `chain.parent_phase`."""
    chain = ds.Chain(
        tracking_issue=882,
        repo="tinaudio/synth-setter",
        parent_phase=72,
        task_prefix="Task 5",
        plan_prs=[
            ds.ChainPR(
                id="PR-15",
                title="ci(workflows): parallel WDS run",
                linked_issue=ds.LinkedIssue(strategy="create_under_phase", parent_issue=874),
            )
        ],
    )
    ctx = _minimal_context(chain=chain)
    out = wh.render(ctx, "prompt.md.j2")
    assert "create a new Task under #874" in out
    assert "under Phase #72" not in out


def test_compute_level_detects_cycle_with_descriptive_error() -> None:
    """A `chain.yaml` cycle raises ValueError naming the offending path."""
    rows = [
        ds.ChainPR(id="PR-A", title="t", depends_on=["PR-B"]),
        ds.ChainPR(id="PR-B", title="t", depends_on=["PR-A"]),
    ]
    with pytest.raises(ValueError, match=r"Cycle detected.*PR-A.*PR-B|Cycle detected.*PR-B.*PR-A"):
        wh.render_ascii_graph(rows)


def test_main_skips_chain_save_when_issue_override_in_play(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--issue` in play => chain.yaml MUST NOT be rewritten with reconciled rows."""
    chain_path = tmp_path / "chain.yaml"
    original_chain = ds.Chain(
        tracking_issue=882,
        repo="org/repo",
        parent_phase=72,
        task_prefix="Task 5",
        plan_prs=[ds.ChainPR(id="PR-5", title="refactor(pipeline): relocate")],
    )
    ds.save_chain(chain_path, original_chain)
    before_bytes = chain_path.read_bytes()

    # Build a derived state that differs from the manifest (a new pr_number).
    reconciled = ds.Chain(
        tracking_issue=882,
        repo="org/repo",
        parent_phase=72,
        task_prefix="Task 5",
        plan_prs=[
            ds.ChainPR(
                id="PR-5",
                title="refactor(pipeline): relocate",
                status="merged",
                pr_number=12345,
            )
        ],
    )
    fake_state = ds.DerivedState(
        repo="org/repo",
        tracking_issue=999,
        chain=reconciled,
        prior_handoffs=[],
        done_since=[],
        in_flight=[],
        phase_numbering=ds.PhaseTaskNumbering(72, "Task 5", []),
        worktrees=[],
        now_utc="2026-05-12 00:00 UTC",
    )
    monkeypatch.setattr(ds, "derive_state", lambda *a, **kw: fake_state)
    exit_code = wh.main(["--chain", str(chain_path), "--issue", "999", "--dry-run"])
    assert exit_code == 0
    # --dry-run skips save unconditionally; verify the override path also skips
    # without --dry-run.
    posted_url: list[str] = []
    monkeypatch.setattr(wh, "post_comment", lambda *a, **kw: posted_url.append("u") or "u")
    monkeypatch.setattr(wh, "save_prompt_locally", lambda *a, **kw: tmp_path / "prompt.md")
    exit_code2 = wh.main(["--chain", str(chain_path), "--issue", "999"])
    assert exit_code2 == 0
    assert chain_path.read_bytes() == before_bytes


def test_main_catches_calledprocesserror_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gh failure surfaces as a one-line stderr message, not a stack trace."""
    chain_path = tmp_path / "chain.yaml"
    ds.save_chain(
        chain_path,
        ds.Chain(
            tracking_issue=882,
            repo="org/repo",
            parent_phase=72,
            task_prefix="Task 5",
            plan_prs=[],
        ),
    )

    def boom(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ARG001
        raise subprocess.CalledProcessError(
            returncode=7, cmd=["gh", "api", "..."], stderr="HTTP 401: auth required"
        )

    monkeypatch.setattr(ds, "derive_state", boom)
    exit_code = wh.main(["--chain", str(chain_path), "--dry-run"])
    assert exit_code == 7
