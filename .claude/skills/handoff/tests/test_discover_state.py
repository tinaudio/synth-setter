"""Unit tests for handoff/helpers/discover_state.py — pure functions only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from helpers import discover_state as ds  # noqa: E402


def test_filter_handoff_comments_picks_only_handoff_headers() -> None:
    """Filter handoff comments picks only handoff headers."""
    comments = [
        {
            "databaseId": 1,
            "url": "u1",
            "createdAt": "2026-05-10T00:00:00Z",
            "body": "## Handoff update — A",
        },
        {
            "databaseId": 2,
            "url": "u2",
            "createdAt": "2026-05-09T00:00:00Z",
            "body": "Some unrelated comment.",
        },
        {
            "databaseId": 3,
            "url": "u3",
            "createdAt": "2026-05-11T00:00:00Z",
            "body": "## Handoff update — B",
        },
        {
            "databaseId": 4,
            "url": "u4",
            "createdAt": "2026-05-08T00:00:00Z",
            "body": "## Other heading",
        },
    ]
    handoffs = ds.filter_handoff_comments(comments)
    assert [h.comment_id for h in handoffs] == [3, 1]
    assert handoffs[0].url == "u3"
    assert handoffs[0].created_at > handoffs[1].created_at


def test_filter_handoff_comments_empty_input_returns_empty() -> None:
    """Filter handoff comments empty input returns empty."""
    assert ds.filter_handoff_comments([]) == []


def test_extract_task_numbers_matches_prefix_and_sorts() -> None:
    """Extract task numbers matches prefix and sorts."""
    titles = [
        "Task 5.1: Schemas",
        "Task 5.10: WDS Writer",
        "Task 5.3: Code Health",
        "Task 6.1: Different Phase",  # ignored
        "Bug: unrelated",
    ]
    assert ds.extract_task_numbers(titles, "Task 5") == [1, 3, 10]


def test_extract_task_numbers_no_matches_returns_empty() -> None:
    """Extract task numbers no matches returns empty."""
    assert ds.extract_task_numbers(["nothing here", "Task 6.1: bar"], "Task 5") == []


def test_phase_numbering_next_free_skips_one_past_max() -> None:
    """Phase numbering next free skips one past max."""
    numbering = ds.PhaseTaskNumbering(
        phase_number=72, task_prefix="Task 5", used_numbers=[1, 3, 7]
    )
    assert numbering.next_free == 8


def test_phase_numbering_next_free_when_empty_starts_at_one() -> None:
    """Phase numbering next free when empty starts at one."""
    numbering = ds.PhaseTaskNumbering(phase_number=72, task_prefix="Task 5", used_numbers=[])
    assert numbering.next_free == 1


def test_parse_worktree_porcelain_parses_typical_output() -> None:
    """Parse worktree porcelain parses typical output."""
    text = (
        "worktree /home/build/synth-setter\n"
        "HEAD abc123\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /home/build/synth-setter/.claude.worktrees/pr-4-render-config\n"
        "HEAD def456\n"
        "branch refs/heads/pr-4-render-config\n"
        "\n"
        "worktree /tmp/detached\n"
        "HEAD 789abc\n"
        "detached\n"
        "\n"
    )
    entries = ds.parse_worktree_porcelain(text)
    assert len(entries) == 3
    assert entries[0]["branch"] == "main"
    assert entries[1]["branch"] == "pr-4-render-config"
    assert entries[2].get("detached") is True
    assert "branch" not in entries[2]


def test_parse_worktree_porcelain_empty_returns_empty() -> None:
    """Parse worktree porcelain empty returns empty."""
    assert ds.parse_worktree_porcelain("") == []


def test_annotate_worktrees_marks_merged_safe_to_remove() -> None:
    """Annotate worktrees marks merged safe to remove."""
    raw = [
        {"path": "/a", "head": "h1", "branch": "feat-x"},
        {"path": "/b", "head": "h2", "branch": "feat-y"},
        {"path": "/c", "head": "h3", "branch": "feat-z"},
    ]
    pr_by_branch = {
        "feat-x": (101, "MERGED"),
        "feat-y": (102, "OPEN"),
        # feat-z has no PR
    }
    out = ds.annotate_worktrees(raw, pr_by_branch)
    by_branch = {entry.branch: entry for entry in out}
    assert by_branch["feat-x"].safe_to_remove is True
    assert by_branch["feat-x"].associated_pr_number == 101
    assert by_branch["feat-y"].safe_to_remove is False
    assert by_branch["feat-z"].safe_to_remove is False
    assert by_branch["feat-z"].associated_pr_number is None


def test_count_copilot_after_filters_by_login_and_timestamp() -> None:
    """Count copilot after filters by login and timestamp."""
    comments = [
        {"login": "copilot", "created_at": "2026-05-11T00:00:00Z"},
        {"login": "Copilot", "created_at": "2026-05-12T00:00:00Z"},
        {"login": "human-reviewer", "created_at": "2026-05-12T00:00:00Z"},
        {"login": "copilot-pr-reviewer[bot]", "created_at": "2026-05-09T00:00:00Z"},
    ]
    # since_iso == push time; only strictly-after Copilot comments count.
    count = ds.count_copilot_after(comments, "2026-05-10T00:00:00Z")
    assert count == 2


def test_count_copilot_after_zero_when_since_empty() -> None:
    """Count copilot after zero when since empty."""
    comments = [{"login": "copilot", "created_at": "2026-05-12T00:00:00Z"}]
    assert ds.count_copilot_after(comments, "") == 0


def test_summarize_checks_failing_beats_pending_beats_passing() -> None:
    """Summarize checks failing beats pending beats passing."""
    failing = [
        {"conclusion": "SUCCESS"},
        {"conclusion": "FAILURE"},
        {"status": "IN_PROGRESS"},
    ]
    assert ds.summarize_checks(failing) == "failing"

    pending = [
        {"conclusion": "SUCCESS"},
        {"status": "IN_PROGRESS"},
    ]
    assert ds.summarize_checks(pending) == "pending"

    passing = [
        {"conclusion": "SUCCESS"},
        {"conclusion": "NEUTRAL"},
        {"conclusion": "SKIPPED"},
    ]
    assert ds.summarize_checks(passing) == "passing"


def test_summarize_checks_empty_is_passing() -> None:
    """Summarize checks empty is passing."""
    assert ds.summarize_checks([]) == "passing"


def test_count_unresolved_threads_filters_resolved() -> None:
    """Count unresolved threads filters resolved."""
    threads = [
        {"isResolved": True},
        {"isResolved": False},
        {"isResolved": False},
        {},
    ]
    assert ds.count_unresolved_threads(threads) == 3


def test_title_prefix_strips_trailing_pr_number_suffix() -> None:
    """Title prefix strips trailing pr number suffix."""
    assert ds._title_prefix("feat(pipeline): foo (#123)") == "feat(pipeline): foo"
    assert ds._title_prefix("internal-fix(schemas): bar") == "internal-fix(schemas): bar"
    assert ds._title_prefix("Feat(Pipeline): Foo (#42)") == "feat(pipeline): foo"


def test_update_chain_from_state_promotes_merged_and_in_flight() -> None:
    """Update chain from state promotes merged and in flight."""
    chain = ds.Chain(
        tracking_issue=882,
        repo="org/repo",
        parent_phase=72,
        task_prefix="Task 5",
        plan_prs=[
            ds.ChainPR(id="PR-5", title="refactor(pipeline): relocate pipeline → src/pipeline"),
            ds.ChainPR(
                id="PR-6", title="internal-fix(pipeline): post-relocation doc + comment sweep"
            ),
            ds.ChainPR(id="PR-7", title="internal-fix(schemas): tighten validators"),
        ],
    )
    done_since = [
        ds.ClosedPR(
            number=999,
            title="refactor(pipeline): relocate pipeline → src/pipeline (#999)",
            merged_at="2026-05-12T00:00:00Z",
            closes_issues=[],
        )
    ]
    in_flight = [
        ds.InFlightPR(
            number=1001,
            title="internal-fix(pipeline): post-relocation doc + comment sweep",
            branch="b",
            head_oid="h",
            head_committer_date="",
            mergeable="MERGEABLE",
            review_decision="REVIEW_REQUIRED",
            checks_state="passing",
            unresolved_threads=0,
            new_copilot_comments_since_push=0,
        )
    ]
    out = ds.update_chain_from_state(chain, done_since, in_flight)
    by_id = {row.id: row for row in out.plan_prs}
    assert by_id["PR-5"].status == "merged"
    assert by_id["PR-5"].pr_number == 999
    assert by_id["PR-6"].status == "in_flight"
    assert by_id["PR-6"].pr_number == 1001
    assert by_id["PR-7"].status == "pending"
    assert by_id["PR-7"].pr_number is None


def test_chain_round_trip_preserves_fields(tmp_path: Path) -> None:
    """Chain round trip preserves fields."""
    chain = ds.Chain(
        tracking_issue=882,
        repo="org/repo",
        parent_phase=72,
        task_prefix="Task 5",
        plan_prs=[
            ds.ChainPR(
                id="PR-5",
                title="refactor(pipeline): relocate",
                source_commits=["aaa", "bbb"],
                notes=["watch this one"],
                linked_issue=ds.LinkedIssue(strategy="existing", existing=874),
                status="merged",
                pr_number=999,
                depends_on=[],
            ),
            ds.ChainPR(
                id="PR-6",
                title="internal-fix(pipeline): foo",
                depends_on=["PR-5"],
            ),
        ],
    )
    path = tmp_path / "chain.yaml"
    ds.save_chain(path, chain)
    reloaded = ds.load_chain(path)
    assert reloaded.tracking_issue == 882
    assert reloaded.task_prefix == "Task 5"
    assert len(reloaded.plan_prs) == 2
    pr5 = reloaded.plan_prs[0]
    assert pr5.source_commits == ["aaa", "bbb"]
    assert pr5.notes == ["watch this one"]
    assert pr5.linked_issue.strategy == "existing"
    assert pr5.linked_issue.existing == 874
    assert pr5.status == "merged"
    assert pr5.pr_number == 999
    assert reloaded.plan_prs[1].depends_on == ["PR-5"]


def test_fetch_commit_committer_date_uses_raw_jq_string_not_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`gh api --jq` prints scalar values unquoted; helper must use stripped stdout."""
    captured: dict = {}

    def fake_run_gh(args: list[str]) -> str:
        captured["args"] = args
        return "2026-05-11T20:34:56Z\n"

    monkeypatch.setattr(ds, "run_gh", fake_run_gh)
    out = ds._fetch_commit_committer_date("org/repo", "abc1234")
    assert out == "2026-05-11T20:34:56Z"
    assert captured["args"][:2] == ["api", "repos/org/repo/commits/abc1234"]


def test_fetch_commit_committer_date_returns_empty_on_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helper returns "" on `gh` failure rather than propagating CalledProcessError."""
    import subprocess as sp

    def boom(args: list[str]) -> str:
        raise sp.CalledProcessError(returncode=1, cmd=["gh", *args])

    monkeypatch.setattr(ds, "run_gh", boom)
    assert ds._fetch_commit_committer_date("org/repo", "abc1234") == ""


def test_load_chain_reads_the_shipped_chain_yaml() -> None:
    """The chain.yaml shipped with the skill loads cleanly into a Chain."""
    chain = ds.load_chain(SKILL_ROOT / "chain.yaml")
    assert chain.tracking_issue == 882
    assert chain.parent_phase == 72
    assert chain.task_prefix == "Task 5"
    ids = [pr.id for pr in chain.plan_prs]
    assert ids == [
        "PR-5",
        "PR-6",
        "PR-7",
        "PR-8",
        "PR-9",
        "PR-11",
        "PR-12",
        "PR-13",
        "PR-14",
        "PR-15",
    ]
    pr7 = next(pr for pr in chain.plan_prs if pr.id == "PR-7")
    assert any("b516b67" in note for note in pr7.notes)
