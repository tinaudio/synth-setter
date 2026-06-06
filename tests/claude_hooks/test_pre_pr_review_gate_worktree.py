"""Tests for the worktree-aware ref/dir resolution in ``pre-pr-review-gate.sh``.

The mandated workflow runs each branch in its own ``git worktree``, but the
gate fires from the primary checkout (on the default branch). These tests build
a hermetic repo with a linked worktree and assert the gate derives both the
ancestry/lag ref and the sentinel's base dir from the command's ``--head``
branch — not the primary ``HEAD``/cwd. Each invocation runs from the primary
checkout to mirror the real PreToolUse hook environment.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOK_PATH = _REPO_ROOT / "agent" / "hooks" / "pre-pr-review-gate.sh"
_SENTINEL_PY = _REPO_ROOT / "agent" / "_shared" / "review_sentinel.py"


def _git(*args: str, cwd: Path) -> str:
    """Run ``git`` in ``cwd`` and return its stripped stdout.

    :param *args: Arguments after the ``git`` executable.
    :param cwd: Working directory the command runs in.
    :returns: Captured stdout with surrounding whitespace stripped.
    """
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=str(cwd),
    ).stdout.strip()


def _sentinel_name(sha: str) -> str:
    """Return the sentinel filename encoding ``sha`` via the shared helper.

    :param sha: 40-char commit SHA to encode in the filename.
    :returns: Basename like ``repo-review-full-no-comments.<sha>.md``.
    """
    return subprocess.run(  # noqa: S603
        ["python3", str(_SENTINEL_PY), "make", sha],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(repo: Path, message: str, *, file_body: str) -> str:
    """Write ``file_body`` to ``f.txt``, commit it, and return the new HEAD SHA.

    :param repo: Repo (or worktree) the commit lands in.
    :param message: Commit subject.
    :param file_body: Content written to the tracked file before committing.
    :returns: The resulting commit SHA.
    """
    (repo / "f.txt").write_text(file_body)
    _git("add", "f.txt", cwd=repo)
    _git("commit", "-m", message, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


def _write_sentinel(directory: Path, sha: str, body: str = "# review PASS\n") -> Path:
    """Write a ≥200-byte sentinel for ``sha`` under ``directory``.

    :param directory: Directory the ``.agent-reviews`` tree is created in.
    :param sha: SHA the sentinel filename encodes.
    :param body: Sentinel content; padded to clear the 200-byte stub guard.
    :returns: Path to the written sentinel.
    """
    reviews = directory / ".agent-reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    sentinel = reviews / _sentinel_name(sha)
    sentinel.write_text(body + ("padding\n" * 40))
    return sentinel


def _make_repo_with_worktree(tmp_path: Path) -> tuple[Path, Path, str]:
    """Build a primary repo on ``main`` plus a linked worktree on ``feat/x``.

    The worktree's branch tip is one commit ahead of ``main``, so a sentinel
    encoding the worktree tip is NOT reachable from the primary ``HEAD`` —
    exactly the case the legacy gate mishandled.

    :param tmp_path: pytest tmp dir the repo and worktree are created under.
    :returns: ``(primary_root, worktree_root, worktree_tip_sha)``.
    """
    primary = tmp_path / "primary"
    primary.mkdir()
    _git("init", "-q", "-b", "main", cwd=primary)
    _git("config", "user.email", "t@t.t", cwd=primary)
    _git("config", "user.name", "t", cwd=primary)
    _commit(primary, "root", file_body="root\n")

    worktree = tmp_path / "wt"
    _git("worktree", "add", "-q", "-b", "feat/x", str(worktree), "HEAD", cwd=primary)
    tip = _commit(worktree, "branch work", file_body="branch\n")
    return primary, worktree, tip


def _run_gate(
    primary: Path,
    command: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the hook from ``primary`` with ``command`` as the Bash tool input.

    ``PR_TITLE_GATE``/``REVIEW_COMMENT_GATE``/``REVIEW_BLOCK_GATE`` are off so
    only the ref/dir-resolution logic under test decides the outcome.

    :param primary: Primary-checkout cwd the hook runs from (mirrors the real
        PreToolUse environment, always on the default branch).
    :param command: The ``gh pr create`` command the gate inspects.
    :param env: Extra environment variables overlaid on the inherited env.
    :returns: Completed subprocess result with captured stdout/stderr and code.
    """
    payload: dict[str, Any] = {"tool_input": {"command": command}}
    overlay = {
        "PR_TITLE_GATE": "off",
        "REVIEW_COMMENT_GATE": "off",
        "REVIEW_BLOCK_GATE": "off",
    }
    if env:
        overlay.update(env)
    return subprocess.run(  # noqa: S603
        ["bash", str(_HOOK_PATH)],  # noqa: S607
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(primary),
        env={**os.environ, **overlay},
    )


def test_gate_allows_worktree_branch_sentinel_at_branch_tip(tmp_path: Path) -> None:
    """A relative sentinel in the ``--head`` worktree at its tip is allowed (exit 0).

    The sentinel SHA equals the worktree branch tip (lag 0). It lives only in
    the worktree's ``.agent-reviews``, not in the primary cwd — proving the
    gate resolved both the ref AND the base dir from ``--head``.

    :param tmp_path: pytest tmp dir for the synthetic repo.
    """
    primary, worktree, tip = _make_repo_with_worktree(tmp_path)
    sentinel = _write_sentinel(worktree, tip)
    rel = sentinel.relative_to(worktree)
    command = f"gh pr create --head feat/x --title x --body y  # REVIEW_FULL={rel}"
    result = _run_gate(primary, command)
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_gate_blocks_when_sentinel_sha_not_ancestor_of_branch_tip(tmp_path: Path) -> None:
    """A sentinel on a divergent ``main`` commit (not on ``feat/x``) is blocked (exit 2).

    A fresh ``main`` commit made after the branch forked is off the branch's
    history, so resolving ancestry against the branch (not HEAD) must reject it
    even though the legacy gate — checking against the primary ``HEAD`` — would
    have allowed it.

    :param tmp_path: pytest tmp dir for the synthetic repo.
    """
    primary, worktree, _tip = _make_repo_with_worktree(tmp_path)
    # Advance main past the fork point; this SHA is not on feat/x's line.
    divergent = _commit(primary, "main moves on", file_body="main2\n")
    # Sentinel lives in the worktree (so the path resolves) but encodes the
    # divergent main SHA, which is off the branch's first-parent line.
    sentinel = _write_sentinel(worktree, divergent)
    rel = sentinel.relative_to(worktree)
    command = f"gh pr create --head feat/x --title x --body y  # REVIEW_FULL={rel}"
    result = _run_gate(primary, command)
    assert result.returncode == 2, (result.returncode, result.stdout)
    assert "is not an ancestor of branch feat/x" in result.stderr


def test_gate_blocks_when_lag_exceeds_max(tmp_path: Path) -> None:
    """A sentinel two first-parent commits behind the branch tip blocks at lag 1 (exit 2).

    :param tmp_path: pytest tmp dir for the synthetic repo.
    """
    primary, worktree, _tip = _make_repo_with_worktree(tmp_path)
    behind = _git("rev-parse", "HEAD~1", cwd=worktree)
    # HEAD~1 is one commit back; add one more so the sentinel is 2 behind.
    _commit(worktree, "more work", file_body="more\n")
    sentinel = _write_sentinel(worktree, behind)
    rel = sentinel.relative_to(worktree)
    command = f"gh pr create --head feat/x --title x --body y  # REVIEW_FULL={rel}"
    result = _run_gate(primary, command, env={"REVIEW_MAX_LAG": "1"})
    assert result.returncode == 2, (result.returncode, result.stdout)
    assert "first-parent commits behind branch feat/x" in result.stderr


def test_gate_without_head_flag_uses_primary_head(tmp_path: Path) -> None:
    """Absent ``--head``, the gate keeps the legacy HEAD/cwd path (exit 0).

    The sentinel encodes the primary HEAD and sits in the primary cwd's
    ``.agent-reviews`` — the pre-fix behavior, unchanged.

    :param tmp_path: pytest tmp dir for the synthetic repo.
    """
    primary, _worktree, _tip = _make_repo_with_worktree(tmp_path)
    main_head = _git("rev-parse", "HEAD", cwd=primary)
    sentinel = _write_sentinel(primary, main_head)
    rel = sentinel.relative_to(primary)
    command = f"gh pr create --title x --body y  # REVIEW_FULL={rel}"
    result = _run_gate(primary, command)
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_gate_with_unresolvable_head_stays_strict(tmp_path: Path) -> None:
    """An unknown ``--head`` branch falls back to HEAD/cwd, not a loosened gate (exit 2).

    The branch doesn't exist, so the ref stays ``HEAD`` (primary ``main``) and
    the relative path resolves against cwd. The worktree sentinel is therefore
    unreachable, and the gate blocks — never letting the PR through.

    :param tmp_path: pytest tmp dir for the synthetic repo.
    """
    primary, worktree, tip = _make_repo_with_worktree(tmp_path)
    sentinel = _write_sentinel(worktree, tip)
    rel = sentinel.relative_to(worktree)
    command = f"gh pr create --head feat/does-not-exist --title x --body y  # REVIEW_FULL={rel}"
    result = _run_gate(primary, command)
    assert result.returncode == 2, (result.returncode, result.stdout)
    assert "does not point at a file" in result.stderr
