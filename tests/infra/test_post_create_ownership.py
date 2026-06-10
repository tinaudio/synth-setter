"""Invariant: post-create.sh corrects bind-mount ownership before it writes .git.

The local devcontainer bind-mounts the host checkout, so a root-owned host tree
lands every file root-owned inside the container. Unless post-create.sh fixes
ownership first, the unprivileged `dev` user can't write `.git`, run
`pre-commit install`, or commit. These are static checks on the script text —
running the script for real has destructive side effects in a populated
workspace (see test_post_create_performance.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _line_index(text: str, pattern: str) -> int:
    """Return the index of the first non-comment line matching `pattern`.

    Only full-line `^\\s*#…` comments are skipped, so a pattern that also appears in the script's
    prose (e.g. `pre-commit install`, named in several comments) matches the actual command line.

    :param text: The script source to scan.
    :param pattern: A regex searched against each non-comment line.
    :returns: 0-based index of the first matching line, or -1 if none match.
    """
    for idx, line in enumerate(text.splitlines()):
        if re.match(r"^\s*#", line):
            continue
        if re.search(pattern, line):
            return idx
    return -1


@pytest.mark.infra
def test_post_create_chowns_workspace_recursively_to_current_user(
    post_create_script: Path,
) -> None:
    """The script recursively chowns the workspace tree to the running user.

    :param post_create_script: Path to `.devcontainer/post-create.sh`.
    """
    text = post_create_script.read_text()
    chown_idx = _line_index(text, r"\bchown\s+-R\b")
    assert chown_idx != -1, (
        "post-create.sh must recursively chown the workspace so a root-owned "
        "host bind mount doesn't leave .git unwritable for the dev user."
    )
    chown_line = text.splitlines()[chown_idx]
    assert re.search(r"\$\(id\s+-u\)", chown_line), (
        "the chown must target the running user — `$(id -u)` — not a literal "
        f"uid; got: {chown_line.strip()!r}"
    )


@pytest.mark.infra
def test_post_create_ownership_fix_precedes_git_writes(
    post_create_script: Path,
) -> None:
    """The chown runs before `pre-commit install`, which writes `.git/hooks`.

    `pre-commit install` is the unambiguous marker for a workspace-`.git`
    write: the dev-block `git config --local/--worktree` loop that also writes
    `.git/config` sits just above it in the same block, so anchoring here pins
    the chown above the whole dev-block git section. (`core.hooksPath` itself
    is a poor anchor — it also appears in the root pre-exec block's
    `--system`/`--global` unset, which never touches the workspace `.git`.)

    :param post_create_script: Path to `.devcontainer/post-create.sh`.
    """
    text = post_create_script.read_text()
    chown_idx = _line_index(text, r"\bchown\s+-R\b")
    precommit_idx = _line_index(text, r"\bpre-commit\s+install\b")
    assert chown_idx != -1, "no chown found in post-create.sh"
    assert precommit_idx != -1, "no `pre-commit install` found in post-create.sh"
    assert chown_idx < precommit_idx, (
        f"chown (line {chown_idx + 1}) must precede `pre-commit install` "
        f"(line {precommit_idx + 1}); otherwise the .git write fails on a "
        "root-owned bind mount before ownership is corrected."
    )


@pytest.mark.infra
def test_post_create_ownership_fix_is_guarded_for_idempotency(
    post_create_script: Path,
) -> None:
    """The chown is conditional so an already-correct rebuild skips the recursive walk.

    An unconditional recursive chown on every container create would burn the post-create time
    budget and churn inode metadata for no reason once the tree is already dev-owned.

    :param post_create_script: Path to `.devcontainer/post-create.sh`.
    """
    text = post_create_script.read_text()
    chown_idx = _line_index(text, r"\bchown\s+-R\b")
    assert chown_idx != -1, "no chown found in post-create.sh"
    preceding = "\n".join(text.splitlines()[:chown_idx])
    assert re.search(r"\bstat\s+-c\s+%u\b", preceding), (
        "the chown must be guarded by an ownership check (stat -c %u) so it "
        "only runs when the workspace is not already owned by the dev user."
    )
