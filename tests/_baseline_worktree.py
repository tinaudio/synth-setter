"""Lazy, session-scoped git-worktree provisioning for ref-comparison tests.

Tests that compare the resolved Hydra config from a known-good baseline ref
against the live tree request the ``worktree_for_ref`` fixture and call it
with a git ref string. The fixture validates the ref exists locally, then
materializes a detached worktree under either ``tmp_path_factory`` (default,
removed at session end) or
``<--compare-baseline-configs-keep-yaml-dir>/worktrees/<sanitized-ref>``
(left in place for inspection).

Per-ref results are cached for the session, so repeated requests for the
same ref incur no additional ``git worktree add`` cost. Worktree creation is
lazy — ``pytest --collect-only`` and any test run that doesn't request the
fixture pays zero git I/O.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sanitize_ref(ref: str) -> str:
    """Return a filesystem-safe slug for ``ref`` suitable as a directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", ref).strip("-") or "ref"


def _ref_exists(ref: str) -> bool:
    """Return True iff ``ref`` resolves to a commit in the local repo."""
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"],  # noqa: S607 — git on PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _is_git_checkout() -> bool:
    """Return True iff REPO_ROOT is inside a git working tree."""
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],  # noqa: S607 — git on PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


@pytest.fixture(scope="session")
def worktree_for_ref(
    pytestconfig: pytest.Config,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[str], Path]]:
    """Yield a callable that materializes a detached worktree for a given ref.

    Lazy and cached per session: the first call for a ref creates the
    worktree, subsequent calls for the same ref return the cached path. No
    git I/O happens until a test actually invokes the callable.

    When ``--compare-baseline-configs-keep-yaml-dir`` is set, worktrees live
    under ``<keep-yaml-dir>/worktrees/<sanitized-ref>`` and are left in place
    at session end for inspection. Otherwise, they live under a tmp path and
    are removed at session end via ``git worktree remove --force``.
    """
    if not _is_git_checkout():
        raise pytest.UsageError(f"REPO_ROOT ({REPO_ROOT}) is not a git checkout")

    keep_dir = pytestconfig.getoption("--compare-baseline-configs-keep-yaml-dir")
    skip_cleanup = keep_dir is not None
    if keep_dir is not None:
        base_dir = Path(keep_dir).resolve() / "worktrees"
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = tmp_path_factory.mktemp("baseline-worktrees")

    cache: dict[str, Path] = {}

    def _get(ref: str) -> Path:
        if ref in cache:
            return cache[ref]

        if not _ref_exists(ref):
            raise pytest.UsageError(
                f"baseline ref {ref!r} not found locally; "
                f"run `git fetch origin {ref}` or override with "
                f"pytest --compare-baseline-configs-baseline-ref=<other>"
            )

        # Clear any stale worktree entries left by killed prior runs before adding ours.
        subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "-C", str(REPO_ROOT), "worktree", "prune"],  # noqa: S607 — git on PATH
            capture_output=True,
            text=True,
            check=False,
        )

        path = base_dir / _sanitize_ref(ref)
        # If a previous run with --compare-baseline-configs-keep-yaml-dir left
        # this path behind, remove it before re-adding so `git worktree add`
        # doesn't fail on conflict.
        if path.exists():
            subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(path)],  # noqa: S607 — git on PATH
                capture_output=True,
                text=True,
                check=False,
            )

        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(path), ref],  # noqa: S607 — git on PATH
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise pytest.UsageError(
                f"git worktree add failed for ref {ref!r}: {result.stderr.strip()}"
            )

        cache[ref] = path
        return path

    yield _get

    if skip_cleanup:
        return

    for path in cache.values():
        subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(path)],  # noqa: S607 — git on PATH
            capture_output=True,
            text=True,
            check=False,
        )
