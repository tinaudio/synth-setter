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

Public API: ``worktree_for_ref`` (fixture), and the helpers ``git``,
``git_or_warn``, ``ref_exists``, ``try_fetch_ref`` — exported without leading
underscores so other test modules can import them without violating the
"underscore = private" Python convention.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import shutil
import subprocess
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _git_lock(lock_path: Path) -> Iterator[None]:
    """Process-wide exclusive lock around git fetch + worktree operations.

    Without this, multiple xdist workers can race on shared git state
    (``.git/config.lock``, ``FETCH_HEAD.lock``) when concurrently fetching
    refs or registering worktrees. Worker-id-suffixed worktree paths solve
    the path/name collision but don't help with the per-repo locks git
    grabs internally. The lock is held only across the fetch + add block,
    not the whole test, so contention is brief.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def git(*argv: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run ``git -C REPO_ROOT <argv...>``, capture output, return the CompletedProcess.

    Centralizes the shape of every git invocation in this module so the
    ``# noqa: S603, S607`` justification (fixed argv, git on PATH) lives in
    one place. ``check`` defaults to False because most callers inspect
    returncode/stderr themselves to produce specific error messages.
    """
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", str(REPO_ROOT), *argv],  # noqa: S607 — git on PATH
        capture_output=True,
        text=True,
        check=check,
    )


def git_or_warn(*argv: str, context: str) -> subprocess.CompletedProcess[str]:
    """Run ``git(*argv)`` and emit ``warnings.warn`` on non-zero exit.

    Use for best-effort cleanup steps where a failure is interesting (so it
    surfaces in pytest's warnings summary) but shouldn't abort the test.
    ``context`` is a short label included in the warning so a reader can
    tell which call failed.
    """
    result = git(*argv)
    if result.returncode != 0:
        warnings.warn(f"{context}: {result.stderr.strip()}", stacklevel=2)
    return result


def _xdist_worker_id() -> str:
    """Return the pytest-xdist worker id, or ``"master"`` if not under xdist."""
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def _sanitize_ref(ref: str) -> str:
    """Return a filesystem-safe slug for ``ref``, suffixed per xdist worker.

    The suffix matters under ``pytest -n auto`` — without it, every worker
    that requests the same ref produces an identical basename, and
    ``git worktree add`` registers them under the same name in
    ``.git/worktrees/<name>/``, which collides on the second worker. Per-
    worker suffixing gives each worker its own worktree namespace.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ref).strip("-") or "ref"
    worker = _xdist_worker_id()
    return slug if worker == "master" else f"{slug}-{worker}"


def ref_exists(ref: str) -> bool:
    """Return True iff ``ref`` resolves to a commit in the local repo."""
    return git("rev-parse", "--verify", f"{ref}^{{commit}}").returncode == 0


def try_fetch_ref(ref: str) -> list[str]:
    """Best-effort ``git fetch`` to acquire ``ref`` locally; return per-attempt stderr.

    Two-step: first tries ``git fetch origin <ref>``, which works for SHAs and
    branch tips on remotes with ``uploadpack.allowAnySHA1InWant`` (GitHub
    default) — but importantly does NOT create a local ``refs/tags/X`` when
    ``ref`` is a tag name, so tag lookups by name still fail. If the ref is
    still not resolvable, falls back to an explicit tag refspec that does
    create the local tag ref. Returns the trimmed stderr from each attempt
    so the caller can include it in a diagnostic error message.

    No ``--depth=1``: shallow SHA fetches into already-shallow CI clones hit a
    pack-negotiation bug where the server omits subtree objects whose SHAs
    differ from the client's HEAD (it assumes "client has HEAD, probably has
    these subtrees too" — but a depth-1 client has only HEAD's specific tree).
    The fetch then succeeds at returning a commit object, ``ref_exists``
    returns True, but ``git worktree add`` fails with ``unable to read tree``.
    Without ``--depth=1``, git negotiates a complete pack relative to the
    client's haves (still incremental — only sends objects the client lacks).
    """
    stderrs: list[str] = []
    r1 = git("fetch", "origin", ref)
    stderrs.append(r1.stderr.strip() or "(empty)")
    if ref_exists(ref):
        return stderrs
    refspec = f"+refs/tags/{ref}:refs/tags/{ref}"
    r2 = git("fetch", "origin", refspec)
    stderrs.append(r2.stderr.strip() or "(empty)")
    return stderrs


def _is_git_checkout() -> bool:
    """Return True iff REPO_ROOT is inside a git working tree."""
    result = git("rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == "true"


def _git_common_dir() -> Path:
    """Return the shared ``.git`` directory for REPO_ROOT.

    Uses ``git rev-parse --git-common-dir`` so this works correctly when
    REPO_ROOT is a linked worktree (where ``.git`` is a *file* pointing back
    to the main repo's ``.git/worktrees/<name>/`` rather than a directory).
    All linked worktrees of the same repo resolve to the same path here, so
    a lock placed in this directory serializes across worktrees too — not
    just across xdist workers in one checkout.
    """
    return Path(git("rev-parse", "--git-common-dir", check=True).stdout.strip())


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
    # Process-wide lock file in the shared `.git/` (resolved via
    # `git rev-parse --git-common-dir` so this works from a linked worktree
    # where REPO_ROOT/.git is a *file*, not a directory). All linked worktrees
    # of the same repo land on the same lock path, so the lock serializes
    # across worktrees — not just across xdist workers in one checkout.
    lock_path = _git_common_dir() / "baseline_worktree.lock"

    def _get(ref: str) -> Path:
        if ref in cache:
            return cache[ref]

        with _git_lock(lock_path):
            fetch_stderrs: list[str] = []
            if not ref_exists(ref):
                # Try to fetch it from origin (handles CI shallow clones and freshly-
                # cut tags). If fetch fails, raise RuntimeError per-test (not
                # pytest.UsageError, which would abort the whole session).
                fetch_stderrs = try_fetch_ref(ref)
            if not ref_exists(ref):
                attempts = "\n".join(
                    f"  attempt {i + 1} stderr: {s}" for i, s in enumerate(fetch_stderrs)
                )
                raise RuntimeError(
                    f"baseline ref {ref!r} not found locally and `git fetch` did "
                    f"not resolve it.\n{attempts}\n"
                    f"Update the relevant BASELINE constant in "
                    f"tests/test_compare_baseline_configs.py, or check network "
                    f"access to origin."
                )

            # Clear any stale worktree entries left by killed prior runs before adding ours.
            git_or_warn("worktree", "prune", context="git worktree prune")

            path = base_dir / _sanitize_ref(ref)
            # If a previous run with --compare-baseline-configs-keep-yaml-dir left
            # this path behind, remove it before re-adding so `git worktree add`
            # doesn't fail on conflict.
            if path.exists():
                git_or_warn(
                    "worktree",
                    "remove",
                    "--force",
                    str(path),
                    context=f"git worktree remove --force {path}",
                )
                # Defensive: `git worktree remove` no-ops if the path isn't a
                # registered worktree (e.g., directory left behind by an interrupted
                # prior run, or a manually deleted .git/worktrees/X entry). Nuke
                # the directory so the subsequent `git worktree add` doesn't fail
                # with "already exists".
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)

            result = git("worktree", "add", "--detach", str(path), ref)
            if result.returncode != 0:
                raise pytest.UsageError(
                    f"git worktree add failed for ref {ref!r}: {result.stderr.strip()}"
                )

        cache[ref] = path
        return path

    yield _get

    if skip_cleanup:
        return

    with _git_lock(lock_path):
        for path in cache.values():
            git_or_warn(
                "worktree",
                "remove",
                "--force",
                str(path),
                context=f"git worktree remove --force {path}",
            )
