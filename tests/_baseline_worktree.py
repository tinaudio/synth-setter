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

import os
import re
import shutil
import subprocess
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*argv: str, check: bool = False) -> subprocess.CompletedProcess[str]:
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


def _ref_exists(ref: str) -> bool:
    """Return True iff ``ref`` resolves to a commit in the local repo."""
    return _git("rev-parse", "--verify", f"{ref}^{{commit}}").returncode == 0


def _try_fetch_ref(ref: str) -> list[str]:
    """Best-effort ``git fetch`` to acquire ``ref`` locally; return per-attempt stderr.

    Two-step: first tries ``git fetch --depth=1 origin <ref>``, which works
    for SHAs and branch tips on remotes with ``uploadpack.allowAnySHA1InWant``
    (GitHub default) — but importantly does NOT create a local ``refs/tags/X``
    when ``ref`` is a tag name, so tag lookups by name still fail. If the ref
    still isn't resolvable, falls back to an explicit tag refspec that does
    create the local tag ref. Returns the trimmed stderr from each attempt
    so the caller can include it in a diagnostic error message.
    """
    stderrs: list[str] = []
    r1 = _git("fetch", "--depth=1", "origin", ref)
    stderrs.append(r1.stderr.strip() or "(empty)")
    if _ref_exists(ref):
        return stderrs
    refspec = f"+refs/tags/{ref}:refs/tags/{ref}"
    r2 = _git("fetch", "--depth=1", "origin", refspec)
    stderrs.append(r2.stderr.strip() or "(empty)")
    return stderrs


def _is_git_checkout() -> bool:
    """Return True iff REPO_ROOT is inside a git working tree."""
    result = _git("rev-parse", "--is-inside-work-tree")
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

        fetch_stderrs: list[str] = []
        if not _ref_exists(ref):
            # Try to fetch it from origin (handles CI shallow clones and freshly-
            # cut tags). If fetch fails, raise RuntimeError per-test (not
            # pytest.UsageError, which would abort the whole session).
            fetch_stderrs = _try_fetch_ref(ref)
        if not _ref_exists(ref):
            attempts = "\n".join(
                f"  attempt {i + 1} stderr: {s}" for i, s in enumerate(fetch_stderrs)
            )
            raise RuntimeError(
                f"baseline ref {ref!r} not found locally and `git fetch` did not "
                f"resolve it.\n{attempts}\n"
                f"Update the relevant BASELINE constant in "
                f"tests/test_compare_baseline_configs.py, or check network access "
                f"to origin."
            )

        # Clear any stale worktree entries left by killed prior runs before adding ours.
        _git("worktree", "prune")

        path = base_dir / _sanitize_ref(ref)
        # If a previous run with --compare-baseline-configs-keep-yaml-dir left
        # this path behind, remove it before re-adding so `git worktree add`
        # doesn't fail on conflict.
        if path.exists():
            _git("worktree", "remove", "--force", str(path))
            # Defensive: `git worktree remove` no-ops if the path isn't a
            # registered worktree (e.g., directory left behind by an interrupted
            # prior run, or a manually deleted .git/worktrees/X entry). Nuke
            # the directory so the subsequent `git worktree add` doesn't fail
            # with "already exists".
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

        result = _git("worktree", "add", "--detach", str(path), ref)
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
        result = _git("worktree", "remove", "--force", str(path))
        if result.returncode != 0:
            # Surface stale-worktree leakage through pytest's warnings summary
            # (instead of swallowing stderr silently).
            warnings.warn(
                f"git worktree remove --force {path} failed: {result.stderr.strip()}",
                stacklevel=2,
            )
