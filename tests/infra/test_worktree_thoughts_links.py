"""`make link-thoughts` links a worktree's gitignored thoughts/ to the primary's central one.

`thoughts/` is gitignored, so each `git worktree add` grows its own copy and scatters qrspi docs
(research/structure/plan/design) across worktrees. The target replaces the worktree's thoughts/ with
a symlink to the primary's central thoughts/, migrating any pre-existing files first and preserving a
divergent collision as ``<name>.from-<worktree>`` rather than dropping it. The spawn command printed
by the SessionStart banner and worktree-guard chains it on so the link appears naturally.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.infra

# Bound subprocess calls so a hung git/make can't wedge the suite.
_TIMEOUT_S = 60

for _tool in ("git", "make"):
    if shutil.which(_tool) is None:
        pytest.skip(f"{_tool} not on PATH", allow_module_level=True)


def _git(repo: Path, *args: str) -> None:
    """Run a git subcommand in ``repo``, raising on non-zero exit.

    :param repo: checkout the subcommand runs against (``git -C``).
    :param *args: subcommand and its arguments.
    """
    subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", str(repo), *args],  # noqa: S607 — git on PATH
        check=True,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )


def _make_link_thoughts(cwd: Path) -> str:
    """Run ``make link-thoughts`` in ``cwd``, returning its stdout.

    :param cwd: directory to run the target from (a primary checkout or worktree).
    :returns: the target's stdout (the linked/migrated/no-op message lines).
    """
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["make", "link-thoughts"],  # noqa: S607 — make on PATH
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    ).stdout


def _init_primary_repo(path: Path) -> None:
    """Init a committed git checkout at ``path`` with the project Makefile.

    :param path: empty directory to turn into the primary checkout.
    """
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "test")
    shutil.copy(PROJECT_ROOT / "Makefile", path / "Makefile")
    _git(path, "add", "Makefile")
    _git(path, "commit", "-qm", "init")


def test_link_thoughts_symlinks_fresh_worktree_to_central(tmp_path: Path) -> None:
    """`make link-thoughts` links a fresh worktree's thoughts/ to central, idempotently.

    :param tmp_path: holds the primary checkout and the worktree added off it.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)
    worktree = tmp_path / "wt"
    _git(primary, "worktree", "add", "--detach", "-q", str(worktree))
    assert not (worktree / "thoughts").exists(), (
        "precondition: gitignored thoughts/ absent in fresh worktree"
    )

    _make_link_thoughts(worktree)
    link = worktree / "thoughts"
    assert link.is_symlink()
    assert link.resolve() == (primary / "thoughts").resolve()

    # Re-running must report the existing link, not nest a symlink inside it.
    stdout = _make_link_thoughts(worktree)
    assert "already linked" in stdout
    assert link.is_symlink()
    assert link.resolve() == (primary / "thoughts").resolve()


def test_link_thoughts_is_noop_in_primary(tmp_path: Path) -> None:
    """`make link-thoughts` leaves the primary alone — its thoughts/ is already central.

    :param tmp_path: holds the primary checkout.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)

    stdout = _make_link_thoughts(primary)

    assert "already central" in stdout
    assert not (primary / "thoughts").is_symlink()


def test_link_thoughts_migrates_worktree_files_into_central(tmp_path: Path) -> None:
    """`make link-thoughts` migrates a pre-existing worktree file into central before linking.

    :param tmp_path: holds the primary checkout and the worktree with local thoughts/.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)
    worktree = tmp_path / "wt"
    _git(primary, "worktree", "add", "--detach", "-q", str(worktree))
    doc = worktree / "thoughts" / "shared" / "research" / "doc.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("worktree-only")

    _make_link_thoughts(worktree)

    central_doc = primary / "thoughts" / "shared" / "research" / "doc.md"
    assert central_doc.read_text() == "worktree-only"
    assert (worktree / "thoughts").is_symlink()
    # The migrated file is now reachable through the symlink, not a separate copy.
    assert (
        worktree / "thoughts" / "shared" / "research" / "doc.md"
    ).read_text() == "worktree-only"


def test_link_thoughts_preserves_divergent_collision(tmp_path: Path) -> None:
    """A divergent worktree file survives as ``<name>.from-<worktree>``, never dropped.

    :param tmp_path: holds the primary checkout and the worktree with a colliding doc.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)
    worktree = tmp_path / "wt"
    _git(primary, "worktree", "add", "--detach", "-q", str(worktree))

    rel = Path("shared") / "design" / "doc.md"
    central_doc = primary / "thoughts" / rel
    central_doc.parent.mkdir(parents=True)
    central_doc.write_text("central-version")
    wt_doc = worktree / "thoughts" / rel
    wt_doc.parent.mkdir(parents=True)
    wt_doc.write_text("worktree-version")

    stdout = _make_link_thoughts(worktree)

    assert "collision" in stdout
    assert central_doc.read_text() == "central-version", (
        "central copy is authoritative on collision"
    )
    preserved = primary / "thoughts" / "shared" / "design" / f"doc.md.from-{worktree.name}"
    assert preserved.read_text() == "worktree-version", "divergent worktree copy is preserved"


def test_link_thoughts_skips_identical_collision_quietly(tmp_path: Path) -> None:
    """A worktree file identical to central produces no ``.from-`` copy and no collision notice.

    :param tmp_path: holds the primary checkout and the worktree with an identical doc.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)
    worktree = tmp_path / "wt"
    _git(primary, "worktree", "add", "--detach", "-q", str(worktree))

    rel = Path("shared") / "plans" / "doc.md"
    (primary / "thoughts" / rel).parent.mkdir(parents=True)
    (primary / "thoughts" / rel).write_text("same")
    (worktree / "thoughts" / rel).parent.mkdir(parents=True)
    (worktree / "thoughts" / rel).write_text("same")

    stdout = _make_link_thoughts(worktree)

    assert "collision" not in stdout
    assert not list((primary / "thoughts" / "shared" / "plans").glob("*.from-*"))


def test_link_thoughts_symlink_is_gitignored(tmp_path: Path) -> None:
    """After linking, the worktree's thoughts symlink is gitignored, not a dirty untracked entry.

    The project ignores ``thoughts`` without a trailing slash precisely so this symlink is caught;
    a dir-only ``thoughts/`` pattern misses it, leaving the link untracked and committable.

    :param tmp_path: holds the primary checkout (with .gitignore) and its worktree.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)
    shutil.copy(PROJECT_ROOT / ".gitignore", primary / ".gitignore")
    _git(primary, "add", ".gitignore")
    _git(primary, "commit", "-qm", "gitignore")
    worktree = tmp_path / "wt"
    _git(primary, "worktree", "add", "--detach", "-q", str(worktree))

    _make_link_thoughts(worktree)

    status = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", str(worktree), "status", "--porcelain"],  # noqa: S607 — git on PATH
        check=True,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    ).stdout
    assert (worktree / "thoughts").is_symlink()
    assert "thoughts" not in status, f"symlink should be gitignored; git status: {status!r}"
