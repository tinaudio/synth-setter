"""`make link-plugins` mirrors the primary checkout's gitignored plugins/ into a fresh worktree.

`plugins/` is gitignored, so `git worktree add` produces a worktree without the
`plugins/<plugin>.vst3` symlink that configs/CLI resolve relative to cwd. The target backfills it;
the spawn command printed by the SessionStart banner and worktree-guard chains it on so the link
appears naturally.
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


def _make_link_plugins(cwd: Path) -> str:
    """Run ``make link-plugins`` in ``cwd``, returning its stdout.

    :param cwd: directory to run the target from (a primary checkout or worktree).
    :returns: the target's stdout (the linked/no-op message lines).
    """
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["make", "link-plugins"],  # noqa: S607 — make on PATH
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


@pytest.fixture
def primary_with_plugin(tmp_path: Path) -> tuple[Path, Path]:
    """Build a committed primary checkout whose untracked plugins/ holds one symlinked VST.

    Mirrors the real layout: the entry is a symlink to an out-of-tree target (as
    the primary's `Surge XT.vst3 -> /usr/lib/vst3/...` is), and plugins/ is
    untracked so a fresh worktree won't receive it.

    :param tmp_path: holds the throwaway checkout, its worktrees, and the VST target.
    :returns: ``(primary_root, vst_target)`` — the checkout and the file the
        mirrored symlink must ultimately resolve to.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)

    vst_target = tmp_path / "system" / "Surge XT.vst3"
    vst_target.parent.mkdir()
    vst_target.write_text("vst")
    (primary / "plugins").mkdir()
    (primary / "plugins" / "Surge XT.vst3").symlink_to(vst_target)
    return primary, vst_target


def test_link_plugins_mirrors_primary_into_worktree(
    primary_with_plugin: tuple[Path, Path], tmp_path: Path
) -> None:
    """`make link-plugins` symlinks each primary plugins/ entry into the worktree, idempotently.

    :param primary_with_plugin: ``(primary_root, vst_target)`` fixture.
    :param tmp_path: holds the worktree added off the primary.
    """
    primary, vst_target = primary_with_plugin
    worktree = tmp_path / "wt"
    _git(primary, "worktree", "add", "--detach", "-q", str(worktree))
    assert not (worktree / "plugins").exists(), (
        "precondition: gitignored plugins/ absent in fresh worktree"
    )

    _make_link_plugins(worktree)
    linked = worktree / "plugins" / "Surge XT.vst3"
    assert linked.is_symlink()
    assert linked.resolve() == vst_target.resolve()

    # `ln -sfn` must overwrite the stale link rather than fail or nest into it.
    _make_link_plugins(worktree)
    assert linked.is_symlink()
    assert linked.resolve() == vst_target.resolve()


def test_link_plugins_is_noop_in_primary(primary_with_plugin: tuple[Path, Path]) -> None:
    """`make link-plugins` leaves the primary's plugins/ untouched.

    :param primary_with_plugin: ``(primary_root, vst_target)`` fixture.
    """
    primary, vst_target = primary_with_plugin
    plugins = primary / "plugins"
    before = sorted(p.name for p in plugins.iterdir())

    stdout = _make_link_plugins(primary)

    assert "nothing to link" in stdout
    assert sorted(p.name for p in plugins.iterdir()) == before
    assert (plugins / "Surge XT.vst3").resolve() == vst_target.resolve()


def test_link_plugins_skips_when_primary_has_no_plugins(tmp_path: Path) -> None:
    """`make link-plugins` exits cleanly without creating plugins/ when the primary has none.

    :param tmp_path: holds the primary checkout (no plugins/) and its worktree.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)
    worktree = tmp_path / "wt"
    _git(primary, "worktree", "add", "--detach", "-q", str(worktree))

    stdout = _make_link_plugins(worktree)

    assert "No" in stdout and "install-surge-xt" in stdout
    assert not (worktree / "plugins").exists()


def test_link_plugins_mirrors_a_broken_symlink(tmp_path: Path) -> None:
    """`make link-plugins` mirrors a primary entry even when its symlink target is missing.

    A primary whose system VST isn't installed has a dangling `plugins/<x>.vst3`; the worktree
    should still receive the (equally dangling) link, not silently skip it.

    :param tmp_path: holds the primary checkout (dangling plugin symlink) and its worktree.
    """
    primary = tmp_path / "primary"
    _init_primary_repo(primary)
    (primary / "plugins").mkdir()
    (primary / "plugins" / "Surge XT.vst3").symlink_to(tmp_path / "absent" / "Surge XT.vst3")
    worktree = tmp_path / "wt"
    _git(primary, "worktree", "add", "--detach", "-q", str(worktree))

    _make_link_plugins(worktree)

    linked = worktree / "plugins" / "Surge XT.vst3"
    assert linked.is_symlink()
    assert not linked.exists(), (
        "target is still absent — the dangling link is mirrored, not resolved"
    )
