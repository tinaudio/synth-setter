"""Git hook installation invariants for fresh worktrees."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(
    command: list[str], cwd: Path, env: dict[str, str], stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run a command in the isolated Git repository.

    :param command: Argument vector to execute.
    :param cwd: Command working directory.
    :param env: Process environment.
    :param stdin: Text supplied to the command's standard input.
    :returns: Completed process with captured text output.
    """
    return subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _initialize_repository(repo: Path) -> None:
    """Create a repository containing an intentionally unformatted file.

    :param repo: Empty directory to initialize.
    """
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "init", "-q"], cwd=repo, check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [git, "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(  # noqa: S603
        [git, "config", "user.name", "Test User"], cwd=repo, check=True
    )
    (repo / "probe.txt").write_text("needs-formatting")
    subprocess.run([git, "add", "probe.txt"], cwd=repo, check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [git, "commit", "-q", "-m", "test: add probe"], cwd=repo, check=True
    )


def _write_pre_commit_fixture(repo: Path, bin_dir: Path) -> dict[str, str]:
    """Configure the production pre-push entry with a deterministic formatter.

    :param repo: Temporary Git repository.
    :param bin_dir: Directory that receives the ``uv`` test adapter.
    :returns: Environment that resolves ``uv run pre-commit`` to this test environment.
    """
    formatter = repo / "format-probe.sh"
    formatter.write_text('#!/usr/bin/env bash\nprintf "\\n" >> "$1"\n')
    formatter.chmod(formatter.stat().st_mode | stat.S_IXUSR)
    (repo / ".pre-commit-config.yaml").write_text(
        """repos:
  - repo: local
    hooks:
      - id: pre-push-all-files
        name: Run all pre-commit hooks before push
        entry: uv run pre-commit run --all-files
        language: system
        pass_filenames: false
        always_run: true
        stages: [pre-push]
      - id: format-probe
        name: Format probe
        entry: ./format-probe.sh
        language: system
        files: ^probe\\.txt$
        stages: [pre-commit]
"""
    )
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f'#!/usr/bin/env bash\n[[ "$1 $2" == "run pre-commit" ]] || exit 2\n'
        f'shift 2\nexec "{sys.executable}" -m pre_commit "$@"\n'
    )
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def test_worktree_setup_installs_both_hooks_and_blocks_formatter_changes(tmp_path: Path) -> None:
    """Installed pre-push enforcement rejects and materializes required formatting.

    :param tmp_path: Isolated filesystem root supplied by pytest.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _initialize_repository(repo)
    env = _write_pre_commit_fixture(repo, tmp_path / "bin")

    install = _run(["make", "-f", str(_PROJECT_ROOT / "Makefile"), "install-git-hooks"], repo, env)

    assert install.returncode == 0, install.stderr
    pre_commit_hook = repo / ".git" / "hooks" / "pre-commit"
    pre_push_hook = repo / ".git" / "hooks" / "pre-push"
    assert pre_commit_hook.is_file() and os.access(pre_commit_hook, os.X_OK)
    assert pre_push_hook.is_file() and os.access(pre_push_hook, os.X_OK)

    head = _run([shutil.which("git") or "git", "rev-parse", "HEAD"], repo, env)
    assert head.returncode == 0
    ref_update = f"refs/heads/main {head.stdout.strip()} refs/heads/main {'0' * 40}\n"
    push = _run([str(pre_push_hook), "origin", "unused"], repo, env, ref_update)

    assert push.returncode != 0
    assert "files were modified by this hook" in push.stdout
    assert (repo / "probe.txt").read_text() == "needs-formatting\n"


def test_pre_push_stage_uses_the_all_files_enforcement_entry() -> None:
    """The committed configuration retains the behavior exercised above."""
    config = (_PROJECT_ROOT / ".pre-commit-config.yaml").read_text()

    assert "id: pre-push-all-files" in config
    assert "entry: uv run pre-commit run --all-files" in config
    assert "stages: [pre-push]" in config
