"""Release commits keep uv.lock's own-package version current."""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest


@pytest.fixture(scope="module", name="project_version")
def project_version_fixture(project_root: Path) -> str:
    """Read the declared project version.

    :param project_root: Repository root fixture.
    :returns: Declared synth-setter version.
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    return str(pyproject["project"]["version"])


def _release_tool_env(tmp_path: Path) -> tuple[dict[str, str], str, str]:
    """Create a PATH where the release build must install uv.

    :param tmp_path: Isolated directory for the pip-installed uv link.
    :returns: Sandboxed environment, git path, and Semantic Release path.
    """
    uv = shutil.which("uv")
    git = shutil.which("git")
    semantic_release = shutil.which("semantic-release")
    assert uv is not None
    assert git is not None
    assert semantic_release is not None

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pip_shim = bin_dir / "pip"
    pip_shim.write_text(
        f'#!/bin/sh\ncase "$*" in "install uv=="*) ln -s "{uv}" "{bin_dir}/uv";; *) exit 1;; esac\n'
    )
    pip_shim.chmod(0o755)

    env = os.environ | {"PATH": f"{bin_dir}:{Path(git).parent}:/usr/bin:/bin"}
    assert shutil.which("uv", path=env["PATH"]) is None
    return env, git, semantic_release


def _initialize_release_repo(
    tmp_path: Path, project_version: str, tools: tuple[dict[str, str], str, str]
) -> None:
    """Seed the tagged history consumed by Semantic Release.

    :param tmp_path: Repository containing copied release inputs.
    :param project_version: Version recorded by the baseline tag.
    :param tools: Sandboxed environment and resolved tool paths.
    """
    env, git, _ = tools
    commands = (
        [git, "init", "--initial-branch=main"],
        [git, "config", "user.name", "Release Test"],
        [git, "config", "user.email", "release-test@example.com"],
        [git, "remote", "add", "origin", "https://github.com/tinaudio/synth-setter.git"],
        [git, "add", "README.md", "pyproject.toml", "uv.lock"],
        [git, "commit", "-m", "chore: baseline"],
        [git, "tag", f"v{project_version}"],
    )
    for command in commands:
        subprocess.run(command, cwd=tmp_path, env=env, check=True)  # noqa: S603


def test_uv_lock_records_current_project_version(project_root: Path, project_version: str) -> None:
    """Guard clean-checkout uv commands against release-created lock drift.

    :param project_root: Checkout containing the authoritative lock.
    :param project_version: Version that the lock must reproduce.
    """
    with (project_root / "uv.lock").open("rb") as fh:
        lock = tomllib.load(fh)

    own_entry = next(package for package in lock["package"] if package["name"] == "synth-setter")

    assert own_entry["version"] == project_version


def test_semantic_release_commits_lock_matching_version_stamp(
    project_root: Path,
    project_version: str,
    tmp_path: Path,
) -> None:
    """Exercise v9 locally because release commits skip the lock-check workflow.

    :param project_root: Checkout supplying the release inputs.
    :param project_version: Version used to seed the release tag.
    :param tmp_path: Isolated repository that protects the checkout from release writes.
    """
    for filename in ("README.md", "pyproject.toml", "uv.lock"):
        shutil.copy2(project_root / filename, tmp_path / filename)

    tools = _release_tool_env(tmp_path)
    env, git, semantic_release = tools
    _initialize_release_repo(tmp_path, project_version, tools)

    subprocess.run(  # noqa: S603 — resolved release binary and fixed argv
        [
            semantic_release,
            "version",
            "--patch",
            "--no-push",
            "--no-vcs-release",
            "--no-changelog",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    assert shutil.which("uv", path=env["PATH"]) == str(tmp_path / "bin" / "uv")

    with (tmp_path / "pyproject.toml").open("rb") as fh:
        released_pyproject = tomllib.load(fh)
    with (tmp_path / "uv.lock").open("rb") as fh:
        lock = tomllib.load(fh)
    own_entry = next(package for package in lock["package"] if package["name"] == "synth-setter")

    assert own_entry["version"] == released_pyproject["project"]["version"]
    committed_files = subprocess.run(  # noqa: S603 — resolved git binary and fixed show argv
        [git, "show", "--pretty=", "--name-only", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "uv.lock" in committed_files
