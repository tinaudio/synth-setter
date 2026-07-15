"""Release commits keep uv.lock's own-package version current."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

UV_VERSION = "0.11.28"


@pytest.fixture(scope="module", name="pyproject")
def pyproject_fixture(project_root: Path) -> dict:
    """Parse the repository pyproject.toml once per module.

    :param project_root: Repository root fixture.
    :returns: Parsed pyproject.toml contents.
    """
    with (project_root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_uv_lock_records_current_project_version(project_root: Path, pyproject: dict) -> None:
    """The lock's synth-setter entry matches the declared project version.

    :param project_root: Repository root fixture.
    :param pyproject: Parsed pyproject.toml fixture.
    """
    with (project_root / "uv.lock").open("rb") as fh:
        lock = tomllib.load(fh)

    own_entry = next(package for package in lock["package"] if package["name"] == "synth-setter")

    assert own_entry["version"] == pyproject["project"]["version"]


def test_release_build_command_refreshes_uv_lock_with_pinned_uv(
    pyproject: dict,
) -> None:
    """The semantic-release build step regenerates the lock with pinned uv.

    :param pyproject: Parsed pyproject.toml fixture.
    """
    build_command = pyproject["tool"]["semantic_release"]["build_command"]

    assert f"uv=={UV_VERSION}" in build_command
    assert "uv lock" in build_command


def test_release_commit_assets_include_uv_lock(pyproject: dict) -> None:
    """The refreshed lock rides in the release commit.

    :param pyproject: Parsed pyproject.toml fixture.
    """
    assets = pyproject["tool"]["semantic_release"]["assets"]

    assert "uv.lock" in assets
