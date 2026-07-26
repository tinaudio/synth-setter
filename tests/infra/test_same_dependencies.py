"""Dependency-placement contracts for the Stable Audio 3 runtime."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

_SA3_REQUIREMENT = (
    "stable-audio-3 @ "
    "git+https://github.com/Stability-AI/stable-audio-3@"
    "124e8a799f57a1f665495ecb72e547d0a62867f1"
)


@pytest.fixture(scope="module")
def pyproject(project_root: Path) -> dict[str, Any]:
    """Load the project metadata under test.

    :param project_root: Repository root containing ``pyproject.toml``.
    :returns: Parsed project metadata.
    """
    with (project_root / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def test_sa3_requirement_is_in_torch_group_not_project_or_extras(
    pyproject: dict[str, Any],
) -> None:
    """SA3 ships with the normal heavy runtime while extras remain backend-only.

    :param pyproject: Parsed project metadata.
    """
    assert _SA3_REQUIREMENT in pyproject["dependency-groups"]["torch"]
    assert _SA3_REQUIREMENT not in pyproject["project"]["dependencies"]
    assert set(pyproject["project"]["optional-dependencies"]) == {"cpu", "cu128"}


def test_stable_audio_tools_is_absent_from_metadata_and_lock(
    project_root: Path, pyproject: dict[str, Any]
) -> None:
    """The retired legacy runtime has no active dependency or lock metadata.

    :param project_root: Repository root containing ``uv.lock``.
    :param pyproject: Parsed project metadata.
    """
    lock_text = (project_root / "uv.lock").read_text()
    lock = tomllib.loads(lock_text)

    assert "stable-audio-tools" not in lock_text
    assert all(
        metadata["name"] != "stable-audio-tools"
        for metadata in pyproject["tool"]["uv"]["dependency-metadata"]
    )
    assert all(package["name"] != "stable-audio-tools" for package in lock["package"])
    assert all(
        metadata["name"] != "stable-audio-tools"
        for metadata in lock["manifest"]["dependency-metadata"]
    )


def test_sa3_metadata_override_is_retained(pyproject: dict[str, Any]) -> None:
    """SA3 keeps its package-scoped relaxed dependency metadata.

    :param pyproject: Parsed project metadata.
    """
    metadata = next(
        item
        for item in pyproject["tool"]["uv"]["dependency-metadata"]
        if item["name"] == "stable-audio-3"
    )

    assert metadata["version"] == "0.1.0"
    assert "torch>=2.7" in metadata["requires-dist"]
    assert "torchaudio>=2.7" in metadata["requires-dist"]
    assert "numpy>=2.2.6" in metadata["requires-dist"]
