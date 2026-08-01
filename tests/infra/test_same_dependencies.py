"""Dependency-placement contracts for embedding model runtimes."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypedDict, cast

import pytest

from synth_setter.pipeline.data.matpac_plus import TINYMU_PACKAGE_COMMIT
from synth_setter.pipeline.data.meanaudio import MEANAUDIO_PACKAGE_COMMIT

_TINYMU_REQUIREMENT = f"tinymu @ git+https://github.com/ktinubu/TinyMU@{TINYMU_PACKAGE_COMMIT}"
_MEANAUDIO_REQUIREMENT = (
    f"meanaudio @ git+https://github.com/xiquan-li/MeanAudio.git@{MEANAUDIO_PACKAGE_COMMIT}"
)
_MEANAUDIO_REQUIRES_DIST = [
    "einops>=0.6",
    "huggingface-hub>=0.26",
    "librosa>=0.8.1",
    "numpy>=1.21",
    "omegaconf",
    "torch>=2.5.1",
]
_SA3_REQUIREMENT = (
    "stable-audio-3 @ "
    "git+https://github.com/Stability-AI/stable-audio-3@"
    "124e8a799f57a1f665495ecb72e547d0a62867f1"
)

_DependencyMetadata = TypedDict(
    "_DependencyMetadata",
    {"name": str, "version": str, "requires-dist": list[str]},
)


class _NamedPackage(TypedDict):
    name: str


_UvTable = TypedDict("_UvTable", {"dependency-metadata": list[_DependencyMetadata]})


class _ToolTable(TypedDict):
    uv: _UvTable


_ProjectTable = TypedDict(
    "_ProjectTable",
    {"dependencies": list[str], "optional-dependencies": dict[str, list[str]]},
)
_Pyproject = TypedDict(
    "_Pyproject",
    {
        "dependency-groups": dict[str, list[str]],
        "project": _ProjectTable,
        "tool": _ToolTable,
    },
)
_Manifest = TypedDict("_Manifest", {"dependency-metadata": list[_DependencyMetadata]})


class _Lock(TypedDict):
    manifest: _Manifest
    package: list[_NamedPackage]


@pytest.fixture(scope="module")
def pyproject(project_root: Path) -> _Pyproject:
    """Load the project metadata under test.

    :param project_root: Repository root containing ``pyproject.toml``.
    :returns: Parsed project metadata.
    """
    with (project_root / "pyproject.toml").open("rb") as file:
        return cast("_Pyproject", tomllib.load(file))


def test_sa3_requirement_is_in_torch_group_not_project_or_extras(
    pyproject: _Pyproject,
) -> None:
    """SA3 ships with the normal heavy runtime rather than a feature extra.

    :param pyproject: Parsed project metadata.
    """
    assert _SA3_REQUIREMENT in pyproject["dependency-groups"]["torch"]
    assert _SA3_REQUIREMENT not in pyproject["project"]["dependencies"]
    assert set(pyproject["project"]["optional-dependencies"]) == {"cpu", "cu128"}


def test_tinymu_package_is_pinned_in_normal_torch_runtime(
    project_root: Path, pyproject: _Pyproject
) -> None:
    """MATPAC installs from one immutable package commit in the standard runtime.

    :param project_root: Repository root containing ``uv.lock``.
    :param pyproject: Parsed project metadata.
    """
    assert _TINYMU_REQUIREMENT in pyproject["dependency-groups"]["torch"]
    lock_text = (project_root / "uv.lock").read_text()
    assert (
        'source = { git = "https://github.com/ktinubu/TinyMU?rev='
        f"{TINYMU_PACKAGE_COMMIT}#{TINYMU_PACKAGE_COMMIT}"
        '" }'
    ) in lock_text


def test_meanaudio_package_and_metadata_override_match_lock(
    project_root: Path, pyproject: _Pyproject
) -> None:
    """MeanAudio is direct, immutable, and limited to the VAE import closure.

    :param project_root: Repository root containing ``uv.lock``.
    :param pyproject: Parsed project metadata.
    """
    assert _MEANAUDIO_REQUIREMENT in pyproject["dependency-groups"]["torch"]
    assert _MEANAUDIO_REQUIREMENT not in pyproject["project"]["dependencies"]
    metadata = next(
        item
        for item in pyproject["tool"]["uv"]["dependency-metadata"]
        if item["name"] == "meanaudio"
    )
    assert metadata == {
        "name": "meanaudio",
        "version": "1.0.0",
        "requires-dist": _MEANAUDIO_REQUIRES_DIST,
    }

    lock_text = (project_root / "uv.lock").read_text()
    lock = cast("_Lock", tomllib.loads(lock_text))
    lock_metadata = next(
        item for item in lock["manifest"]["dependency-metadata"] if item["name"] == "meanaudio"
    )
    assert lock_metadata == metadata
    assert any(package["name"] == "meanaudio" for package in lock["package"])
    assert (
        'source = { git = "https://github.com/xiquan-li/MeanAudio.git?rev='
        f"{MEANAUDIO_PACKAGE_COMMIT}#{MEANAUDIO_PACKAGE_COMMIT}"
        '" }'
    ) in lock_text


def test_legacy_same_runtime_is_absent_from_metadata_and_lock(
    project_root: Path, pyproject: _Pyproject
) -> None:
    """The retired legacy runtime has no active dependency or lock metadata.

    :param project_root: Repository root containing ``uv.lock``.
    :param pyproject: Parsed project metadata.
    """
    lock_text = (project_root / "uv.lock").read_text()
    lock = cast("_Lock", tomllib.loads(lock_text))

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


def test_sa3_metadata_override_is_retained(pyproject: _Pyproject) -> None:
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
