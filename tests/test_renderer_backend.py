"""Tests for the render-artifact pre-flight shared by the render config and probe wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.renderer_backend import (
    FAUST_PLUGIN_NAME,
    SURGEPY_PLUGIN_NAME,
    TORCHSYNTH_PLUGIN_NAME,
    missing_render_artifacts,
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from an empty workspace so declared relative paths resolve there.

    :param tmp_path: Workspace root.
    :param monkeypatch: Switches the process CWD to it.
    :returns: The workspace root.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_missing_render_artifacts_resolvable_paths_returns_nothing(workspace: Path) -> None:
    """Both artifacts present relative to the CWD leaves nothing to report.

    :param workspace: CWD the relative paths resolve against.
    """
    (workspace / "plugins" / "Stub.vst3").mkdir(parents=True)
    (workspace / "preset.vstpreset").write_bytes(b"stub")

    assert missing_render_artifacts("plugins/Stub.vst3", "preset.vstpreset") == ()


def test_missing_render_artifacts_absent_bundle_is_reported_as_declared(workspace: Path) -> None:
    """A bundle path that resolves nowhere is reported verbatim, not absolutized.

    :param workspace: CWD the relative paths resolve against.
    """
    (workspace / "preset.vstpreset").write_bytes(b"stub")

    assert missing_render_artifacts("plugins/Stub.vst3", "preset.vstpreset") == (
        "plugins/Stub.vst3",
    )


def test_missing_render_artifacts_absent_preset_is_reported(workspace: Path) -> None:
    """A declared preset the renderer would fail to open is reported.

    :param workspace: CWD the relative paths resolve against.
    """
    (workspace / "plugins" / "Stub.vst3").mkdir(parents=True)

    assert missing_render_artifacts("plugins/Stub.vst3", "preset.vstpreset") == (
        "preset.vstpreset",
    )


def test_missing_render_artifacts_reports_both_in_declaration_order(workspace: Path) -> None:
    """Both artifacts absent yields the bundle before the preset.

    :param workspace: Empty CWD the relative paths resolve against.
    """
    assert missing_render_artifacts("plugins/Stub.vst3", "preset.vstpreset") == (
        "plugins/Stub.vst3",
        "preset.vstpreset",
    )


@pytest.mark.parametrize(
    "backend_name", [TORCHSYNTH_PLUGIN_NAME, FAUST_PLUGIN_NAME, SURGEPY_PLUGIN_NAME]
)
def test_missing_render_artifacts_in_process_backend_name_is_not_a_path(
    backend_name: str, workspace: Path
) -> None:
    """An in-process backend sentinel names no bundle on disk, so it is never missing.

    :param backend_name: ``plugin_path`` sentinel selecting an in-process renderer.
    :param workspace: Empty CWD, proving the sentinel is not stat-ed.
    """
    assert missing_render_artifacts(backend_name, "") == ()


def test_missing_render_artifacts_undeclared_preset_is_not_required(workspace: Path) -> None:
    """An empty preset path means the backend takes no preset, not a missing file.

    :param workspace: CWD holding only the bundle.
    """
    (workspace / "plugins" / "Stub.vst3").mkdir(parents=True)

    assert missing_render_artifacts("plugins/Stub.vst3", "") == ()


def test_missing_render_artifacts_expands_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``~`` is expanded, matching the DawDreamer renderer's own resolution.

    :param tmp_path: Stands in as the home directory.
    :param monkeypatch: Points ``$HOME`` at it.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Stub.vst3").mkdir()

    assert missing_render_artifacts("~/Stub.vst3", "") == ()
