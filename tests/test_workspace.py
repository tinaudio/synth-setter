"""Pin :func:`synth_setter.workspace.operator_workspace` resolution order.

The helper is import-time critical for every console-script launcher
(#1261): wheel installs have no ``.project-root`` reachable from
``__file__`` and used to crash at import. The three branches below cover
the resolution order it advertises — env override, checkout detection,
CWD fallback — plus the ``PROJECT_ROOT`` side effect that
``configs/paths/default.yaml`` interpolates against.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from synth_setter import workspace


@pytest.fixture(autouse=True)
def _reset_workspace_cache() -> None:
    """Drop the ``@cache`` between cases so each test resolves fresh."""
    workspace.operator_workspace.cache_clear()


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``$SYNTH_SETTER_WORKSPACE`` takes precedence over the checkout."""
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    resolved = workspace.operator_workspace()
    assert resolved == tmp_path.resolve()


def test_checkout_root_detected_from_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env override, the checkout's ``.project-root`` resolves the workspace."""
    monkeypatch.delenv("SYNTH_SETTER_WORKSPACE", raising=False)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    resolved = workspace.operator_workspace()
    assert (resolved / ".project-root").is_file(), (
        f"expected {resolved}/.project-root to exist under a normal test invocation"
    )


def test_project_root_env_set_as_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The helper publishes the resolved path as ``$PROJECT_ROOT`` for Hydra interpolation."""
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    workspace.operator_workspace()
    import os  # local import — only this test cares about the env side effect

    assert os.environ["PROJECT_ROOT"] == str(tmp_path.resolve())


def test_project_root_env_not_overwritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pre-set ``$PROJECT_ROOT`` is preserved — operator wins."""
    monkeypatch.setenv("PROJECT_ROOT", "/already/set")
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    workspace.operator_workspace()
    import os

    assert os.environ["PROJECT_ROOT"] == "/already/set"


def test_cwd_fallback_when_no_checkout_reachable(tmp_path: Path) -> None:
    """Spawn a subprocess from a layout with no ``.project-root`` reachable.

    Mirrors the wheel-install crash repro in #1261: copy ``synth_setter``
    into a clean tree, point ``PYTHONPATH`` at it, ensure ``$PWD`` /
    ``Path.cwd()`` becomes the fallback.
    """
    import shutil

    src_pkg = Path(__file__).resolve().parents[1] / "src" / "synth_setter"
    dest = tmp_path / "site-packages"
    dest.mkdir()
    shutil.copytree(src_pkg, dest / "synth_setter", symlinks=True)
    # Confirm the synthetic layout has no marker reachable.
    for parent in (dest / "synth_setter").parents:
        assert not (parent / ".project-root").is_file(), (
            f"unexpected .project-root under {parent} — test setup leaked"
        )

    proc = subprocess.run(  # noqa: S603 — invoking python with controlled argv
        [
            sys.executable,
            "-s",
            "-c",
            ("from synth_setter.workspace import operator_workspace; print(operator_workspace())"),
        ],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(dest),
            "PYTHONNOUSERSITE": "1",
            "PATH": "/usr/bin:/bin",
            # Strip both overrides so the cwd branch fires.
            "SYNTH_SETTER_WORKSPACE": "",
            "PROJECT_ROOT": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path.resolve())


@pytest.mark.parametrize(
    "module",
    [
        "synth_setter.cli.train",
        "synth_setter.cli.eval",
        "synth_setter.cli.generate_dataset",
        "synth_setter.cli.finalize_dataset",
    ],
)
def test_launcher_imports_without_project_root(tmp_path: Path, module: str) -> None:
    """Each console-script module imports cleanly from a layout with no marker.

    Failure here means #1261's import-time crash regressed: a launcher
    walked up from ``__file__`` looking for ``.project-root`` and raised
    ``FileNotFoundError`` before publishing its ``main`` callable.
    """
    import shutil

    src_pkg = Path(__file__).resolve().parents[1] / "src" / "synth_setter"
    dest = tmp_path / "site-packages"
    dest.mkdir()
    shutil.copytree(src_pkg, dest / "synth_setter", symlinks=True)

    proc = subprocess.run(  # noqa: S603 — controlled argv
        [sys.executable, "-s", "-c", f"import {module}"],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(dest),
            "PYTHONNOUSERSITE": "1",
            "PATH": "/usr/bin:/bin",
            "SYNTH_SETTER_WORKSPACE": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
