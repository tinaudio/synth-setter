"""Pin :func:`synth_setter.workspace.operator_workspace` resolution order.

Covers env override, checkout detection, CWD fallback, and the
``PROJECT_ROOT`` side effect. See #1261 for the import-time crash these
tests pin against.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from synth_setter import workspace


@pytest.fixture(autouse=True)
def _reset_workspace_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the ``@cache`` and any leaked ``$PROJECT_ROOT`` between cases."""
    workspace.operator_workspace.cache_clear()
    monkeypatch.delenv("PROJECT_ROOT", raising=False)


def _stage_synthetic_package(tmp_path: Path) -> Path:
    """Copy ``src/synth_setter`` into ``tmp_path/site-packages`` for subprocess use."""
    src_pkg = Path(__file__).resolve().parents[1] / "src" / "synth_setter"
    dest = tmp_path / "site-packages"
    dest.mkdir()
    shutil.copytree(src_pkg, dest / "synth_setter", symlinks=True)
    for parent in (dest / "synth_setter").parents:
        assert not (parent / ".project-root").is_file(), (
            f"unexpected .project-root under {parent} — test setup leaked"
        )
    return dest


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``$SYNTH_SETTER_WORKSPACE`` takes precedence over the checkout."""
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    resolved = workspace.operator_workspace()
    assert resolved == tmp_path.resolve()
    assert resolved.is_absolute()


def test_checkout_root_detected_from_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env override, the parents-walk lands on the test file's checkout root."""
    monkeypatch.delenv("SYNTH_SETTER_WORKSPACE", raising=False)
    expected = Path(workspace.__file__).resolve().parents[2]
    resolved = workspace.operator_workspace()
    assert resolved == expected
    assert (resolved / ".project-root").is_file()


def test_project_root_env_set_as_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The helper publishes the resolved path as ``$PROJECT_ROOT`` for Hydra interpolation."""
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    workspace.operator_workspace()
    assert os.environ["PROJECT_ROOT"] == str(tmp_path.resolve())


def test_project_root_env_not_overwritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pre-set ``$PROJECT_ROOT`` is preserved — operator wins."""
    monkeypatch.setenv("PROJECT_ROOT", "/already/set")
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    workspace.operator_workspace()
    assert os.environ["PROJECT_ROOT"] == "/already/set"


def test_cwd_fallback_when_no_checkout_reachable(tmp_path: Path) -> None:
    """Spawn a subprocess from a layout with no ``.project-root`` reachable.

    Mirrors the wheel-install crash repro in #1261: copy ``synth_setter``
    into a clean tree, point ``PYTHONPATH`` at it, leave both
    workspace-resolution envs unset so the cwd branch fires.
    """
    dest = _stage_synthetic_package(tmp_path)

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
            "HOME": str(tmp_path),
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
    """Each console-script module imports cleanly and publishes ``$PROJECT_ROOT``.

    Failure here means #1261's import-time crash regressed: a launcher
    walked up from ``__file__`` looking for ``.project-root`` and raised
    ``FileNotFoundError`` before publishing its ``main`` callable. No
    workspace env is set so the cwd branch fires — the exact wheel-install
    failure mode the issue described. The post-import ``$PROJECT_ROOT``
    assertion confirms the side effect ``configs/paths/default.yaml``'s
    ``${oc.env:PROJECT_ROOT}`` interpolation depends on actually fires.
    """
    dest = _stage_synthetic_package(tmp_path)

    proc = subprocess.run(  # noqa: S603 — controlled argv
        [
            sys.executable,
            "-s",
            "-c",
            (f"import os; import {module}; print(os.environ.get('PROJECT_ROOT', '<unset>'))"),
        ],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(dest),
            "PYTHONNOUSERSITE": "1",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path.resolve()), (
        f"launcher import did not publish $PROJECT_ROOT (got {proc.stdout.strip()!r})"
    )
