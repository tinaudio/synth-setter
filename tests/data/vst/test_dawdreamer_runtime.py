"""DawDreamer worker-runtime capability checks."""

from __future__ import annotations

import sys

import pytest

from synth_setter.data.vst.dawdreamer_runtime import ensure_dawdreamer_runtime


def test_pedalboard_backend_does_not_probe_dawdreamer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pedalboard workers do not require the optional DawDreamer package.

    :param monkeypatch: Replaces the import seam with an unexpected-call failure.
    """
    monkeypatch.setattr(
        "synth_setter.data.vst.dawdreamer_runtime.import_module",
        lambda _name: pytest.fail("DawDreamer import should not run"),
    )

    ensure_dawdreamer_runtime("pedalboard")


def test_dawdreamer_backend_supported_worker_imports_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported CPython worker with the package installed passes.

    :param monkeypatch: Pins a supported runtime and records the package import.
    """
    imported: list[str] = []
    monkeypatch.setattr(sys, "version_info", (3, 11, 9))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        "synth_setter.data.vst.dawdreamer_runtime.import_module",
        lambda name: imported.append(name),
    )

    ensure_dawdreamer_runtime("dawdreamer")

    assert imported == ["dawdreamer"]


def test_dawdreamer_backend_python_313_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python 3.13 workers fail with the supported interpreter range.

    :param monkeypatch: Pins the worker interpreter to unsupported Python 3.13.
    """
    monkeypatch.setattr(sys, "version_info", (3, 13, 1))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match=r"DawDreamer.*CPython 3\.11 or 3\.12.*3\.13"):
        ensure_dawdreamer_runtime("dawdreamer")


def test_dawdreamer_backend_linux_arm64_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux arm64 workers fail because the pinned release has no wheel.

    :param monkeypatch: Pins the worker to the unsupported Linux arm64 wheel target.
    """
    monkeypatch.setattr(sys, "version_info", (3, 12, 4))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")

    with pytest.raises(RuntimeError, match=r"DawDreamer.*Linux/aarch64.*Linux/x86_64"):
        ensure_dawdreamer_runtime("dawdreamer")


def test_dawdreamer_backend_missing_package_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported worker without DawDreamer reports how to install it.

    :param monkeypatch: Pins a supported worker and makes the package import fail.
    """
    monkeypatch.setattr(sys, "version_info", (3, 12, 4))
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    def _missing(_name: str) -> None:
        raise ModuleNotFoundError("No module named 'dawdreamer'")

    monkeypatch.setattr(
        "synth_setter.data.vst.dawdreamer_runtime.import_module",
        _missing,
    )

    with pytest.raises(RuntimeError, match=r"DawDreamer.*uv sync.*worker"):
        ensure_dawdreamer_runtime("dawdreamer")
