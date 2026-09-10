"""Failure-boundary tests for FaustWasm artifact compilation and publication."""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst import faustwasm_artifacts
from synth_setter.data.vst.faustwasm_artifacts import (
    compile_faustwasm_artifact,
    export_faustwasm_artifact,
    repository_faustwasm_script,
)
from synth_setter.synth_spec import SYNTHS, SynthName

_ROOT = Path(__file__).parents[3]
_NODE_MODULE = _ROOT / "node_modules/@grame/faustwasm/package.json"


def _configured_backend_version() -> str:
    """Return the FaustWasm version pinned by the render configuration.

    :returns: Configured backend version.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return str(compose(config_name="render/faustwasm").render.backend_version)


def _set_fake_checkout(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Relocate repository asset discovery beneath ``root``.

    :param monkeypatch: Replaces module-root discovery.
    :param root: Temporary checkout root.
    :returns: Expected script path in the relocated checkout.
    """
    module_path = root / "src/synth_setter/data/vst/faustwasm_artifacts.py"
    monkeypatch.setattr(faustwasm_artifacts, "__file__", str(module_path))
    return root / "scripts/faustwasm/export-artifacts.mjs"


def test_repository_script_missing_checkout_asset_reports_install_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed wheel without repository scripts reports the required checkout.

    :param tmp_path: Isolated checkout-shaped root.
    :param monkeypatch: Redirects module path discovery to the isolated root.
    """
    _set_fake_checkout(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="unavailable outside a synth-setter checkout"):
        repository_faustwasm_script("export-artifacts.mjs")


def test_repository_script_missing_node_package_reports_npm_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkout without installed FaustWasm dependencies reports ``npm ci``.

    :param tmp_path: Isolated checkout-shaped root.
    :param monkeypatch: Redirects module path discovery to the isolated root.
    """
    script = _set_fake_checkout(monkeypatch, tmp_path)
    script.parent.mkdir(parents=True)
    script.touch()

    with pytest.raises(RuntimeError, match="run `npm ci`"):
        repository_faustwasm_script("export-artifacts.mjs")


def test_repository_script_missing_node_executable_reports_path_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed assets without a Node executable fail before subprocess launch.

    :param tmp_path: Isolated checkout-shaped root.
    :param monkeypatch: Redirects module and executable discovery.
    """
    script = _set_fake_checkout(monkeypatch, tmp_path)
    script.parent.mkdir(parents=True)
    script.touch()
    package = tmp_path / "node_modules/@grame/faustwasm/package.json"
    package.parent.mkdir(parents=True)
    package.write_text("{}")
    monkeypatch.setattr(faustwasm_artifacts.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="requires Node.js on PATH"):
        repository_faustwasm_script("export-artifacts.mjs")


def test_compile_artifact_rejects_non_faust_identity_before_filesystem_write(
    tmp_path: Path,
) -> None:
    """Only registered checked-in Faust sources reach the compiler.

    :param tmp_path: Absent output directory used to detect premature writes.
    """
    output = tmp_path / "artifact"

    with pytest.raises(ValueError, match="registered Faust source"):
        compile_faustwasm_artifact(
            SYNTHS[SynthName("surge_xt")],
            output,
            backend_version=_configured_backend_version(),
        )

    assert not output.exists()


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("identity", "faust_bubble"),
        ("faustwasmVersion", "0.0.0"),
        ("sourceSha256", "0" * 64),
        ("mode", "poly"),
        ("voices", 1),
        ("outputs", 2),
        ("parameters", []),
    ],
)
def test_compile_artifact_rejects_manifest_provenance_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    """Every registry-owned manifest field is checked after compilation.

    :param tmp_path: Isolated source and candidate artifact directories.
    :param monkeypatch: Replaces only the compiler process with a drifted manifest producer.
    :param field: Registry-owned manifest field under test.
    :param replacement: Value that contradicts the compile request.
    """
    synth = SYNTHS[SynthName("faust_filter_osc")]
    baseline = tmp_path / "baseline"
    backend_version = _configured_backend_version()
    compile_faustwasm_artifact(synth, baseline, backend_version=backend_version)
    payload = json.loads((baseline / "manifest.json").read_text())
    payload[field] = replacement

    def _write_drifted_manifest(args: list[str], **_kwargs: Any) -> None:
        output = Path(args[-1])
        (output / "manifest.json").write_text(json.dumps(payload))

    monkeypatch.setattr(faustwasm_artifacts.subprocess, "run", _write_drifted_manifest)

    with pytest.raises(ValueError, match="provenance does not match"):
        compile_faustwasm_artifact(
            synth,
            tmp_path / "candidate",
            backend_version=backend_version,
        )


def test_export_cleanup_failure_does_not_mask_successful_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Best-effort staging cleanup cannot turn a completed publication into failure.

    :param tmp_path: Isolated publication destination.
    :param monkeypatch: Leaves staging in place and makes its cleanup fail.
    :param caplog: Captures the actionable cleanup warning.
    """
    manifest = object()

    def _compile(_synth: object, staging: Path, **_kwargs: str) -> object:
        (staging / "manifest.json").write_text("{}")
        return manifest

    def _fail_cleanup(_path: Path) -> None:
        raise OSError("busy")

    monkeypatch.setattr(faustwasm_artifacts, "compile_faustwasm_artifact", _compile)
    monkeypatch.setattr(faustwasm_artifacts, "_rename_without_replace", lambda *_args: None)
    monkeypatch.setattr(faustwasm_artifacts.shutil, "rmtree", _fail_cleanup)

    with caplog.at_level(logging.WARNING, logger=faustwasm_artifacts.__name__):
        result = export_faustwasm_artifact(
            SynthName("faust_filter_osc"),
            tmp_path / "published",
            backend_version=_configured_backend_version(),
        )

    assert result is manifest
    assert "failed to remove FaustWasm staging directory" in caplog.text


def test_rename_without_replace_windows_delegates_to_atomic_os_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows publication uses the platform's non-replacing directory rename.

    :param tmp_path: Supplies concrete source and destination paths.
    :param monkeypatch: Selects Windows and records the OS rename result.
    """
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(faustwasm_artifacts.os, "name", "nt")
    monkeypatch.setattr(os, "rename", lambda src, dst: calls.append((src, dst)))

    faustwasm_artifacts._rename_without_replace(source, destination)

    assert calls == [(source, destination)]


def test_rename_without_replace_darwin_uses_exclusive_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Darwin publication requests the exclusive native rename operation.

    :param tmp_path: Supplies concrete source and destination paths.
    :param monkeypatch: Selects Darwin and injects a successful libc boundary.
    """
    calls: list[tuple[bytes, bytes, int]] = []

    class _Libc:
        def renamex_np(self, source: bytes, destination: bytes, flag: int) -> int:
            """Record the exclusive rename invocation.

            :param source: Encoded source path.
            :param destination: Encoded destination path.
            :param flag: Native rename flag.
            :returns: Native success status.
            """
            calls.append((source, destination, flag))
            return 0

    monkeypatch.setattr(faustwasm_artifacts.sys, "platform", "darwin")
    monkeypatch.setattr(faustwasm_artifacts.ctypes, "CDLL", lambda *_args, **_kwargs: _Libc())
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    faustwasm_artifacts._rename_without_replace(source, destination)

    assert calls == [(os.fsencode(source), os.fsencode(destination), 4)]


@pytest.mark.parametrize(
    ("error_number", "error_type"),
    [(errno.EEXIST, FileExistsError), (errno.EACCES, OSError)],
)
def test_rename_without_replace_translates_native_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    error_type: type[OSError],
) -> None:
    """Native collision and OS errors retain their public exception classes.

    :param tmp_path: Supplies concrete source and destination paths.
    :param monkeypatch: Injects a deterministic native rename failure.
    :param error_number: Native errno returned by libc.
    :param error_type: Public exception expected from the errno.
    """
    class _Libc:
        def renameat2(
            self,
            old_dir_fd: int,
            old_path: bytes,
            new_dir_fd: int,
            new_path: bytes,
            flags: int,
        ) -> int:
            """Return the injected native failure.

            :param old_dir_fd: Source directory descriptor.
            :param old_path: Encoded source path.
            :param new_dir_fd: Destination directory descriptor.
            :param new_path: Encoded destination path.
            :param flags: Native rename flags.
            :returns: Native failure status.
            """
            ctypes.set_errno(error_number)
            return -1

    monkeypatch.setattr(faustwasm_artifacts.sys, "platform", "linux")
    monkeypatch.setattr(faustwasm_artifacts.ctypes, "CDLL", lambda *_args, **_kwargs: _Libc())
    destination = tmp_path / "destination"

    with pytest.raises(error_type) as exc_info:
        faustwasm_artifacts._rename_without_replace(tmp_path / "source", destination)

    assert exc_info.value.filename == destination
