"""Production-path tests for persistent FaustWasm artifact export."""

from __future__ import annotations

import json
import shutil
import subprocess
from importlib.resources import as_file
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst import faustwasm_artifacts
from synth_setter.data.vst.faustwasm_artifacts import (
    compile_faustwasm_artifact,
    run_faustwasm_render_worker,
)
from synth_setter.resources import faustwasm_dir
from synth_setter.synth_spec import SYNTHS, SynthName
from synth_setter.tools.export_faustwasm import main

_NODE_UNAVAILABLE = shutil.which("node") is None


def _configured_backend_version() -> str:
    """Return the authored FaustWasm version from Hydra's render contract.

    :returns: Configured backend version.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return str(compose(config_name="render/faustwasm").render.backend_version)


def test_faustwasm_resources_include_pinned_runtime() -> None:
    """Packaged Node assets include the configured compiler and render workers."""
    with as_file(faustwasm_dir()) as resource_directory:
        package = json.loads((resource_directory / "vendor/package.json").read_text())
        assert package["version"] == _configured_backend_version()
        assert (resource_directory / "export-artifacts.mjs").is_file()
        assert (resource_directory / "render-worker.mjs").is_file()
        assert (resource_directory / "runtime.mjs").is_file()
        assert (resource_directory / "package-version.mjs").is_file()
        compiler_directory = resource_directory / "vendor/libfaust-wasm"
        assert tuple(compiler_directory.glob("libfaust-wasm.data.binpart*"))
        assert (compiler_directory / "libfaust-wasm.js").is_file()
        assert tuple(compiler_directory.glob("libfaust-wasm.wasm.binpart*"))


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("identity", "expected_mode", "expected_outputs"),
    [
        ("faust_bright_organ", "poly", 2),
        ("faust_filter_osc", "mono", 1),
        ("faust_syrinx_bird", "mono", 1),
        ("faust_syrinx2_bird", "mono", 1),
        ("faust_tract3_bird", "mono", 1),
    ],
)
def test_export_cli_persists_hashed_artifact_consumed_by_real_runtime(
    tmp_path: Path,
    identity: str,
    expected_mode: str,
    expected_outputs: int,
) -> None:
    """The public entrypoint's files load and render without recompilation.

    :param tmp_path: Isolated persistent artifact and render destination.
    :param identity: Registered source selected through the public entrypoint.
    :param expected_mode: Native mode compiled from source metadata.
    :param expected_outputs: Native channel count compiled from source metadata.
    """
    output_dir = tmp_path / identity
    main(["--synth", identity, "--output", str(output_dir)])
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["identity"] == identity
    assert manifest["faustwasmVersion"] == _configured_backend_version()
    assert manifest["mode"] == expected_mode
    assert manifest["outputs"] == expected_outputs
    assert len(manifest["files"]["dsp"]["sha256"]) == 64

    patch = {
        parameter["canonicalAddress"]: (
            parameter["values"][0]
            if parameter["kind"] == "discrete"
            else (parameter["min"] + parameter["max"]) / 2.0
        )
        for parameter in manifest["parameters"]
    }
    request_path = tmp_path / f"{identity}-render.json"
    audio_path = tmp_path / f"{identity}.f32"
    request_path.write_text(
        json.dumps(
            {
                "expectedFaustWasmVersion": _configured_backend_version(),
                "sampleRate": 44_100,
                "blockSize": 128,
                "frames": 513,
                "note": 60,
                "velocity": 100,
                "startFrame": 0,
                "endFrame": 256,
                "params": patch,
            }
        )
    )
    run_faustwasm_render_worker(output_dir, request_path, audio_path)
    audio = np.fromfile(audio_path, dtype="<f4").reshape(expected_outputs, 513)

    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 1e-6


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_real_runtime_rejects_configured_backend_version_mismatch(tmp_path: Path) -> None:
    """The render worker checks its request against installed package metadata.

    :param tmp_path: Isolated persistent artifact and render destination.
    """
    output_dir = tmp_path / "artifact"
    main(["--synth", "faust_filter_osc", "--output", str(output_dir)])
    manifest = json.loads((output_dir / "manifest.json").read_text())
    request_path = tmp_path / "render.json"
    request_path.write_text(
        json.dumps(
            {
                "expectedFaustWasmVersion": "999.0.0",
                "sampleRate": 44_100,
                "blockSize": 128,
                "frames": 128,
                "note": 60,
                "velocity": 100,
                "startFrame": 0,
                "endFrame": 64,
                "params": {
                    parameter["canonicalAddress"]: (
                        parameter["values"][0]
                        if parameter["kind"] == "discrete"
                        else (parameter["min"] + parameter["max"]) / 2.0
                    )
                    for parameter in manifest["parameters"]
                },
            }
        )
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_faustwasm_render_worker(output_dir, request_path, tmp_path / "audio.f32")

    assert "FaustWasm version mismatch" in exc_info.value.stderr


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_real_runtime_rejects_persisted_manifest_version_mismatch(tmp_path: Path) -> None:
    """The Node consumer compares persisted provenance with its installed package.

    :param tmp_path: Isolated persistent artifact and render destination.
    """
    output_dir = tmp_path / "artifact"
    main(["--synth", "faust_filter_osc", "--output", str(output_dir)])
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    patch = {
        parameter["canonicalAddress"]: (
            parameter["values"][0]
            if parameter["kind"] == "discrete"
            else (parameter["min"] + parameter["max"]) / 2.0
        )
        for parameter in manifest["parameters"]
    }
    manifest["faustwasmVersion"] = "999.0.0"
    manifest_path.write_text(json.dumps(manifest))
    request_path = tmp_path / "render.json"
    request_path.write_text(
        json.dumps(
            {
                "expectedFaustWasmVersion": _configured_backend_version(),
                "sampleRate": 44_100,
                "blockSize": 128,
                "frames": 128,
                "note": 60,
                "velocity": 100,
                "startFrame": 0,
                "endFrame": 64,
                "params": patch,
            }
        )
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_faustwasm_render_worker(output_dir, request_path, tmp_path / "audio.f32")

    assert "artifact 999.0.0, installed" in exc_info.value.stderr


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_real_runtime_rejects_corrupted_persisted_module(tmp_path: Path) -> None:
    """The Node consumer rejects module bytes that drift from the manifest digest.

    :param tmp_path: Isolated persistent artifact and render destination.
    """
    output_dir = tmp_path / "artifact"
    main(["--synth", "faust_filter_osc", "--output", str(output_dir)])
    manifest = json.loads((output_dir / "manifest.json").read_text())
    module_path = output_dir / manifest["files"]["dsp"]["path"]
    module_path.write_bytes(module_path.read_bytes() + b"corrupt")
    request_path = tmp_path / "render.json"
    request_path.write_text(
        json.dumps(
            {
                "expectedFaustWasmVersion": _configured_backend_version(),
                "sampleRate": 44_100,
                "blockSize": 128,
                "frames": 128,
                "note": 60,
                "velocity": 100,
                "startFrame": 0,
                "endFrame": 64,
                "params": {
                    parameter["canonicalAddress"]: (
                        parameter["values"][0]
                        if parameter["kind"] == "discrete"
                        else (parameter["min"] + parameter["max"]) / 2.0
                    )
                    for parameter in manifest["parameters"]
                },
            }
        )
    )

    with pytest.raises(subprocess.CalledProcessError, match="render-worker.mjs") as exc_info:
        run_faustwasm_render_worker(output_dir, request_path, tmp_path / "audio.f32")

    assert "artifact digest mismatch" in exc_info.value.stderr


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
def test_compile_rejects_backend_version_that_differs_from_installed_package(
    tmp_path: Path,
) -> None:
    """The Node compiler compares Hydra-owned provenance with package metadata.

    :param tmp_path: Isolated compilation destination.
    """
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        compile_faustwasm_artifact(
            SYNTHS[SynthName("faust_bright_organ")],
            tmp_path,
            backend_version="999.0.0",
        )

    assert "FaustWasm version mismatch" in exc_info.value.stderr


@pytest.mark.skipif(_NODE_UNAVAILABLE, reason="Node.js is unavailable")
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("identity", "faust_filter_osc"),
        ("faustwasmVersion", "999.0.0"),
        ("sourceSha256", "0" * 64),
        ("mode", "mono"),
        ("voices", 999),
        ("outputs", 999),
        ("parameters", []),
    ],
)
def test_compile_rejects_each_corrupted_manifest_provenance_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
) -> None:
    """The public compiler rejects every registry-owned manifest field.

    :param tmp_path: Isolated compilation destination.
    :param monkeypatch: Corrupts the staged manifest after the real Node compile.
    :param field: Registry-owned manifest field under test.
    :param invalid_value: Value that contradicts the compile request.
    """
    real_run = subprocess.run

    def run_then_corrupt(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = real_run(*args, **kwargs)
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest[field] = invalid_value
        manifest_path.write_text(json.dumps(manifest))
        return result

    monkeypatch.setattr(faustwasm_artifacts.subprocess, "run", run_then_corrupt)

    with pytest.raises(ValueError, match="artifact provenance does not match"):
        compile_faustwasm_artifact(
            SYNTHS[SynthName("faust_bright_organ")],
            tmp_path,
            backend_version=_configured_backend_version(),
        )


def test_export_cli_existing_output_fails_without_modifying_destination(tmp_path: Path) -> None:
    """An existing destination remains untouched when publication is refused.

    :param tmp_path: Isolated destination containing caller-owned state.
    """
    output_dir = tmp_path / "artifact"
    output_dir.mkdir()
    marker = output_dir / "owned.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="output destination already exists"):
        main(["--synth", "faust_filter_osc", "--output", str(output_dir)])

    assert marker.read_text() == "keep"
    assert not (output_dir / "manifest.json").exists()
