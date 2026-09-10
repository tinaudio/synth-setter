"""Production-path tests for persistent FaustWasm artifact export."""

from __future__ import annotations

import json
import subprocess
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
from synth_setter.synth_spec import SYNTHS, SynthName
from synth_setter.tools.export_faustwasm import main

_ROOT = Path(__file__).parents[1]
_NODE_MODULE = _ROOT / "node_modules/@grame/faustwasm/package.json"


def _configured_backend_version() -> str:
    """Return the authored FaustWasm version from Hydra's render contract.

    :returns: Configured backend version.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return str(compose(config_name="render/faustwasm").render.backend_version)


def _relocate_artifact_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Relocate module-root discovery to a real temporary checkout layout.

    :param monkeypatch: Replaces the module's runtime file location.
    :param tmp_path: Temporary checkout root.
    :returns: Temporary checkout root containing the relocated module.
    """
    relocated = tmp_path / "src/synth_setter/data/vst/faustwasm_artifacts.py"
    relocated.parent.mkdir(parents=True)
    relocated.touch()
    monkeypatch.setattr(faustwasm_artifacts, "__file__", str(relocated))
    return tmp_path


def test_runtime_resolver_missing_script_raises_checkout_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relocated module without scripts reports the checkout requirement.

    :param monkeypatch: Relocates module-root discovery.
    :param tmp_path: Real temporary checkout layout.
    """
    _relocate_artifact_module(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="Node assets are unavailable"):
        faustwasm_artifacts.repository_faustwasm_script("render-worker.mjs")


def test_runtime_resolver_missing_package_raises_install_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A checkout script without its npm package reports the install command.

    :param monkeypatch: Relocates module-root discovery.
    :param tmp_path: Real temporary checkout layout.
    """
    root = _relocate_artifact_module(monkeypatch, tmp_path)
    script = root / "scripts/faustwasm/render-worker.mjs"
    script.parent.mkdir(parents=True)
    script.touch()

    with pytest.raises(RuntimeError, match=r"not installed; run `npm ci`"):
        faustwasm_artifacts.repository_faustwasm_script("render-worker.mjs")


def test_runtime_resolver_empty_path_raises_node_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A complete temporary layout still requires Node on the real PATH lookup.

    :param monkeypatch: Relocates module-root discovery and empties PATH.
    :param tmp_path: Real temporary checkout layout.
    """
    root = _relocate_artifact_module(monkeypatch, tmp_path)
    script = root / "scripts/faustwasm/render-worker.mjs"
    script.parent.mkdir(parents=True)
    script.touch()
    package = root / "node_modules/@grame/faustwasm/package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"version": "0.18.3"}')
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RuntimeError, match="requires Node.js on PATH"):
        faustwasm_artifacts.repository_faustwasm_script("render-worker.mjs")


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
@pytest.mark.parametrize(
    ("identity", "expected_mode", "expected_outputs"),
    [
        ("faust_bright_organ", "poly", 2),
        ("faust_filter_osc", "mono", 1),
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


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
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


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
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


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
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


@pytest.mark.skipif(not _NODE_MODULE.is_file(), reason="run `npm ci` to install @grame/faustwasm")
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
