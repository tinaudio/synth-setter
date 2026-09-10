"""Production-path tests for persistent FaustWasm artifact export."""

from __future__ import annotations

import json
import subprocess
from importlib.resources import as_file
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.faustwasm_artifacts import run_faustwasm_render_worker
from synth_setter.resources import faustwasm_dir
from synth_setter.tools.export_faustwasm import main

_ROOT = Path(__file__).parents[1]
_NODE_MODULE = _ROOT / "node_modules/@grame/faustwasm/package.json"


def test_faustwasm_resources_include_pinned_runtime() -> None:
    """Packaged Node assets include the pinned compiler and render workers."""
    with as_file(faustwasm_dir()) as resource_dir:
        package = json.loads((resource_dir / "vendor/package.json").read_text())

        assert package["version"] == "0.18.3"
        assert (resource_dir / "export-artifacts.mjs").is_file()
        assert (resource_dir / "render-worker.mjs").is_file()
        assert (resource_dir / "runtime.mjs").is_file()
        assert (resource_dir / "vendor/faustwasm.mjs").is_file()
        compiler_dir = resource_dir / "vendor/libfaust-wasm"
        assert tuple(compiler_dir.glob("libfaust-wasm.data.binpart*"))
        assert (compiler_dir / "libfaust-wasm.js").is_file()
        assert tuple(compiler_dir.glob("libfaust-wasm.wasm.binpart*"))


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
                "sampleRate": 44_100,
                "blockSize": 128,
                "frames": 512,
                "note": 60,
                "velocity": 100,
                "startFrame": 0,
                "endFrame": 256,
                "params": patch,
            }
        )
    )
    run_faustwasm_render_worker(output_dir, request_path, audio_path)
    audio = np.fromfile(audio_path, dtype="<f4").reshape(expected_outputs, 512)

    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 1e-6


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
