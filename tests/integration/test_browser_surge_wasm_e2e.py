"""Real trained music-sketch inference and unchanged Surge WASM audio in Chromium."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import sh
import soundfile as sf
import torch

from synth_setter.cli.clap_render import resolve_inverse_checkpoint
from synth_setter.cli.sketch_render import (
    _prepare_inputs,
    _resolve_stats,
    load_render_config,
)
from synth_setter.data.vst.param_spec import (
    decode_model_output,
    require_note_params,
    require_scalar_synth_params,
)
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.tools.export_browser_surge_bundle import _load_settings


@pytest.mark.slow
@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_surgepy
def test_browser_surge_wasm_real_checkpoint_renders_predicted_patch(tmp_path: Path) -> None:
    """A static page consumes real audio and renders its ONNX prediction with Surge WASM.

    :param tmp_path: Retained model, screenshot, waveform and browser run artifacts.
    """
    repo = Path(__file__).resolve().parents[2]
    settings = _load_settings()
    render = load_render_config()
    native = make_audio_renderer(render)
    sketch = native.render({}, midi_note=60, velocity=100, note_start_and_end=(0.0, 2.0))
    content = native.render({}, midi_note=72, velocity=80, note_start_and_end=(0.5, 1.5))
    sf.write(tmp_path / "sketch.wav", sketch.T, render.sample_rate, subtype="FLOAT")
    sf.write(tmp_path / "content.wav", content.T, render.sample_rate, subtype="FLOAT")
    model_dir = tmp_path / "model"
    sh.Command(sys.executable)(
        [
            "-m",
            "synth_setter.tools.export_browser_surge_bundle",
            "--checkpoint",
            settings.checkpoint,
            "--checkpoint-sha256",
            settings.checkpoint_sha256,
            "--stats",
            settings.stats,
            "--stats-sha256",
            settings.stats_sha256,
            "--output",
            str(model_dir),
        ],
        _cwd=repo,
        _timeout=600,
    )
    engine = os.environ.get("SYNTH_SETTER_SURGE_WASM_BUNDLE")
    if not engine:
        pytest.fail(
            "Build src/synth_setter/surgewasm/build.mjs and set SYNTH_SETTER_SURGE_WASM_BUNDLE"
        )
    sh.Command("node")(
        [
            "src/synth_setter/web/surge/export-site.mjs",
            "--model",
            str(model_dir),
            "--engine",
            engine,
            "--output",
            str(tmp_path / "site"),
        ],
        _cwd=repo,
        _timeout=60,
    )
    sh.Command("node")(
        [
            "src/synth_setter/web/surge/e2e.mjs",
            str(tmp_path / "site"),
            str(tmp_path / "content.wav"),
            str(tmp_path / "sketch.wav"),
            str(tmp_path / "record.json"),
        ],
        _cwd=repo,
        _timeout=600,
    )
    record = json.loads((tmp_path / "record.json").read_text())
    predicted = np.asarray(record["audio"], dtype=np.float32)
    assert predicted.shape == content.shape
    downloaded, downloaded_rate = sf.read(tmp_path / "record.json.wav", dtype="float32")
    assert downloaded_rate == render.sample_rate
    np.testing.assert_array_equal(downloaded.T, predicted)
    assert np.isfinite(predicted).all()
    assert float(np.abs(predicted).max()) > 1e-5
    assert np.isfinite(record["normalizedMelMae"])
    assert record["normalizedMelMae"] >= 0
    assert "1.4" in record["rendererVersion"]
    assert record["engineCommit"] == "dd68c74346c828ef25bd6504867936648161b7a7"
    model = VSTFlowMatchingModule.load_from_checkpoint(
        resolve_inverse_checkpoint(settings.checkpoint, settings.checkpoint_sha256),
        map_location="cpu",
        weights_only=False,
    ).eval()
    batch = _prepare_inputs(
        sketch_audio=sketch,
        content_audio=content,
        stats_path=_resolve_stats(settings.stats, settings.stats_sha256),
        model=model,
        render=render,
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(
        np.asarray(record["mel"]).reshape(batch["mel"].shape),
        batch["mel"].numpy(),
        atol=2e-3,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(record["sketch"]).reshape(batch["sketch_ctrl"].shape),
        batch["sketch_ctrl"].numpy(),
        atol=2e-3,
        rtol=1e-3,
    )
    with torch.no_grad():
        expected = model.sample_batch(
            batch,
            noise=torch.tensor([record["noise"]], dtype=torch.float32),
            content_cfg_strength=2.0,
            sketch_cfg_strength=3.0,
            sample_steps=8,
        ).numpy()[0]
    np.testing.assert_allclose(record["params"], expected, rtol=2e-3, atol=2e-3)
    synth, note = decode_model_output(
        np.asarray(record["params"]), param_specs[render.param_spec_name]
    )
    assert record["patch"]["synth"] == pytest.approx(synth)
    assert record["patch"]["note"]["pitch"] == note["pitch"]
    assert record["patch"]["note"]["note_start_and_end"] == pytest.approx(
        note["note_start_and_end"]
    )
    sf.write(tmp_path / "browser-pred.wav", predicted.T, render.sample_rate, subtype="FLOAT")
    native_note = require_note_params(note)
    note_start, note_end = sorted(native_note["note_start_and_end"])
    reference = native.render(
        require_scalar_synth_params(synth),
        midi_note=native_note["pitch"],
        velocity=100,
        note_start_and_end=(note_start, note_end),
    )
    sf.write(tmp_path / "native-reference.wav", reference.T, render.sample_rate, subtype="FLOAT")
    comparison = {
        "cross_version_rmse": float(np.sqrt(np.mean((predicted - reference) ** 2))),
        "browser_peak": float(np.abs(predicted).max()),
        "native_peak": float(np.abs(reference).max()),
    }
    (tmp_path / "comparison.json").write_text(json.dumps(comparison, indent=2))
