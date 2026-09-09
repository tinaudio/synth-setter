"""Real SurgePy audio through trained ONNX graphs, Chromium, and native evaluation."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sh
import soundfile as sf
import torch

from synth_setter.cli.clap_render import resolve_inverse_checkpoint
from synth_setter.cli.sketch_render import load_render_config
from synth_setter.data.vst.core import write_wav
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.renderer_factory import make_audio_renderer

_CHECKPOINT_SHA = "87a0fcb20f9efbcb56fffc6e3ee6e1c8743c47013c297e33516b472543468c07"
_STATS_SHA = "c0c45d75a8b77004b3802c761bc77b5b34e7709a08343b2cf70fee04b7f52a19"
_ARTIFACT_ROOT = "r2://experiments/browser-evaluation"
_CHECKPOINT_URI = f"{_ARTIFACT_ROOT}/{_CHECKPOINT_SHA}/checkpoint.ckpt"
_STATS_URI = f"{_ARTIFACT_ROOT}/{_STATS_SHA}/stats.npz"
_WEB_ROOT = Path(__file__).resolve().parents[2] / "src/synth_setter/web"


@pytest.mark.slow
@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_surgepy
@pytest.mark.skipif(
    not (_WEB_ROOT / "node_modules/@playwright/test").is_dir(),
    reason="Run npm ci --prefix src/synth_setter/web and install its Playwright Chromium browser",
)
def test_browser_flow_real_checkpoint_renders_audio_and_finite_metrics(tmp_path: Path) -> None:
    """The shipped browser's predictions reach the real SurgePy and metric consumers.

    :param tmp_path: Isolated input, ONNX, screenshot, audio, and metric artifacts.
    """
    render = load_render_config()
    audio = make_audio_renderer(render).render(
        {}, midi_note=60, velocity=100, note_start_and_end=(0.0, 2.0)
    )
    source = tmp_path / "input.wav"
    write_wav(audio, str(source), render.sample_rate, render.channels)
    np.testing.assert_array_less(1e-4, np.max(np.abs(audio)))
    repo = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "evaluation"
    result = sh.Command("node")(
        [
            str(repo / "src/synth_setter/web/e2e.mjs"),
            str(tmp_path / "browser.png"),
            sys.executable,
            "-m",
            "synth_setter.cli.sketch_render",
            str(source),
            str(source),
            "--checkpoint",
            _CHECKPOINT_URI,
            "--checkpoint-sha256",
            _CHECKPOINT_SHA,
            "--stats",
            _STATS_URI,
            "--stats-sha256",
            _STATS_SHA,
            "--inference-runtime",
            "browser",
            "--sample-steps",
            "8",
            "--seed",
            "17",
            "--output-dir",
            str(output_dir),
            "--no-upload",
        ],
        _cwd=repo,
        _env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        _timeout=420,
        _err_to_out=True,
    )
    assert isinstance(result, str)
    (tmp_path / "browser-cli.log").write_text(result)
    assert "BROWSER_E2E_COMPLETE" in result
    arm = output_dir / "arms/cfg-c2-s2"
    payload = json.loads((arm / "browser/input.json").read_text())
    prediction = json.loads((arm / "browser/prediction.json").read_text())
    actual = np.array(prediction["params"], dtype=np.float32)
    model = VSTFlowMatchingModule.load_from_checkpoint(
        resolve_inverse_checkpoint(_CHECKPOINT_URI, _CHECKPOINT_SHA),
        map_location="cpu",
        weights_only=False,
    ).eval()
    batch = {
        key: torch.tensor(payload[key]["data"], dtype=torch.float32).reshape(payload[key]["shape"])
        for key in ("mel", "sketch_ctrl")
    }
    expected = model.sample_batch(
        batch,
        noise=torch.tensor([payload["noise"]], dtype=torch.float32),
        content_cfg_strength=2.0,
        sketch_cfg_strength=2.0,
        sample_steps=8,
    ).numpy()[0]
    assert actual.shape == (model.hparams["num_params"],)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    predicted_audio, sample_rate = sf.read(arm / "pred.wav", dtype="float32")
    assert sample_rate == render.sample_rate
    assert predicted_audio.shape == audio.T.shape
    assert np.isfinite(predicted_audio).all()
    assert float(np.abs(predicted_audio).max()) > 1e-4
    metrics = pd.read_csv(arm / "metrics.csv")
    assert len(metrics) == 1
    assert np.isfinite(
        metrics[["mss", "wmfcc", "sot", "rms", "mldr", "mldr_mid_side"]].to_numpy()
    ).all()
