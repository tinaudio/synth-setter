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
from synth_setter.cli.sketch_render import (
    _load_audio,
    _prepare_inputs,
    _resolve_stats,
    load_render_config,
)
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
    sketch_audio = make_audio_renderer(render).render(
        {}, midi_note=60, velocity=100, note_start_and_end=(0.0, 2.0)
    )
    content_audio = make_audio_renderer(render).render(
        {}, midi_note=72, velocity=80, note_start_and_end=(0.5, 1.5)
    )
    sketch_source = tmp_path / "sketch-input.wav"
    content_source = tmp_path / "content-input.wav"
    write_wav(sketch_audio, str(sketch_source), render.sample_rate, render.channels)
    write_wav(content_audio, str(content_source), render.sample_rate, render.channels)
    np.testing.assert_array_less(1e-4, np.max(np.abs(sketch_audio)))
    np.testing.assert_array_less(1e-4, np.max(np.abs(content_audio)))
    assert not np.array_equal(sketch_audio, content_audio)
    repo = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "evaluation"
    result = sh.Command("node")(
        [
            str(repo / "src/synth_setter/web/e2e.mjs"),
            str(tmp_path / "browser.png"),
            sys.executable,
            "-m",
            "synth_setter.cli.sketch_render",
            str(sketch_source),
            str(content_source),
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
            "--content-cfg",
            "2",
            "--sketch-cfg",
            "3",
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
    arm = output_dir / "arms/cfg-c2-s3"
    payload = json.loads((arm / "browser/input.json").read_text())
    prediction = json.loads((arm / "browser/prediction.json").read_text())
    actual = np.array(prediction["params"], dtype=np.float32)
    model = VSTFlowMatchingModule.load_from_checkpoint(
        resolve_inverse_checkpoint(_CHECKPOINT_URI, _CHECKPOINT_SHA),
        map_location="cpu",
        weights_only=False,
    ).eval()
    decoded_sketch = _load_audio(sketch_source, render)
    decoded_content = _load_audio(content_source, render)
    batch = _prepare_inputs(
        sketch_audio=decoded_sketch,
        content_audio=decoded_content,
        stats_path=_resolve_stats(_STATS_URI, _STATS_SHA),
        model=model,
        render=render,
        device=torch.device("cpu"),
    )
    for key in ("mel", "sketch_ctrl"):
        transported = torch.tensor(payload[key]["data"], dtype=torch.float32).reshape(
            payload[key]["shape"]
        )
        torch.testing.assert_close(transported, batch[key], rtol=1e-5, atol=1e-5)
    noise = torch.randn(
        (1, model.hparams["num_params"]),
        generator=torch.Generator(device="cpu").manual_seed(17),
        dtype=torch.float32,
        device="cpu",
    )
    torch.testing.assert_close(
        torch.tensor([payload["noise"]], dtype=torch.float32), noise, rtol=0, atol=0
    )
    expected = model.sample_batch(
        batch,
        noise=noise,
        content_cfg_strength=2.0,
        sketch_cfg_strength=3.0,
        sample_steps=8,
    ).numpy()[0]
    assert actual.shape == (model.hparams["num_params"],)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    predicted_audio, sample_rate = sf.read(arm / "pred.wav", dtype="float32")
    assert sample_rate == render.sample_rate
    assert predicted_audio.shape == content_audio.T.shape
    assert np.isfinite(predicted_audio).all()
    assert float(np.abs(predicted_audio).max()) > 1e-4
    for name, original in (("sketch.wav", decoded_sketch), ("target.wav", decoded_content)):
        saved, _ = sf.read(arm / name, dtype="float32")
        # The production WAV writer requantizes decoded samples to PCM16.
        np.testing.assert_allclose(saved, original.T, rtol=0, atol=1 / 2**15)
    provenance = json.loads((arm / "browser/provenance.json").read_text())
    revision = str(sh.Command("git")(["rev-parse", "HEAD"], _cwd=repo)).strip()
    assert provenance["git_revision"].removesuffix("-dirty") == revision
    metrics = pd.read_csv(arm / "metrics.csv")
    assert len(metrics) == 1
    assert np.isfinite(
        metrics[["mss", "wmfcc", "sot", "rms", "mldr", "mldr_mid_side"]].to_numpy()
    ).all()
