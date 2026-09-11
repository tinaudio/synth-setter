"""Real pyFDN response through the exported browser site: inference, Faust render, and metrics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import sh
import torch
from pydantic import BaseModel, ConfigDict

from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.vst.core import write_wav
from synth_setter.data.vst.param_spec import decode_model_output
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.evaluation.compute_audio_metrics import (
    compute_acoustic_parameter_metrics_mono_only,
    compute_mss,
    compute_octave_edc_rmse_db_mono_only,
    compute_octave_rt60_log_rmse_mono_only,
)
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.param_spec_name import ParamSpecName


class _BrowserRun(BaseModel):
    """The page's ``window.fdnEval`` record, validated at the browser-to-Python boundary.

    .. attribute :: model_config

        Strict validation; unknown keys are rejected.

    .. attribute :: state

        Run outcome; only ``complete`` records are scored.

        :type: str

    .. attribute :: mode

        Conditioning mode the page ran.

        :type: str

    .. attribute :: contentCfg

        Content guidance strength.

        :type: float

    .. attribute :: sketchCfg

        Sketch guidance strength.

        :type: float

    .. attribute :: steps

        RK4 integration steps.

        :type: int

    .. attribute :: seed

        Noise seed.

        :type: int

    .. attribute :: weights

        Branch weights over the four velocity branches.

        :type: list[float]

    .. attribute :: noise

        Initial flow state in model space.

        :type: list[float]

    .. attribute :: params

        Sampled model-space parameters.

        :type: list[float]

    .. attribute :: patch

        Canonical native parameter values.

        :type: dict[str, float]

    .. attribute :: metrics

        Impulse-response metrics the page computed.

        :type: dict[str, float]

    .. attribute :: sketch

        Flattened ``(10, 32)`` reverb sketch.

        :type: list[float]

    .. attribute :: target

        Decoded target samples.

        :type: list[float]

    .. attribute :: pred

        FaustWasm render of the prediction.

        :type: list[float]
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    state: str
    mode: str
    contentCfg: float
    sketchCfg: float
    steps: int
    seed: int
    weights: list[float]
    noise: list[float]
    params: list[float]
    patch: dict[str, float]
    metrics: dict[str, float]
    sketch: list[float]
    target: list[float]
    pred: list[float]


_REPO = Path(__file__).resolve().parents[2]
_WEB_ROOT = _REPO / "src/synth_setter/web"
_MODEL_BUNDLE = os.environ.get("BROWSER_FDN_MODEL_BUNDLE")
_FAUST_ARTIFACT = os.environ.get("BROWSER_FDN_FAUST_ARTIFACT")
_CHECKPOINT = os.environ.get("SYNTH_SETTER_FDN_SKETCH_CHECKPOINT")
_STATS = os.environ.get("SYNTH_SETTER_FDN_SKETCH_STATS")
_SPEC_NAME = ParamSpecName("pyfdn_n8_mono_householder")
_SAMPLE_RATE = 44_100
_STEPS = 8
_SEED = 17


def _python_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Score one mono pair with the five metrics the page reports.

    :param target: Mono target response.
    :param pred: Mono predicted response of the same length.
    :returns: Metric values keyed like the page's metrics table.
    """
    acoustic = compute_acoustic_parameter_metrics_mono_only(target[None], pred[None], _SAMPLE_RATE)
    return {
        "mss": float(compute_mss(target[None], pred[None], _SAMPLE_RATE)),
        "octave_edc_rmse_db": compute_octave_edc_rmse_db_mono_only(
            target[None], pred[None], _SAMPLE_RATE
        ),
        "octave_rt60_log_rmse": compute_octave_rt60_log_rmse_mono_only(
            target[None], pred[None], _SAMPLE_RATE
        ),
        "t30_mape": acoustic["t30_mape"],
        "c50_mae_db": acoustic["c50_mae_db"],
    }


@pytest.mark.slow
@pytest.mark.skipif(
    not (_MODEL_BUNDLE and _FAUST_ARTIFACT),
    reason="Set BROWSER_FDN_MODEL_BUNDLE and BROWSER_FDN_FAUST_ARTIFACT to exported directories",
)
@pytest.mark.skipif(
    not (_WEB_ROOT / "node_modules/@playwright/test").is_dir(),
    reason="Run npm ci --prefix src/synth_setter/web and install its Playwright Chromium browser",
)
def test_browser_fdn_site_matches_python_inference_render_and_metrics(tmp_path: Path) -> None:
    """The exported site reproduces Python sampling, pyFDN rendering, and metric scoring.

    :param tmp_path: Isolated site export, target WAV, run record, and screenshot.
    """
    spec = resolve_param_spec(_SPEC_NAME)
    renderer = PyFDNRenderer(param_spec_name=_SPEC_NAME)
    params, _note = spec.sample(np.random.default_rng(5))
    target = renderer.render(params, note_start_and_end=(0.0, 4.0))
    target_wav = tmp_path / "target.wav"
    write_wav(target, str(target_wav), _SAMPLE_RATE, 1)
    target_mono = np.asarray(target, dtype=np.float64).reshape(-1)

    site = tmp_path / "site"
    node = sh.Command("node")
    node(
        str(_WEB_ROOT / "fdn/export-site.mjs"),
        "--model",
        _MODEL_BUNDLE,
        "--faust",
        _FAUST_ARTIFACT,
        "--output",
        str(site),
    )
    record_path = tmp_path / "run.json"
    output = node(
        str(_WEB_ROOT / "fdn/e2e.mjs"),
        str(site),
        str(target_wav),
        str(record_path),
        "both",
        "2",
        "3",
        str(_STEPS),
        str(_SEED),
    )
    assert "BROWSER_FDN_E2E_COMPLETE" in str(output)
    record = _BrowserRun.model_validate_json(record_path.read_text(encoding="utf-8"))
    assert record.state == "complete"

    # The browser decoded the 16-bit WAV, so its target differs from the float render by ≤ 1 LSB.
    browser_target = np.asarray(record.target, dtype=np.float64)
    assert browser_target.shape == target_mono.shape
    assert np.abs(browser_target - target_mono).max() <= 1.5 / 32767

    # Sketch parity against the Python extractor on the exact samples the page saw.
    expected_sketch = extract_reverb_sketch(browser_target, _SAMPLE_RATE)
    np.testing.assert_allclose(
        np.asarray(record.sketch, dtype=np.float32).reshape(10, 32), expected_sketch, atol=1e-5
    )

    # Metric parity on the exact pair the page scored.
    browser_pred = np.asarray(record.pred, dtype=np.float64)
    assert browser_pred.shape == target_mono.shape
    assert np.isfinite(browser_pred).all() and np.abs(browser_pred).max() > 0
    for name, value in _python_metrics(browser_target, browser_pred).items():
        assert record.metrics[name] == pytest.approx(value, rel=1e-4), name

    # Render parity: pyFDN on the browser's decoded parameters vs the FaustWasm render.
    synth_params, _ = decode_model_output(np.asarray(record.params, dtype=np.float32), spec)
    pyfdn_pred = np.asarray(
        renderer.render(synth_params, note_start_and_end=(0.0, 4.0)), dtype=np.float64
    ).reshape(-1)
    scale = np.abs(pyfdn_pred).max()
    assert np.abs(pyfdn_pred - browser_pred).max() <= 1e-3 * scale

    _assert_onnx_replay_parity(Path(str(_MODEL_BUNDLE)), record, browser_target, expected_sketch)
    if _CHECKPOINT and _STATS:
        _assert_sampling_parity(_CHECKPOINT, _STATS, record, browser_target, expected_sketch)


def _assert_onnx_replay_parity(
    bundle: Path, record: _BrowserRun, target: np.ndarray, sketch: np.ndarray
) -> None:
    """Replay the browser's noise and weights through the same graphs on onnxruntime.

    :param bundle: Exported model bundle directory.
    :param record: The page's run record with noise, branch weights, and parameters.
    :param target: Mono target the page decoded.
    :param sketch: Reverb sketch of that target.
    """
    import onnxruntime as ort

    frontend, conditioning, velocity = (
        ort.InferenceSession(str(bundle / name), providers=["CPUExecutionProvider"])
        for name in ("frontend.onnx", "conditioning.onnx", "velocity.onnx")
    )
    mel = frontend.run(None, {"waveform": target.astype(np.float32)[None]})[0]
    content, controls, null_controls = conditioning.run(
        None, {"mel": mel, "sketch_ctrl": sketch[None]}
    )
    weights = np.asarray(record.weights, dtype=np.float32)

    def field(x: np.ndarray, t: float) -> np.ndarray:
        output = velocity.run(
            None,
            {
                "x": x[None],
                "t": np.asarray([[t]], dtype=np.float32),
                "conditioning": content,
                "controls": controls,
                "null_controls": null_controls,
                "branch_weights": weights,
            },
        )[0]
        return np.asarray(output, dtype=np.float32)[0]

    x = np.asarray(record.noise, dtype=np.float32)
    dt = np.float32(1.0 / _STEPS)
    t = np.float32(0.0)
    for _ in range(_STEPS):
        k1 = field(x, t)
        k2 = field(x + dt / 2 * k1, t + dt / 2)
        k3 = field(x + dt / 2 * k2, t + dt / 2)
        k4 = field(x + dt * k3, t + dt)
        x = (x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).astype(np.float32)
        t = np.float32(t + dt)
    np.testing.assert_allclose(
        np.asarray(record.params, dtype=np.float32), x, rtol=2e-4, atol=2e-4
    )


def _assert_sampling_parity(
    checkpoint: str, stats: str, record: _BrowserRun, target: np.ndarray, sketch: np.ndarray
) -> None:
    """Replay the browser's noise through the PyTorch sampler and compare parameters.

    :param checkpoint: Local checkpoint path the bundle was exported from.
    :param stats: Local training mel-statistics archive the bundle was exported with.
    :param record: The page's run record with noise, guidance, and parameters.
    :param target: Mono target the page decoded.
    :param sketch: Reverb sketch of that target.
    """
    from synth_setter.models.mel_frontend import NormalizedMelFrontend
    from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

    model = VSTFlowMatchingModule.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    ).eval()
    with np.load(stats) as archive:
        frontend = NormalizedMelFrontend(
            sample_rate=_SAMPLE_RATE, mean=archive["mean"], std=archive["std"]
        ).eval()
    with torch.no_grad():
        batch = {
            "mel": frontend(torch.from_numpy(target.astype(np.float32))[None]),
            "sketch_ctrl": torch.from_numpy(sketch)[None],
        }
        expected = model.sample_batch(
            batch,
            noise=torch.tensor(record.noise, dtype=torch.float32)[None],
            sample_steps=_STEPS,
            content_cfg_strength=record.contentCfg,
            sketch_cfg_strength=record.sketchCfg,
        )
    np.testing.assert_allclose(
        np.asarray(record.params, dtype=np.float32), expected[0].numpy(), rtol=2e-4, atol=2e-4
    )
