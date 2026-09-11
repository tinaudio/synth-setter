"""Generate golden fixtures for the browser pyFDN JavaScript ports.

The JavaScript sketch extractor, parameter decoder, and impulse-response metrics under
``synth_setter/web/fdn`` must reproduce the Python references. Both sides rebuild the same
synthetic impulse responses from a tiny integer recipe, so the fixtures carry only expected
outputs and the SOS filter tables that scipy designs.

Typical usage::

    uv run python -m synth_setter.tools.browser_fdn_fixtures
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from pyFDN import octave_band_filterbank, octave_bands

from synth_setter.data.vst.param_spec import decode_model_output
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.evaluation.acoustic_parameters import (
    _BAND_COUNT,
    _BAND_START_OCTAVE,
)
from synth_setter.evaluation.compute_audio_metrics import (
    compute_acoustic_parameter_metrics_mono_only,
    compute_mss,
    compute_octave_edc_rmse_db_mono_only,
    compute_octave_rt60_log_rmse_mono_only,
)
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.param_spec_name import ParamSpecName

SAMPLE_RATE = 44_100
FIXTURE_DIR = Path(str(files("synth_setter") / "web" / "fdn" / "fixtures"))
_PARAM_SPEC_NAME = ParamSpecName("pyfdn_n8_mono_householder")
_LCG_MULTIPLIER = 1_664_525
_LCG_INCREMENT = 1_013_904_223
_LCG_MODULUS = 2**32
_TARGET_RECIPE = {"seed": 7, "samples": 4 * SAMPLE_RATE, "decay_seconds": 0.6}
_PRED_RECIPE = {"seed": 11, "samples": 4 * SAMPLE_RATE, "decay_seconds": 0.9}


def synthetic_impulse_response(*, seed: int, samples: int, decay_seconds: float) -> np.ndarray:
    """Build a unit-peak exponentially decaying noise burst from a 32-bit LCG.

    The recipe is intentionally trivial so JavaScript reproduces it bit-for-bit: the
    generator is a Numerical Recipes LCG, uniforms are ``state / 2**32``, and the envelope
    is ``exp(-t / decay_seconds)`` in float64 with a unit direct-path spike at sample 0.

    :param seed: Initial LCG state.
    :param samples: Response length in samples.
    :param decay_seconds: Exponential envelope time constant in seconds.
    :returns: Float64 response shaped ``(samples,)`` with peak amplitude one.
    """
    state = seed % _LCG_MODULUS
    uniforms = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        state = (_LCG_MULTIPLIER * state + _LCG_INCREMENT) % _LCG_MODULUS
        uniforms[index] = state / _LCG_MODULUS
    time = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    response = (2.0 * uniforms - 1.0) * 0.5 * np.exp(-time / decay_seconds)
    response[0] = 1.0
    return response


def _sos_table(start: float, count: int) -> tuple[list[list[list[float]]], list[float]]:
    """Design pyFDN's Butterworth octave bank at the fixed sample rate.

    :param start: Octave offset of the lowest band relative to 1 kHz.
    :param count: Number of bands.
    :returns: Per-band SOS rows and centre frequencies in Hz.
    """
    bands, centres = octave_bands(start=start, n=count, fs=SAMPLE_RATE)
    sos = [np.asarray(section).tolist() for section in octave_band_filterbank(bands, SAMPLE_RATE)]
    return sos, centres.tolist()


def _decode_fixture() -> dict[str, Any]:
    """Decode one deterministic model row through the householder spec.

    :returns: Model row and its renderer-native values.
    """
    spec = resolve_param_spec(_PARAM_SPEC_NAME)
    model_row = np.linspace(-1.2, 1.1, spec.encoded_width, dtype=np.float32)
    synth_params, _note = decode_model_output(model_row, spec)
    return {
        "param_spec_name": _PARAM_SPEC_NAME,
        "model_row": model_row.tolist(),
        "delays": np.asarray(synth_params["delays"]).tolist(),
        "input_matrix": np.asarray(synth_params["input_matrix"]).ravel().tolist(),
        "output_matrix": np.asarray(synth_params["output_matrix"]).ravel().tolist(),
        "direct_matrix": float(np.asarray(synth_params["direct_matrix"]).item()),
        "rt_dc_seconds": float(np.asarray(synth_params["post_delay.rt_dc_seconds"]).item()),
        "rt_nyquist_seconds": float(
            np.asarray(synth_params["post_delay.rt_nyquist_seconds"]).item()
        ),
    }


def build_fixtures() -> dict[str, dict[str, Any]]:
    """Evaluate every Python reference on the synthetic responses.

    :returns: JSON-ready payloads keyed by fixture file stem.
    """
    target = synthetic_impulse_response(**_TARGET_RECIPE)
    pred = synthetic_impulse_response(**_PRED_RECIPE)
    target_pair = target[None, :]
    pred_pair = pred[None, :]
    acoustic = compute_acoustic_parameter_metrics_mono_only(target_pair, pred_pair, SAMPLE_RATE)
    sketch_sos, sketch_centres = _sos_table(start=-4.0, count=8)
    metric_sos, metric_centres = _sos_table(start=_BAND_START_OCTAVE, count=_BAND_COUNT)
    return {
        "golden": {
            "sample_rate": SAMPLE_RATE,
            "target": {
                "recipe": _TARGET_RECIPE,
                "sketch": extract_reverb_sketch(target, SAMPLE_RATE).tolist(),
            },
            "pred": {
                "recipe": _PRED_RECIPE,
                "sketch": extract_reverb_sketch(pred, SAMPLE_RATE).tolist(),
            },
            "metrics": {
                "mss": compute_mss(target_pair, pred_pair, SAMPLE_RATE),
                "octave_edc_rmse_db": compute_octave_edc_rmse_db_mono_only(
                    target_pair, pred_pair, SAMPLE_RATE
                ),
                "octave_rt60_log_rmse": compute_octave_rt60_log_rmse_mono_only(
                    target_pair, pred_pair, SAMPLE_RATE
                ),
                "t30_mape": acoustic["t30_mape"],
                "c50_mae_db": acoustic["c50_mae_db"],
            },
            "decode": _decode_fixture(),
        },
        "octave_bands": {
            "sample_rate": SAMPLE_RATE,
            "sketch": sketch_sos,
            "sketch_centres_hz": sketch_centres,
            "metrics": metric_sos,
            "metrics_centres_hz": metric_centres,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Write the fixture JSON files.

    :param argv: Optional command-line arguments excluding the executable name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in build_fixtures().items():
        (args.output / f"{name}.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
