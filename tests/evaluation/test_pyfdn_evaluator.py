"""Behavior tests for native pyFDN prediction evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from pyFDN import FDNBuild

import synth_setter.evaluation.pyfdn_evaluator as pyfdn_evaluator
from synth_setter.data.pyfdn_instrument import PyFDNRenderer, params_to_fdn_build
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.vst.param_spec import ParameterValues
from synth_setter.evaluation.pyfdn_evaluator import (
    PyFDNEvaluation,
    decode_pyfdn_model_output,
    evaluate_pyfdn_row,
)


def test_decode_pyfdn_model_output_round_trips_all_91_coordinates() -> None:
    """An in-domain model row decodes without clipping or field loss."""
    expected, expected_notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(17))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(expected, expected_notes)
    model_row = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)

    decoded = decode_pyfdn_model_output(model_row)

    assert decoded.keys() == expected.keys()
    np.testing.assert_array_equal(decoded["delays"], expected["delays"])
    for name in (
        "feedback_matrix",
        "input_matrix",
        "output_matrix",
        "direct_matrix",
    ):
        np.testing.assert_allclose(decoded[name], expected[name], atol=1.2e-7)
    assert decoded["post_delay.rt_dc_seconds"] == pytest.approx(
        expected["post_delay.rt_dc_seconds"], abs=1.2e-7
    )
    assert decoded["post_delay.rt_nyquist_seconds"] == pytest.approx(
        expected["post_delay.rt_nyquist_seconds"], abs=1.2e-7
    )


@pytest.mark.parametrize(
    "row",
    [
        np.zeros(90, dtype=np.float32),
        np.zeros(91, dtype=np.float64),
        np.full(91, np.nan, dtype=np.float32),
        np.full(91, 1.01, dtype=np.float32),
    ],
)
def test_decode_pyfdn_model_output_invalid_row_raises(row: np.ndarray) -> None:
    """Malformed or out-of-domain predictions fail instead of being clipped.

    :param row: Invalid prediction row.
    """
    with pytest.raises((TypeError, ValueError)):
        decode_pyfdn_model_output(row)


def test_evaluate_pyfdn_row_exact_target_prediction_has_identity_metrics(
    source_file: tuple[Path, str],
) -> None:
    """One exact prediction traverses real build, filters, render, and 48 kHz metrics.

    :param source_file: Checksum-pinned production-geometry source audio.
    """
    path, checksum = source_file
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(23))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    model_row = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)

    result = evaluate_pyfdn_row(
        model_row,
        model_row,
        renderer=PyFDNRenderer(path, checksum),
    )

    assert result.status == "finite_render"
    assert result.predicted_audio is not None
    np.testing.assert_array_equal(result.predicted_audio, result.target_audio)
    assert result.audio_metrics == pytest.approx(
        {"mss": 0.0, "wmfcc": 0.0, "sot": 0.0, "rms_cosine": 1.0},
        abs=1e-5,
    )


def test_evaluate_pyfdn_row_builder_rejection_counts_build_invalid(
    source_file: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decoded patch rejected by the real build boundary is classified separately.

    :param source_file: Checksum-pinned production-geometry source audio.
    :param monkeypatch: Scoped build-boundary rejection injection.
    """
    path, checksum = source_file
    target_params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(27))
    target_encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(target_params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(target_encoded).astype(np.float32)
    prediction = np.zeros(91, dtype=np.float32)
    real_build = params_to_fdn_build

    def reject_zero_feedback(params: ParameterValues, *, sample_rate: float) -> FDNBuild:
        """Reject the finite patch chosen for the build-invalid case.

        :param params: Decoded target or predicted patch.
        :param sample_rate: Fixed renderer sample rate.
        :returns: Real target build.
        :raises ValueError: The prediction carries zero feedback.
        """
        if not np.asarray(params["feedback_matrix"]).any():
            raise ValueError("rejected exact prediction")
        return real_build(params, sample_rate=sample_rate)

    monkeypatch.setattr(pyfdn_evaluator, "params_to_fdn_build", reject_zero_feedback)

    result = evaluate_pyfdn_row(
        prediction,
        target,
        renderer=PyFDNRenderer(path, checksum),
    )

    assert result.status == "build_invalid"
    assert result.error is not None and "rejected exact prediction" in result.error


def test_evaluate_pyfdn_row_unstable_exact_prediction_counts_render_invalid(
    source_file: tuple[Path, str],
) -> None:
    """A finite exact build that overflows is classified at the render boundary.

    :param source_file: Checksum-pinned production-geometry source audio.
    """
    path, checksum = source_file
    target_params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(29))
    target_encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(target_params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(target_encoded).astype(np.float32)
    unstable = np.ones(91, dtype=np.float32)

    with np.errstate(over="ignore", invalid="ignore"):
        result = evaluate_pyfdn_row(
            unstable,
            target,
            renderer=PyFDNRenderer(path, checksum),
        )

    assert result.status == "render_invalid"
    assert result.predicted_audio is None
    assert result.error is not None and "finite" in result.error


def test_pyfdn_evaluation_accounts_rows_and_writes_native_artifacts(
    source_file: tuple[Path, str], tmp_path: Path
) -> None:
    """Successful and invalid predictions produce complete counts and inspectable artifacts.

    :param source_file: Checksum-pinned production-geometry source audio.
    :param tmp_path: Isolated evaluation output directory.
    """
    path, checksum = source_file
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(31))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)
    predictions = np.stack(
        [target, np.full(91, np.nan, dtype=np.float32), np.ones(91, dtype=np.float32)]
    )
    targets = np.stack([target, target, target])
    evaluator = PyFDNEvaluation(PyFDNRenderer(path, checksum), tmp_path)

    with np.errstate(over="ignore", invalid="ignore"):
        evaluator.evaluate_batch(predictions, targets)
    metrics = evaluator.finalize()

    assert metrics["pyfdn/rows_total"] == 3.0
    assert metrics["pyfdn/invalid/decode_count"] == 1.0
    assert metrics["pyfdn/invalid/build_count"] == 0.0
    assert metrics["pyfdn/invalid/render_count"] == 1.0
    assert metrics["pyfdn/valid_build_count"] == 2.0
    assert metrics["pyfdn/finite_render_count"] == 1.0
    assert metrics["pyfdn/valid_build_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["pyfdn/finite_render_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["pyfdn/parameter_finite_count"] == 2.0
    assert "pyfdn/parameter_mse/coordinate/delays.0" in metrics
    assert "pyfdn/parameter_mse/field/post_delay_rt_controls" in metrics

    rows = pd.read_csv(tmp_path / "metrics" / "metrics.csv")
    assert rows["status"].tolist() == ["finite_render", "decode_invalid", "render_invalid"]
    assert (tmp_path / "metrics" / "aggregated_metrics.csv").is_file()
    assert (tmp_path / "metrics" / "parameter_metrics.csv").is_file()
    pred_path = tmp_path / "audio" / "sample_0" / "pred.wav"
    target_path = tmp_path / "audio" / "sample_0" / "target.wav"
    assert sf.info(pred_path).subtype == "FLOAT"
    assert sf.info(target_path).subtype == "FLOAT"
    assert not list(tmp_path.rglob("*preview*"))
