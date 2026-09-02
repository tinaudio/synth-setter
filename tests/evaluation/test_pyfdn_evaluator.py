"""Behavior tests for native pyFDN prediction evaluation."""

import importlib
import sys
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
    PyFDNRowEvaluation,
    decode_pyfdn_model_output,
    evaluate_pyfdn_row,
)

type _EvaluatedOutputs = tuple[Path, dict[str, float], np.ndarray, np.ndarray, PyFDNRowEvaluation]


def test_pyfdn_evaluator_imports_without_registry_order_priming() -> None:
    """Hydra can resolve the evaluator target before the VST registry is imported."""
    prefixes = (
        "synth_setter.data.pyfdn",
        "synth_setter.data.vst",
        "synth_setter.evaluation.pyfdn_evaluator",
    )
    saved = {name: module for name, module in sys.modules.items() if name.startswith(prefixes)}
    for name in saved:
        del sys.modules[name]
    try:
        imported = importlib.import_module("synth_setter.evaluation.pyfdn_evaluator")
    finally:
        for name in tuple(sys.modules):
            if name.startswith(prefixes):
                del sys.modules[name]
        sys.modules.update(saved)
        for name, module in saved.items():
            parent_name, _, attribute = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, attribute, module)

    assert imported.PYFDN_N8_MONO_PARAM_SPEC.encoded_width == 91


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
        np.full(91, -1.01, dtype=np.float32),
    ],
)
def test_decode_pyfdn_model_output_invalid_row_raises(row: np.ndarray) -> None:
    """Malformed or out-of-domain predictions fail instead of being clipped.

    :param row: Invalid prediction row.
    """
    with pytest.raises((TypeError, ValueError)):
        decode_pyfdn_model_output(row)


def test_distribution_metrics_use_population_mean_and_standard_deviation() -> None:
    """Two known observations produce exact population statistics."""
    assert pyfdn_evaluator._distribution_metrics([1.0, 3.0]) == {
        "mean": 2.0,
        "std": 1.0,
    }


def test_parameter_metrics_use_finite_row_and_field_coordinate_denominators() -> None:
    """Parameter summaries use finite rows, coordinate means, and exact field spans."""
    squared_error = np.zeros(91, dtype=np.float64)
    squared_error[8] = 6.0
    squared_error[88] = 8.0
    squared_error[89] = 2.0
    squared_error[90] = 4.0

    metrics, rows = pyfdn_evaluator._parameter_metrics(squared_error, finite_count=2)

    assert metrics["pyfdn/parameter_mse"] == pytest.approx(10.0 / 91.0)
    assert metrics["pyfdn/parameter_mse/coordinate/direct_matrix.0.0"] == 4.0
    assert metrics["pyfdn/parameter_mse/field/feedback_matrix"] == pytest.approx(3.0 / 64.0)
    assert metrics["pyfdn/parameter_mse/field/direct_matrix"] == 4.0
    assert metrics["pyfdn/parameter_mse/field/post_delay_rt_controls"] == 1.5
    assert len(rows) == 91


def test_evaluate_pyfdn_row_exact_target_prediction_has_identity_metrics() -> None:
    """One exact prediction traverses real build, filters, render, and 48 kHz metrics."""
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(23))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    model_row = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)

    result = evaluate_pyfdn_row(
        model_row,
        model_row,
        renderer=PyFDNRenderer(),
    )

    assert result.status == "finite_render"
    assert result.predicted_audio is not None
    np.testing.assert_array_equal(result.predicted_audio, result.target_audio)
    assert result.audio_metrics == pytest.approx(
        {"mss": 0.0, "wmfcc": 0.0, "sot": 0.0, "rms_cosine": 1.0},
        abs=1e-5,
    )


def test_evaluate_pyfdn_row_builder_rejection_counts_build_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decoded patch rejected by the real build boundary is classified separately.

    :param monkeypatch: Scoped build-boundary rejection injection.
    """
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
        renderer=PyFDNRenderer(),
    )

    assert result.status == "build_invalid"
    assert result.error is not None and "rejected exact prediction" in result.error


def test_evaluate_pyfdn_row_unstable_exact_prediction_counts_render_invalid() -> None:
    """A finite exact build that overflows is classified at the render boundary."""
    target_params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(29))
    target_encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(target_params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(target_encoded).astype(np.float32)
    unstable = np.ones(91, dtype=np.float32)

    with np.errstate(over="ignore", invalid="ignore"):
        result = evaluate_pyfdn_row(
            unstable,
            target,
            renderer=PyFDNRenderer(),
        )

    assert result.status == "render_invalid"
    assert result.predicted_audio is None
    assert result.error is not None and "finite" in result.error


def test_pyfdn_evaluation_resumes_committed_rows_after_interruption(tmp_path: Path) -> None:
    """A restarted evaluator skips matching committed rows and continues aggregation.

    :param tmp_path: Isolated evaluation output directory.
    """
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(30))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)
    interrupted = PyFDNEvaluation(PyFDNRenderer(), tmp_path)
    interrupted.evaluate_batch(target[None, :], target[None, :])
    pending = tmp_path / "metrics" / "rows" / "sample_1.pending"
    pending.write_text("interrupted")
    orphan = tmp_path / "audio" / "sample_1"
    orphan.mkdir()
    (orphan / "pred.wav").write_bytes(b"partial")

    resumed = PyFDNEvaluation(PyFDNRenderer(), tmp_path)
    predictions = np.stack([target, np.full(91, np.nan, dtype=np.float32)])
    resumed.evaluate_batch(predictions, np.stack([target, target]))
    metrics = resumed.finalize()

    assert metrics["pyfdn/rows_total"] == 2.0
    assert metrics["pyfdn/finite_render_count"] == 1.0
    assert metrics["pyfdn/invalid/decode_count"] == 1.0
    assert (tmp_path / "audio" / "sample_0" / "pred.wav").is_file()
    assert not orphan.exists()
    assert not pending.exists()


def test_pyfdn_evaluation_resume_corrupt_artifact_raises(tmp_path: Path) -> None:
    """Committed progress never skips a native artifact whose bytes changed.

    :param tmp_path: Isolated evaluation output directory.
    """
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(30))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)
    PyFDNEvaluation(PyFDNRenderer(), tmp_path).evaluate_batch(target[None, :], target[None, :])
    (tmp_path / "audio" / "sample_0" / "pred.wav").write_bytes(b"corrupt")

    resumed = PyFDNEvaluation(PyFDNRenderer(), tmp_path)
    with pytest.raises(ValueError, match="artifacts do not match their digests"):
        resumed.evaluate_batch(target[None, :], target[None, :])


def test_pyfdn_evaluation_resume_missing_artifact_raises_value_error(
    tmp_path: Path,
) -> None:
    """Missing committed audio reports a progress contract failure.

    :param tmp_path: Isolated evaluation output directory.
    """
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(30))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)
    PyFDNEvaluation(PyFDNRenderer(), tmp_path).evaluate_batch(target[None, :], target[None, :])
    (tmp_path / "audio" / "sample_0" / "pred.wav").unlink()

    resumed = PyFDNEvaluation(PyFDNRenderer(), tmp_path)
    with pytest.raises(ValueError, match="artifacts do not match their digests"):
        resumed.evaluate_batch(target[None, :], target[None, :])


def test_pyfdn_evaluation_resume_changed_prediction_raises(tmp_path: Path) -> None:
    """Progress from another prediction stream fails instead of mixing artifacts.

    :param tmp_path: Isolated evaluation output directory.
    """
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(30))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)
    PyFDNEvaluation(PyFDNRenderer(), tmp_path).evaluate_batch(target[None, :], target[None, :])
    changed = target.copy()
    changed[0] = np.float32(0.123456)

    resumed = PyFDNEvaluation(PyFDNRenderer(), tmp_path)
    with pytest.raises(ValueError, match="does not match committed sample_0"):
        resumed.evaluate_batch(changed[None, :], target[None, :])


def test_pyfdn_evaluation_existing_audio_artifacts_raise_before_evaluation(
    tmp_path: Path,
) -> None:
    """A reused output directory cannot mix stale audio with current row metrics.

    :param tmp_path: Isolated evaluation output directory.
    """
    stale_audio = tmp_path / "audio" / "sample_0"
    stale_audio.mkdir(parents=True)
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(30))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)
    evaluator = PyFDNEvaluation(PyFDNRenderer(), tmp_path)

    with pytest.raises(ValueError, match="already contains pyFDN evaluation artifacts"):
        evaluator.evaluate_batch(target[None, :], target[None, :])


def test_pyfdn_evaluation_infinite_prediction_is_not_counted_as_finite_out_of_range(
    tmp_path: Path,
) -> None:
    """Infinite coordinates stay decode-invalid without inflating finite range counts.

    :param tmp_path: Isolated evaluation output directory.
    """
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(31))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)
    prediction = target.copy()
    prediction[0] = np.inf
    evaluator = PyFDNEvaluation(PyFDNRenderer(), tmp_path)

    evaluator.evaluate_batch(prediction[None, :], target[None, :])
    evaluator.finalize()

    rows = pd.read_csv(tmp_path / "metrics" / "metrics.csv")
    assert rows.loc[0, "pred_out_of_range_count"] == 0


@pytest.fixture(scope="module")
def evaluated_pyfdn_outputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> _EvaluatedOutputs:
    """Run one mixed-status evaluation for focused artifact assertions.

    :param tmp_path_factory: Factory for the module-scoped output directory.
    :returns: Output root, metrics, target, successful prediction, and native result.
    """
    output_dir = tmp_path_factory.mktemp("evaluated-pyfdn-outputs")
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(31))
    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    target = PYFDN_N8_MONO_PARAM_SPEC.encoded_to_model(encoded).astype(np.float32)
    successful = target.copy()
    successful[88] *= 0.5
    predictions = np.stack(
        [successful, np.full(91, np.nan, dtype=np.float32), np.ones(91, dtype=np.float32)]
    )
    evaluator = PyFDNEvaluation(PyFDNRenderer(), output_dir)
    with np.errstate(over="ignore", invalid="ignore"):
        evaluator.evaluate_batch(predictions, np.stack([target, target, target]))
    metrics = evaluator.finalize()
    expected = evaluate_pyfdn_row(successful, target, renderer=PyFDNRenderer())
    return output_dir, metrics, target, successful, expected


def test_pyfdn_evaluation_accounts_for_each_terminal_status(
    evaluated_pyfdn_outputs: _EvaluatedOutputs,
) -> None:
    """Mixed predictions produce distinct decode, render, and success counts.

    :param evaluated_pyfdn_outputs: Completed mixed-status evaluation.
    """
    output_dir, metrics, _, _, _ = evaluated_pyfdn_outputs

    assert metrics["pyfdn/rows_total"] == 3.0
    assert metrics["pyfdn/invalid/decode_count"] == 1.0
    assert metrics["pyfdn/invalid/build_count"] == 0.0
    assert metrics["pyfdn/invalid/render_count"] == 1.0
    assert metrics["pyfdn/valid_build_count"] == 2.0
    assert metrics["pyfdn/finite_render_count"] == 1.0
    assert metrics["pyfdn/valid_build_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["pyfdn/finite_render_rate"] == pytest.approx(1.0 / 3.0)
    rows = pd.read_csv(output_dir / "metrics" / "metrics.csv")
    assert rows["status"].tolist() == ["finite_render", "decode_invalid", "render_invalid"]


def test_pyfdn_evaluation_writes_metric_csv_schemas(
    evaluated_pyfdn_outputs: _EvaluatedOutputs,
) -> None:
    """Aggregate and coordinate metric CSVs retain their consumer schemas.

    :param evaluated_pyfdn_outputs: Completed mixed-status evaluation.
    """
    output_dir, _, _, _, _ = evaluated_pyfdn_outputs
    rows = pd.read_csv(output_dir / "metrics" / "metrics.csv")
    aggregated = pd.read_csv(output_dir / "metrics" / "aggregated_metrics.csv", index_col=0)
    parameter_metrics = pd.read_csv(output_dir / "metrics" / "parameter_metrics.csv")

    assert rows.columns.tolist() == [
        "sample_id",
        "status",
        "error",
        "pred_min",
        "pred_max",
        "pred_out_of_range_count",
        "target_peak",
        "target_rms",
        "pred_peak",
        "pred_rms",
        "mss",
        "rms_cosine",
        "sot",
        "wmfcc",
    ]
    assert aggregated.columns.tolist() == ["mean", "std"]
    assert all(pd.api.types.is_float_dtype(aggregated[column]) for column in aggregated)
    assert parameter_metrics.columns.tolist() == ["coordinate", "mse"]
    assert pd.api.types.is_float_dtype(parameter_metrics["mse"])


def test_pyfdn_evaluation_writes_exact_native_wavs(
    evaluated_pyfdn_outputs: _EvaluatedOutputs,
) -> None:
    """Native float WAVs preserve the exact evaluated waveforms.

    :param evaluated_pyfdn_outputs: Completed mixed-status evaluation.
    """
    output_dir, _, _, _, expected = evaluated_pyfdn_outputs
    pred_path = output_dir / "audio" / "sample_0" / "pred.wav"
    target_path = output_dir / "audio" / "sample_0" / "target.wav"
    pred_audio, _ = sf.read(pred_path, dtype="float32", always_2d=True)
    target_audio, _ = sf.read(target_path, dtype="float32", always_2d=True)

    assert sf.info(pred_path).subtype == "FLOAT"
    assert sf.info(target_path).subtype == "FLOAT"
    assert expected.predicted_audio is not None
    np.testing.assert_array_equal(pred_audio.T, expected.predicted_audio)
    np.testing.assert_array_equal(target_audio.T, expected.target_audio)


def test_pyfdn_evaluation_writes_exact_model_parameter_rows(
    evaluated_pyfdn_outputs: _EvaluatedOutputs,
) -> None:
    """The parameter artifact preserves exact prediction and target coordinates.

    :param evaluated_pyfdn_outputs: Completed mixed-status evaluation.
    """
    output_dir, _, target, successful, expected = evaluated_pyfdn_outputs
    params_csv = pd.read_csv(output_dir / "audio" / "sample_0" / "params.csv")

    np.testing.assert_allclose(
        params_csv["pred_model"].to_numpy(), successful, rtol=0.0, atol=3e-8
    )
    np.testing.assert_allclose(params_csv["target_model"].to_numpy(), target, rtol=0.0, atol=3e-8)
    np.testing.assert_allclose(
        params_csv["pred_encoded"].to_numpy(),
        (successful.astype(np.float64) + 1.0) / 2.0,
        rtol=0.0,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        params_csv["target_encoded"].to_numpy(),
        (target.astype(np.float64) + 1.0) / 2.0,
        rtol=0.0,
        atol=3e-8,
    )
    assert expected.predicted_params is not None
    np.testing.assert_allclose(
        params_csv["pred_native"].to_numpy(),
        pyfdn_evaluator._native_coordinate_values(expected.predicted_params),
        rtol=0.0,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        params_csv["target_native"].to_numpy(),
        pyfdn_evaluator._native_coordinate_values(expected.target_params),
        rtol=0.0,
        atol=3e-8,
    )


def test_pyfdn_evaluation_single_pair_has_zero_population_std(
    evaluated_pyfdn_outputs: _EvaluatedOutputs,
) -> None:
    """One successful audio pair has zero population standard deviation.

    :param evaluated_pyfdn_outputs: Completed mixed-status evaluation.
    """
    _, metrics, _, _, _ = evaluated_pyfdn_outputs

    assert metrics["pyfdn/audio/mss_std"] == 0.0


def test_pyfdn_evaluation_emits_no_normalized_preview(
    evaluated_pyfdn_outputs: _EvaluatedOutputs,
) -> None:
    """Native metric artifacts are not replaced by normalized previews.

    :param evaluated_pyfdn_outputs: Completed mixed-status evaluation.
    """
    output_dir, _, _, _, _ = evaluated_pyfdn_outputs

    assert not list(output_dir.rglob("*preview*"))
