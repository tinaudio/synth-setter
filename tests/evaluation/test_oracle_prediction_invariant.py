"""Exact parameter checks for predict-mode oracle evaluation."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from synth_setter.cli.eval import (
    _assert_exact_oracle_predictions,
    _run_predict_postprocessing,
)


def _write_pair(
    directory: Path,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    index: int = 0,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(prediction, directory / f"pred-{index}.pt")
    torch.save(target, directory / f"target-params-{index}.pt")


def test_exact_oracle_predictions_matching_artifacts_returns_zero(tmp_path: Path) -> None:
    """Report zero MSE only after exact tensor equality is established.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    values = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    _write_pair(predictions, values, values.clone())

    assert _assert_exact_oracle_predictions(predictions) == 0.0


def test_exact_oracle_predictions_without_predictions_raises(tmp_path: Path) -> None:
    """Reject an empty prediction artifact directory.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    predictions.mkdir()

    with pytest.raises(ValueError, match="no prediction artifacts"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_orphan_target_raises(tmp_path: Path) -> None:
    """Reject a target artifact without its paired prediction.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    torch.save(torch.zeros((1, 2)), predictions / "target-params-0.pt")

    with pytest.raises(ValueError, match="pred-0.pt"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_missing_target_raises(tmp_path: Path) -> None:
    """Reject a prediction artifact without its paired target.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    torch.save(torch.zeros((1, 2)), predictions / "pred-0.pt")

    with pytest.raises(ValueError, match="target-params-0.pt"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_non_batched_tensor_raises(tmp_path: Path) -> None:
    """Reject artifacts outside PredictionWriter's two-dimensional batch contract.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    _write_pair(predictions, torch.zeros(2), torch.zeros(2))

    with pytest.raises(ValueError, match="2D batched tensors"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_different_shape_raises(tmp_path: Path) -> None:
    """Reject paired tensors with different shapes before comparison.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    _write_pair(predictions, torch.zeros((1, 2)), torch.zeros((2, 1)))

    with pytest.raises(ValueError, match="shape mismatch"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_different_dtype_raises(tmp_path: Path) -> None:
    """Reject numerically compatible tensors with different dtypes.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    _write_pair(
        predictions,
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="dtype mismatch"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_empty_tensor_raises(tmp_path: Path) -> None:
    """Reject vacuous equality between empty tensors.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    values = torch.empty((0, 2))
    _write_pair(predictions, values, values.clone())

    with pytest.raises(ValueError, match="must not be empty"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_nonfinite_value_raises(tmp_path: Path) -> None:
    """Reject matching tensors when either contains non-finite values.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    values = torch.tensor([[float("nan")]])
    _write_pair(predictions, values, values.clone())

    with pytest.raises(ValueError, match="non-finite"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_different_value_raises(tmp_path: Path) -> None:
    """Report the maximum difference when an oracle value changes.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    _write_pair(predictions, torch.tensor([[0.25]]), torch.tensor([[0.5]]))

    with pytest.raises(ValueError, match="maximum absolute difference 0.25"):
        _assert_exact_oracle_predictions(predictions)


def test_exact_oracle_predictions_expected_rows_mismatch_raises(tmp_path: Path) -> None:
    """Reject exact paired predictions that omit one intended row.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    values = torch.zeros((99, 2))
    _write_pair(predictions, values, values.clone())

    with pytest.raises(ValueError, match="expected 100 rows, found 99"):
        _assert_exact_oracle_predictions(predictions, expected_rows=100)


def test_exact_oracle_predictions_complete_multibatch_returns_zero(tmp_path: Path) -> None:
    """Count rows across complete paired batches before certifying equality.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    _write_pair(predictions, torch.zeros((2, 3)), torch.zeros((2, 3)), index=0)
    _write_pair(predictions, torch.ones((1, 3)), torch.ones((1, 3)), index=1)

    assert _assert_exact_oracle_predictions(predictions, expected_rows=3) == 0.0


@pytest.mark.parametrize("expected_rows", [True, False, 0, -1, 1.0, "1"])
def test_predict_postprocessing_invalid_expected_rows_raises(
    tmp_path: Path, expected_rows: object
) -> None:
    """Reject non-positive and non-strict-integer expected row counts.

    :param tmp_path: Isolated artifact root.
    :param expected_rows: Invalid configured value under test.
    """
    predictions = tmp_path / "predictions"
    values = torch.zeros((1, 2))
    _write_pair(predictions, values, values.clone())
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "evaluation": {
                "compute_metrics": False,
                "oracle_expected_rows": expected_rows,
                "render_vst": False,
                "require_exact_param_oracle": True,
            },
        }
    )

    with pytest.raises(ValueError, match="oracle_expected_rows must be a positive integer"):
        _run_predict_postprocessing(cfg)


def test_predict_postprocessing_verified_oracle_persists_prefixed_metric(tmp_path: Path) -> None:
    """Preserve renderer-role namespacing on the exact oracle metric.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    values = torch.tensor([[0.0, 1.0]])
    _write_pair(predictions, values, values.clone())
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "evaluation": {
                "compute_metrics": False,
                "metric_prefix": "candidate/",
                "oracle_expected_rows": 1,
                "render_vst": False,
                "require_exact_param_oracle": True,
            },
        }
    )

    assert _run_predict_postprocessing(cfg) == {"candidate/oracle/param_mse": 0.0}
