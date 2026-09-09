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


def test_exact_oracle_predictions_missing_target_raises(tmp_path: Path) -> None:
    """Reject a prediction artifact without its paired target.

    :param tmp_path: Isolated artifact root.
    """
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    torch.save(torch.zeros((1, 2)), predictions / "pred-0.pt")

    with pytest.raises(ValueError, match="target-params-0.pt"):
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
                "render_vst": False,
                "require_exact_param_oracle": True,
            },
        }
    )

    assert _run_predict_postprocessing(cfg) == {"candidate/oracle/param_mse": 0.0}
