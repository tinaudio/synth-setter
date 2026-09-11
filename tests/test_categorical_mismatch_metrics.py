"""Decoded categorical mismatch metric behavior."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer

from synth_setter.data.vst.param_spec import CategoricalParameter, ParamSpec
from synth_setter.metrics import (
    categorical_mismatch_rates,
    number_group_optimal_assignment_categorical_mismatch_rates,
)
from synth_setter.utils.callbacks import LogPerParamMSE


class _RecordingModule:
    """Capture callback metrics without constructing a trainer."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def log_dict(self, metrics: dict[str, float]) -> None:
        """Record one callback metric mapping.

        :param metrics: Metric names and values emitted by the callback.
        """
        self.logged.update(metrics)


def _model(*encoded: float) -> torch.Tensor:
    """Convert one encoded row to model space.

    :param *encoded: Encoded ``[0, 1]`` values for one row.
    :returns: One model-space row.
    """
    return torch.tensor([[value * 2 - 1 for value in encoded]])


def test_categorical_mismatch_scalar_snaps_at_nearest_raw_value_boundary() -> None:
    """Scalar categories use the renderer's nearest-raw-value boundary."""
    spec = ParamSpec(
        [CategoricalParameter("mode", ["low", "mid", "high"], raw_values=[0.0, 0.2, 1.0])],
        [],
    )

    below = categorical_mismatch_rates(_model(0.59), _model(0.2), spec)
    above = categorical_mismatch_rates(_model(0.61), _model(0.2), spec)

    assert below == {"mode": pytest.approx(0.0)}
    assert above == {"mode": pytest.approx(1.0)}


def test_categorical_mismatch_preserves_float64_boundary_side() -> None:
    """Model-to-encoded promotion retains a tiny positive offset above midpoint."""
    spec = ParamSpec([CategoricalParameter("mode", ["off", "on"])], [])
    predicted = torch.tensor([[1e-8]], dtype=torch.float64)
    target = torch.tensor([[1.0]], dtype=torch.float64)

    rates = categorical_mismatch_rates(predicted, target, spec)

    assert rates == {"mode": pytest.approx(0.0)}


def test_categorical_mismatch_onehot_decodes_by_argmax_once_per_field() -> None:
    """A wrong onehot category contributes one mismatch, not one per column."""
    spec = ParamSpec(
        [CategoricalParameter("shape", ["a", "b", "c"], encoding="onehot")],
        [],
    )
    predicted = _model(0.1, 0.8, 0.7)
    target = _model(1.0, 0.0, 0.0)

    rates = categorical_mismatch_rates(predicted, target, spec)

    assert rates == {"shape": pytest.approx(1.0)}
    assert len(rates) == 1


def test_categorical_mismatch_logs_non_numbered_field_in_validation_and_test() -> None:
    """The callback emits exact field names under both loop namespaces."""
    spec = ParamSpec([CategoricalParameter("mode", ["off", "on"])], [])
    callback = LogPerParamMSE("surge_4")
    callback.param_spec = spec
    module = _RecordingModule()
    trainer = cast("Trainer", None)
    pl_module = cast("LightningModule", module)
    batch = {"params": _model(0.0)}
    outputs = {"preds": _model(1.0)}

    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch, 0)
    callback.on_validation_epoch_end(trainer, pl_module)
    callback.on_test_epoch_start(trainer, pl_module)
    callback.on_test_batch_end(trainer, pl_module, outputs, batch, 0)
    callback.on_test_epoch_end(trainer, pl_module)

    assert module.logged["val/categorical_mismatch_rate/mode"] == pytest.approx(1.0)
    assert module.logged["test/categorical_mismatch_rate/mode"] == pytest.approx(1.0)
    assert module.logged[
        "val/number_group_optimal_assignment_categorical_mismatch_rate/mode"
    ] == pytest.approx(1.0)


def test_grouped_categorical_mismatch_assignment_corrects_numbered_swap() -> None:
    """A categorical numbered-family swap has zero grouped mismatch."""
    spec = ParamSpec(
        [
            CategoricalParameter("a_filter_1_type", ["low", "high"]),
            CategoricalParameter("a_filter_2_type", ["low", "high"]),
        ],
        [],
    )

    rates = number_group_optimal_assignment_categorical_mismatch_rates(
        _model(1.0, 0.0), _model(0.0, 1.0), spec
    )

    assert rates == {"a_filter_N_type": pytest.approx(0.0)}


def test_grouped_categorical_mismatch_assigns_each_sample_independently() -> None:
    """Opposing row imbalances cannot cancel through batch-pooled assignment."""
    spec = ParamSpec(
        [
            CategoricalParameter("filter_1_type", ["low", "high"]),
            CategoricalParameter("filter_2_type", ["low", "high"]),
        ],
        [],
    )
    predicted = torch.cat((_model(0.0, 0.0), _model(1.0, 1.0)))
    target = torch.cat((_model(0.0, 1.0), _model(0.0, 1.0)))

    rates = number_group_optimal_assignment_categorical_mismatch_rates(predicted, target, spec)

    assert rates == {"filter_N_type": pytest.approx(0.5)}


def test_grouped_categorical_mismatch_accepts_bfloat16() -> None:
    """Categorical decoding promotes bfloat16 before NumPy conversion."""
    spec = ParamSpec([CategoricalParameter("mode", ["off", "on"])], [])

    rates = categorical_mismatch_rates(
        torch.tensor([[-1.0]], dtype=torch.bfloat16),
        torch.tensor([[1.0]], dtype=torch.bfloat16),
        spec,
    )

    assert rates == {"mode": pytest.approx(1.0)}


def test_grouped_categorical_mismatch_uses_mismatch_not_mse_assignment() -> None:
    """Soft onehot magnitudes cannot override the decoded-category assignment."""
    spec = ParamSpec(
        [
            CategoricalParameter("osc_1_mode", ["a", "b"], encoding="onehot"),
            CategoricalParameter("osc_2_mode", ["a", "b"], encoding="onehot"),
        ],
        [],
    )
    predicted = _model(0.5865, 0.8397, 0.7265, 0.3650)
    target = _model(0.4484, 0.3677, 0.1097, 0.2032)

    rates = number_group_optimal_assignment_categorical_mismatch_rates(predicted, target, spec)

    assert rates == {"osc_N_mode": pytest.approx(0.0)}


def test_grouped_categorical_mismatch_disambiguates_collapsed_labels() -> None:
    """Incompatible numbered families retain distinct published metrics."""
    spec = ParamSpec(
        [
            CategoricalParameter("osc_1_mode", ["a", "b"]),
            CategoricalParameter("osc_2_mode", ["a", "b"]),
            CategoricalParameter("osc_11_mode", ["x", "y"]),
            CategoricalParameter("osc_12_mode", ["x", "y"]),
        ],
        [],
    )

    rates = number_group_optimal_assignment_categorical_mismatch_rates(
        _model(0.0, 0.0, 0.0, 0.0), _model(0.0, 0.0, 0.0, 0.0), spec
    )

    assert rates.keys() == {
        "osc_N_mode__members_osc_1_mode-osc_2_mode",
        "osc_N_mode__members_osc_11_mode-osc_12_mode",
    }


def test_grouped_categorical_mismatch_does_not_combine_incompatible_domains() -> None:
    """Same-width numbered fields with different domains remain singletons."""
    spec = ParamSpec(
        [
            CategoricalParameter("osc_1_mode", ["a", "b"]),
            CategoricalParameter("osc_2_mode", ["x", "y"]),
        ],
        [],
    )

    rates = number_group_optimal_assignment_categorical_mismatch_rates(
        _model(1.0, 0.0), _model(0.0, 1.0), spec
    )

    assert rates == {
        "osc_1_mode": pytest.approx(1.0),
        "osc_2_mode": pytest.approx(1.0),
    }


def test_categorical_mismatch_callback_weights_ragged_batches_by_sample_count() -> None:
    """Ragged batches contribute one mismatch vote per sample."""
    spec = ParamSpec([CategoricalParameter("mode", ["off", "on"])], [])
    callback = LogPerParamMSE("surge_4")
    callback.param_spec = spec
    module = _RecordingModule()
    trainer = cast("Trainer", None)
    pl_module = cast("LightningModule", module)
    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(
        trainer, pl_module, {"preds": _model(0.0)}, {"params": _model(0.0)}, 0
    )
    callback.on_validation_batch_end(
        trainer,
        pl_module,
        {"preds": _model(1.0).repeat(3, 1)},
        {"params": _model(0.0).repeat(3, 1)},
        1,
    )

    callback.on_validation_epoch_end(trainer, pl_module)

    assert module.logged["val/categorical_mismatch_rate/mode"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("predicted", "target", "message"),
    [
        (torch.zeros(2), torch.zeros(2), "matching 2-D shapes"),
        (torch.zeros(1, 2), torch.zeros(1, 1), "matching 2-D shapes"),
        (torch.zeros(1, 2), torch.zeros(1, 2), "ParamSpec width"),
        (torch.empty(0, 1), torch.empty(0, 1), "non-empty batch"),
        (torch.tensor([[float("nan")]]), torch.zeros(1, 1), "finite values"),
    ],
)
def test_categorical_mismatch_rejects_malformed_or_nonfinite_inputs(
    predicted: torch.Tensor, target: torch.Tensor, message: str
) -> None:
    """Malformed, empty, and non-finite model rows fail explicitly.

    :param predicted: Candidate prediction shape or values.
    :param target: Candidate target shape or values.
    :param message: Expected validation-error fragment.
    """
    spec = ParamSpec([CategoricalParameter("mode", ["off", "on"])], [])

    with pytest.raises(ValueError, match=message):
        categorical_mismatch_rates(predicted, target, spec)
