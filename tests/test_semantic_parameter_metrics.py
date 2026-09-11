"""Renderer-effective semantic parameter metric behavior."""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer

from synth_setter.data.vst.param_spec import (
    AngleArrayParameter,
    CategoricalParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    DirectionArrayParameter,
    DiscreteArrayParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)
from synth_setter.metrics import semantic_parameter_distances
from synth_setter.utils.callbacks import LogPerParamMSE


class _RecordingModule:
    """Capture callback metrics without constructing a trainer."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def log_dict(self, metrics: dict[str, float]) -> None:
        """Record callback metrics.

        :param metrics: Metric names and values emitted by the callback.
        """
        self.logged.update(metrics)


def _model(*values: float) -> torch.Tensor:
    """Build one model-space row.

    :param *values: Model-space coordinates.
    :returns: One model-space row.
    """
    return torch.tensor([values], dtype=torch.float32)


def test_angle_semantic_distance_wraps_across_periodic_seam() -> None:
    """Angles adjacent across the negative/positive pi seam remain close."""
    spec = ParamSpec([AngleArrayParameter("phase", (1,))], [])
    epsilon = 0.01

    metrics = semantic_parameter_distances(
        _model(-math.cos(epsilon), math.sin(epsilon)),
        _model(-math.cos(epsilon), -math.sin(epsilon)),
        spec,
    )

    assert metrics["angular_mae_radians/phase"].item() == pytest.approx(2 * epsilon)


def test_angle_semantic_distance_distinguishes_pi_separation() -> None:
    """Opposite angle pairs have the maximum semantic angular error."""
    spec = ParamSpec([AngleArrayParameter("phase", (1,))], [])

    metrics = semantic_parameter_distances(_model(-1.0, 0.0), _model(1.0, 0.0), spec)

    assert metrics["angular_mae_radians/phase"].item() == pytest.approx(torch.pi)


def test_angle_semantic_distance_averages_native_angles_and_samples() -> None:
    """Every native angle in every sample contributes equally."""
    spec = ParamSpec([AngleArrayParameter("phase", (2,))], [])
    predicted = torch.tensor([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])

    metrics = semantic_parameter_distances(predicted, target, spec)

    assert metrics["angular_mae_radians/phase"].item() == pytest.approx(torch.pi / 2)


def test_angle_semantic_distance_projects_off_circle_pair_by_direction() -> None:
    """An off-circle pair decodes to the angle of its direction."""
    spec = ParamSpec([AngleArrayParameter("phase", (1,))], [])

    metrics = semantic_parameter_distances(_model(3.0, 4.0), _model(1.0, 0.0), spec)

    assert metrics["angular_mae_radians/phase"].item() == pytest.approx(0.927295218)


def test_angle_semantic_distance_decodes_zero_pair_to_zero_angle() -> None:
    """A directionless pair follows the renderer's zero-angle fallback."""
    spec = ParamSpec([AngleArrayParameter("phase", (1,))], [])

    metrics = semantic_parameter_distances(_model(0.0, 0.0), _model(1.0, 0.0), spec)

    assert metrics["angular_mae_radians/phase"].item() == 0.0


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [((-1.0, 0.0), 0.0), ((0.0, 1.0), torch.pi / 2)],
)
def test_direction_semantic_distance_is_sign_invariant_axis_angle(
    prediction: tuple[float, float], expected: float
) -> None:
    """Householder axes identify signs but retain orthogonal separation.

    :param prediction: Predicted model-space direction.
    :param expected: Axis angle from the first basis vector.
    """
    spec = ParamSpec([DirectionArrayParameter("axis", (2,))], [])

    metrics = semantic_parameter_distances(_model(*prediction), _model(1.0, 0.0), spec)

    assert metrics["axis_angular_error_radians/axis"].item() == pytest.approx(expected)


def test_direction_semantic_distance_decodes_zero_to_first_basis_vector() -> None:
    """A directionless model output follows the renderer's deterministic fallback."""
    spec = ParamSpec([DirectionArrayParameter("axis", (2,))], [])

    metrics = semantic_parameter_distances(_model(0.0, 0.0), _model(0.0, 1.0), spec)

    assert metrics["axis_angular_error_radians/axis"].item() == pytest.approx(torch.pi / 2)


def test_discrete_literal_semantic_distance_rounds_half_step_up() -> None:
    """A scalar integer exactly at a half step rounds upward."""
    spec = ParamSpec([DiscreteLiteralParameter("steps", 0, 4)], [])

    metrics = semantic_parameter_distances(_model(-0.75), _model(-1.0), spec)

    assert metrics["discrete_mae/steps"].item() == 1.0
    assert metrics["discrete_mismatch_rate/steps"].item() == 1.0


def test_discrete_literal_semantic_distance_preserves_float64_boundary_side() -> None:
    """A tiny float64 offset below a half step remains below that step."""
    spec = ParamSpec([DiscreteLiteralParameter("steps", 0, 4)], [])
    predicted = torch.tensor([[-0.75000001]], dtype=torch.float64)
    target = torch.tensor([[-1.0]], dtype=torch.float64)

    metrics = semantic_parameter_distances(predicted, target, spec)

    assert metrics["discrete_mae/steps"].item() == 0.0
    assert metrics["discrete_mismatch_rate/steps"].item() == 0.0


def test_discrete_literal_semantic_distance_clips_model_output() -> None:
    """An overshooting scalar prediction decodes to the upper native bound."""
    spec = ParamSpec([DiscreteLiteralParameter("steps", 0, 4)], [])

    metrics = semantic_parameter_distances(_model(2.0), _model(1.0), spec)

    assert metrics["discrete_mae/steps"].item() == 0.0
    assert metrics["discrete_mismatch_rate/steps"].item() == 0.0


def test_discrete_array_semantic_distance_uses_array_rounding_and_clipping() -> None:
    """Array integers use NumPy rounding and clip predictions before decoding."""
    spec = ParamSpec([DiscreteArrayParameter("taps", (2,), 0, 4)], [])

    metrics = semantic_parameter_distances(_model(-0.25, 2.0), _model(0.0, 0.5), spec)

    assert metrics["discrete_mae/taps"].item() == pytest.approx(0.5)
    assert metrics["discrete_mismatch_rate/taps"].item() == pytest.approx(0.5)


def test_discrete_array_mismatch_counts_one_vote_per_native_element() -> None:
    """Mismatch magnitude does not change each native element's single vote."""
    spec = ParamSpec([DiscreteArrayParameter("taps", (4,), 0, 4)], [])

    metrics = semantic_parameter_distances(
        _model(-1.0, -1.0, -1.0, 1.0),
        _model(-1.0, -1.0, -1.0, -1.0),
        spec,
    )

    assert metrics["discrete_mismatch_rate/taps"].item() == pytest.approx(0.25)
    assert metrics["discrete_mae/taps"].item() == pytest.approx(1.0)


def test_note_duration_semantic_distance_reports_native_seconds() -> None:
    """Start and end errors are decoded to seconds before averaging."""
    spec = ParamSpec([], [NoteDurationParameter("timing", 10.0)])

    metrics = semantic_parameter_distances(_model(-0.2, 0.2), _model(-0.6, 0.6), spec)

    assert metrics["note_timing_mae_seconds/timing"].item() == pytest.approx(2.0)


def test_semantic_distance_mixed_spec_preserves_spans_and_finite_inventory() -> None:
    """Unsupported and earlier fields neither emit keys nor shift specialized spans."""
    spec = ParamSpec(
        [
            ContinuousParameter("gain", 0.0, 1.0),
            CategoricalParameter("mode", ["a", "b"]),
            AngleArrayParameter("phase", (1,)),
            ContinuousArrayParameter("curve", (2,), -1.0, 1.0),
            DiscreteLiteralParameter("steps", 0, 2),
        ],
        [NoteDurationParameter("timing", 4.0)],
    )
    predicted = _model(0.9, 1.0, 0.0, 1.0, -0.7, 0.8, 0.9, -1.0, 1.0)
    target = _model(-0.9, -1.0, 1.0, 0.0, 0.7, -0.8, -0.9, 1.0, -1.0)

    metrics = semantic_parameter_distances(predicted, target, spec)

    assert set(metrics) == {
        "angular_mae_radians/phase",
        "discrete_mae/steps",
        "discrete_mismatch_rate/steps",
        "note_timing_mae_seconds/timing",
    }
    assert metrics["angular_mae_radians/phase"].item() == pytest.approx(torch.pi / 2)
    assert metrics["discrete_mae/steps"].item() == 2.0
    assert metrics["note_timing_mae_seconds/timing"].item() == pytest.approx(4.0)


def test_semantic_distance_callback_logs_validation_and_test_namespaces() -> None:
    """Semantic metric names retain their namespaces in both evaluation loops."""
    spec = ParamSpec([DiscreteLiteralParameter("steps", 0, 4)], [])
    callback = LogPerParamMSE("surge_4")
    callback.param_spec = spec
    module = _RecordingModule()
    trainer = cast("Trainer", None)
    pl_module = cast("LightningModule", module)
    outputs = {"preds": _model(1.0)}
    batch = {"params": _model(-1.0)}

    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch, 0)
    callback.on_validation_epoch_end(trainer, pl_module)
    callback.on_test_epoch_start(trainer, pl_module)
    callback.on_test_batch_end(trainer, pl_module, outputs, batch, 0)
    callback.on_test_epoch_end(trainer, pl_module)

    assert module.logged["val/discrete_mae/steps"] == pytest.approx(4.0)
    assert module.logged["test/discrete_mismatch_rate/steps"] == pytest.approx(1.0)


def test_semantic_distance_callback_weights_ragged_batches_by_sample_count() -> None:
    """Epoch means weight semantic batch means by their sample counts."""
    spec = ParamSpec([DiscreteLiteralParameter("steps", 0, 4)], [])
    callback = LogPerParamMSE("surge_4")
    callback.param_spec = spec
    module = _RecordingModule()
    trainer = cast("Trainer", None)
    pl_module = cast("LightningModule", module)
    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(
        trainer, pl_module, {"preds": _model(-1.0)}, {"params": _model(-1.0)}, 0
    )
    callback.on_validation_batch_end(
        trainer,
        pl_module,
        {"preds": _model(1.0).repeat(3, 1)},
        {"params": _model(-1.0).repeat(3, 1)},
        1,
    )

    callback.on_validation_epoch_end(trainer, pl_module)

    assert module.logged["val/discrete_mae/steps"] == pytest.approx(3.0)
    assert module.logged["val/discrete_mismatch_rate/steps"] == pytest.approx(0.75)


def test_semantic_distance_callback_weights_samples_across_distributed_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed sums and counts determine the semantic epoch mean.

    :param monkeypatch: Supplies three remote samples with maximum native error.
    """
    spec = ParamSpec([DiscreteLiteralParameter("steps", 0, 4)], [])
    callback = LogPerParamMSE("surge_4")
    callback.param_spec = spec
    module = _RecordingModule()
    trainer = cast("Trainer", None)
    pl_module = cast("LightningModule", module)
    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(
        trainer, pl_module, {"preds": _model(-1.0)}, {"params": _model(-1.0)}, 0
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def add_remote_samples(total_and_count: torch.Tensor) -> None:
        total_and_count[:-1] += 12.0
        total_and_count[-1] += 3.0

    monkeypatch.setattr(torch.distributed, "all_reduce", add_remote_samples)

    callback.on_validation_epoch_end(trainer, pl_module)

    assert module.logged["val/discrete_mae/steps"] == pytest.approx(3.0)


def test_semantic_distance_callback_preserves_abs_cosine_metric() -> None:
    """Direction semantic logging leaves the existing geometric key available."""
    spec = ParamSpec([DirectionArrayParameter("axis", (2,))], [])
    callback = LogPerParamMSE("surge_4")
    callback.param_spec = spec
    module = _RecordingModule()
    trainer = cast("Trainer", None)
    pl_module = cast("LightningModule", module)
    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(
        trainer, pl_module, {"preds": _model(0.0, 1.0)}, {"params": _model(1.0, 0.0)}, 0
    )

    callback.on_validation_epoch_end(trainer, pl_module)

    assert module.logged["val/axis_angular_error_radians/axis"] == pytest.approx(torch.pi / 2)
    assert module.logged["val/per_param_abs_cosine_distance/axis"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("predicted", "target", "message"),
    [
        (torch.zeros(2), torch.zeros(2), "matching 2-D shapes"),
        (torch.zeros(1, 2), torch.zeros(2, 2), "matching 2-D shapes"),
        (torch.zeros(1, 3), torch.zeros(1, 3), "ParamSpec width"),
        (torch.empty(0, 2), torch.empty(0, 2), "non-empty batch"),
        (torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, 2), "finite values"),
        (torch.zeros(1, 2), torch.tensor([[0.0, float("inf")]]), "finite values"),
    ],
)
def test_semantic_distance_rejects_malformed_or_nonfinite_inputs(
    predicted: torch.Tensor, target: torch.Tensor, message: str
) -> None:
    """Invalid rows fail before any plausible metric is returned.

    :param predicted: Candidate prediction tensor.
    :param target: Candidate target tensor.
    :param message: Expected validation-error fragment.
    """
    spec = ParamSpec([AngleArrayParameter("phase", (1,))], [])

    with pytest.raises(ValueError, match=message):
        semantic_parameter_distances(predicted, target, spec)
