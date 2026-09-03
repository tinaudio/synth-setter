"""Behavioral tests for ParamSpec-canonicalized parameter MSE."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteArrayParameter,
    DiscreteLiteralParameter,
    ParamSpec,
    spec_quantize_model_output,
)
from synth_setter.metrics import spec_quantized_per_param_mse


def test_spec_quantized_per_param_mse_clips_continuous_prediction() -> None:
    """Out-of-range continuous predictions score at the rendered boundary."""
    spec = ParamSpec([ContinuousParameter("gain")], [])

    result = spec_quantized_per_param_mse(
        torch.tensor([[1.5]]),
        torch.tensor([[1.0]]),
        spec,
    )

    assert result.item() == 0.0


def test_spec_quantized_per_param_mse_onehot_category_uses_selected_value() -> None:
    """Onehot logits score as the category selected for rendering."""
    spec = ParamSpec(
        [CategoricalParameter("mode", ["a", "b"], encoding="onehot")],
        [],
    )

    result = spec_quantized_per_param_mse(
        torch.tensor([[-0.2, 0.8]]),
        torch.tensor([[-1.0, 1.0]]),
        spec,
    )

    assert torch.equal(result, torch.zeros(2))


def test_spec_quantized_per_param_mse_scalar_discrete_uses_integral_value() -> None:
    """Scalar discrete predictions score as the integer passed to the renderer."""
    spec = ParamSpec([], [DiscreteLiteralParameter("pitch", min=0, max=4)])

    result = spec_quantized_per_param_mse(
        torch.tensor([[0.1]]),
        torch.tensor([[0.0]]),
        spec,
    )

    assert result.item() == 0.0


def test_spec_quantized_per_param_mse_scalar_category_uses_nearest_host_value() -> None:
    """Scalar categorical predictions snap to the nearest host raw value."""
    spec = ParamSpec(
        synth_params=[
            CategoricalParameter(
                "mode",
                ["a", "b"],
                raw_values=[0.25, 0.75],
                encoding="scalar",
            )
        ],
        note_params=[],
    )

    result = spec_quantized_per_param_mse(
        torch.tensor([[0.2]]),
        torch.tensor([[0.5]]),
        spec,
    )

    assert result.item() == 0.0


def test_spec_quantize_model_output_matches_rendered_values() -> None:
    """NumPy postprocessing exposes the same clipped and snapped model-space row."""
    spec = ParamSpec(
        [
            ContinuousParameter("gain"),
            CategoricalParameter(
                "mode",
                ["a", "b"],
                raw_values=[0.25, 0.75],
                encoding="scalar",
            ),
        ],
        [DiscreteLiteralParameter("pitch", min=0, max=4)],
    )

    result = spec_quantize_model_output(np.array([1.5, 0.2, 0.1]), spec)

    np.testing.assert_array_equal(result, np.array([1.0, 0.5, 0.0]))


def test_spec_quantized_per_param_mse_wrong_continuous_value_reports_error() -> None:
    """A genuine rendered-value mismatch produces a nonzero score."""
    spec = ParamSpec([ContinuousParameter("gain")], [])

    result = spec_quantized_per_param_mse(
        torch.tensor([[0.0]]),
        torch.tensor([[1.0]]),
        spec,
    )

    assert result.item() == 1.0


def test_spec_quantized_per_param_mse_rounds_discrete_array() -> None:
    """Discrete array coordinates score at their nearest integral values."""
    spec = ParamSpec([DiscreteArrayParameter("steps", shape=(2,), min=0, max=4)], [])

    result = spec_quantized_per_param_mse(
        torch.tensor([[0.3, -0.3]]),
        torch.tensor([[0.0, 0.0]]),
        spec,
    )

    assert torch.equal(result, torch.tensor([0.25, 0.25]))


def test_spec_quantized_per_param_mse_matches_render_postprocessor() -> None:
    """Training-time scoring and offline rendering share quantization semantics."""
    spec = ParamSpec(
        [
            CategoricalParameter(
                "mode",
                ["a", "b"],
                raw_values=[0.25, 0.75],
                encoding="scalar",
            ),
            CategoricalParameter("shape", ["sine", "saw"], encoding="onehot"),
            DiscreteArrayParameter("steps", shape=(2,), min=0, max=4),
        ],
        [DiscreteLiteralParameter("pitch", min=0, max=4)],
    )
    predicted = torch.tensor([[0.2, -0.2, 0.8, 0.3, -0.3, 0.3]])
    rendered = torch.from_numpy(spec_quantize_model_output(predicted[0].numpy(), spec)).unsqueeze(
        0
    )

    result = spec_quantized_per_param_mse(predicted, rendered, spec)

    assert torch.equal(result, torch.zeros(6))


def test_spec_quantized_per_param_mse_nonfinite_prediction_raises_value_error() -> None:
    """A NaN category cannot collapse into a finite, misleading score."""
    spec = ParamSpec([CategoricalParameter("mode", ["a", "b"], encoding="onehot")], [])

    with pytest.raises(ValueError, match="finite"):
        spec_quantized_per_param_mse(torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, 2), spec)


def test_spec_quantize_model_output_nonfinite_prediction_raises_value_error() -> None:
    """Offline rendering rejects NaN instead of selecting the first category."""
    spec = ParamSpec([CategoricalParameter("mode", ["a", "b"], encoding="onehot")], [])

    with pytest.raises(ValueError, match="finite"):
        spec_quantize_model_output(np.array([np.nan, 0.0]), spec)


def test_spec_quantized_per_param_mse_mismatched_shapes_raise_value_error() -> None:
    """Quantized scoring rejects shapes that could broadcast silently."""
    spec = ParamSpec([ContinuousParameter("gain")], [])

    with pytest.raises(ValueError, match="matching 2-D shapes"):
        spec_quantized_per_param_mse(torch.zeros(2, 1), torch.zeros(1, 1), spec)
