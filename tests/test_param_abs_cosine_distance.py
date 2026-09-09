"""Absolute cosine distance on logical parameter geometry."""

import pytest
import torch

from synth_setter.data.vst.param_spec import (
    AngleArrayParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    DirectionArrayParameter,
    DiscreteArrayParameter,
    ParamSpec,
)
from synth_setter.metrics import spec_per_param_abs_cosine_distance


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [([2.0, 0.0], 0.0), ([-2.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([3.0, 4.0], 0.4)],
)
def test_abs_cosine_direction_alignment_returns_distance(
    prediction: list[float], expected: float
) -> None:
    """Direction distance ignores magnitude and sign, but not orientation.

    :param prediction: Predicted model-space direction.
    :param expected: Absolute cosine distance from the first basis vector.
    """
    spec = ParamSpec([DirectionArrayParameter("direction", (2,))], [])

    result = spec_per_param_abs_cosine_distance(
        torch.tensor([prediction]), torch.tensor([[1.0, 0.0]]), spec
    )

    assert result["direction"].item() == pytest.approx(expected)


@pytest.mark.parametrize("parameter_type", [ContinuousArrayParameter, DiscreteArrayParameter])
def test_abs_cosine_array_multidimensional_span_uses_whole_vector(
    parameter_type: type[ContinuousArrayParameter],
) -> None:
    """Array coordinates contribute one cosine, not one scalar cosine each.

    :param parameter_type: Ordinary continuous or discrete array schema.
    """
    spec = ParamSpec([parameter_type("array", (2, 2), -10, 10)], [])

    result = spec_per_param_abs_cosine_distance(
        torch.tensor([[3.0, 0.0, 0.0, 4.0]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), spec
    )

    assert result["array"].item() == pytest.approx(0.4)


def test_abs_cosine_angles_opposite_pairs_do_not_cancel() -> None:
    """Absolute value is taken per angle before averaging the logical array."""
    spec = ParamSpec([AngleArrayParameter("angles", (2,))], [])

    result = spec_per_param_abs_cosine_distance(
        torch.tensor([[2.0, 0.0, -3.0, 0.0]]), torch.tensor([[1.0, 0.0, 1.0, 0.0]]), spec
    )

    assert result["angles"].item() == pytest.approx(0.0)


def test_abs_cosine_angles_multiple_samples_keeps_pairs_within_each_sample() -> None:
    """Each prediction is compared only with its own sample's target angles."""
    spec = ParamSpec([AngleArrayParameter("angles", (2,))], [])

    result = spec_per_param_abs_cosine_distance(
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
        spec,
    )

    assert result["angles"].item() == pytest.approx(0.75)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("target", [[1.0, 0.0], [0.0, 0.0]])
def test_abs_cosine_zero_vector_returns_maximum_distance(
    dtype: torch.dtype, target: list[float]
) -> None:
    """A directionless prediction never earns alignment credit or produces NaN.

    :param dtype: Model output precision.
    :param target: Nonzero or directionless target vector.
    """
    spec = ParamSpec([DirectionArrayParameter("direction", (2,))], [])

    result = spec_per_param_abs_cosine_distance(
        torch.zeros(1, 2, dtype=dtype), torch.tensor([target], dtype=dtype), spec
    )

    assert result["direction"].item() == 1.0


def test_abs_cosine_mixed_spec_preserves_spans_and_excludes_scalars() -> None:
    """Non-array columns do not shift or contaminate the array measurements."""
    spec = ParamSpec(
        [ContinuousParameter("scalar", -1, 1), AngleArrayParameter("angles", (2,))],
        [DirectionArrayParameter("direction", (2,))],
    )

    result = spec_per_param_abs_cosine_distance(
        torch.tensor([[0.9, 1.0, 0.0, 0.0, 2.0, -3.0, -4.0]]),
        torch.tensor([[0.1, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]),
        spec,
    )

    assert {name: value.item() for name, value in result.items()} == pytest.approx(
        {"angles": 0.5, "direction": 0.4}
    )


def test_abs_cosine_angle_wrapping_matches_physical_alignment() -> None:
    """The periodic seam does not make nearby angles appear distant."""
    spec = ParamSpec([AngleArrayParameter("angles", (1,))], [])
    prediction = torch.tensor(spec.encode({"angles": [torch.pi]}, {})).unsqueeze(0) * 2 - 1
    target = torch.tensor(spec.encode({"angles": [-torch.pi]}, {})).unsqueeze(0) * 2 - 1

    result = spec_per_param_abs_cosine_distance(prediction, target, spec)

    assert result["angles"].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (torch.ones(2), torch.ones(2), "matching 2-D shapes"),
        (torch.ones(1, 2), torch.ones(2, 2), "matching 2-D shapes"),
        (torch.ones(1, 3), torch.ones(1, 3), "ParamSpec width"),
        (torch.empty(0, 2), torch.empty(0, 2), "non-empty batch"),
        (torch.tensor([[float("nan"), 0.0]]), torch.ones(1, 2), "finite"),
        (torch.ones(1, 2), torch.tensor([[0.0, float("inf")]]), "finite"),
    ],
)
def test_abs_cosine_invalid_inputs_raise_value_error(
    prediction: torch.Tensor, target: torch.Tensor, message: str
) -> None:
    """Malformed inputs fail rather than log plausible but incorrect distances.

    :param prediction: Malformed prediction or valid counterpart.
    :param target: Malformed target or valid counterpart.
    :param message: Diagnostic identifying the violated contract.
    """
    spec = ParamSpec([DirectionArrayParameter("direction", (2,))], [])

    with pytest.raises(ValueError, match=message):
        spec_per_param_abs_cosine_distance(prediction, target, spec)
