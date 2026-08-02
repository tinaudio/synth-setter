"""Behavioral tests for :class:`synth_setter.metrics.BestSwapParamMSE`.

The metric is the optimistic bracket to plain ``param_mse``: the MSE after the
error-minimizing one-to-one matching of predicted scalars to target scalars,
which for squared error is sort-both-and-compare (rearrangement inequality).
"""

from __future__ import annotations

import pytest
import torch

from synth_setter.data.vst.param_spec import CategoricalParameter, ContinuousParameter, ParamSpec
from synth_setter.metrics import (
    BestSwapParamMSE,
    NumberGroupSwapParamMSE,
    best_swap_per_param_mse,
    number_group_swap_per_param_mse,
)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Plain elementwise MSE reference.

    :param pred: Predicted vectors.
    :param target: Target vectors.
    :returns: Mean squared error as a float.
    """
    return (pred - target).square().mean().item()


def test_best_swap_per_param_mse_attributes_errors_to_target_dimensions() -> None:
    """Matched errors retain the target parameter identity after sorting."""
    predicted = torch.tensor([[1.0, 4.0, 8.0], [2.0, 4.0, 7.0]])
    target = torch.tensor([[7.0, 1.0, 3.0], [6.0, 1.0, 4.0]])

    result = best_swap_per_param_mse(predicted, target)

    assert torch.equal(result, torch.tensor([1.0, 0.5, 0.5]))


def test_best_swap_per_param_mse_repeated_targets_use_stable_identity() -> None:
    """Equal targets receive matches in their original order deterministically."""
    predicted = torch.tensor([[0.0, 3.0, 5.0]])
    target = torch.tensor([[2.0, 2.0, 4.0]])

    result = best_swap_per_param_mse(predicted, target)

    assert torch.equal(result, torch.tensor([4.0, 1.0, 1.0]))


def test_best_swap_per_param_mse_invalid_shapes_raise_value_error() -> None:
    """Per-parameter scoring rejects inputs the scalar metric rejects."""
    with pytest.raises(ValueError, match="matching 2-D shapes"):
        best_swap_per_param_mse(torch.zeros(2, 4), torch.zeros(2, 5))


def test_number_group_swap_per_param_mse_swaps_collapsed_name_group() -> None:
    """Names differing only by number values can exchange their predictions."""
    spec = ParamSpec(
        synth_params=[
            ContinuousParameter("osc 1 gain"),
            ContinuousParameter("osc2 gain"),
            ContinuousParameter("master gain"),
        ],
        note_params=[],
    )
    predicted = torch.tensor([[0.0, 1.0, 0.5]])
    target = torch.tensor([[1.0, 0.0, 1.0]])

    result = number_group_swap_per_param_mse(predicted, target, spec)

    assert torch.equal(result, torch.tensor([0.0, 0.0, 0.25]))


def test_number_group_swap_per_param_mse_non_2d_input_raises_value_error() -> None:
    """Parameter matching rejects rows without a batch dimension."""
    spec = ParamSpec(synth_params=[ContinuousParameter("osc_1")], note_params=[])

    with pytest.raises(ValueError, match="matching 2-D shapes"):
        number_group_swap_per_param_mse(torch.zeros(1), torch.zeros(1), spec)


def test_number_group_swap_per_param_mse_wrong_spec_width_raises_value_error() -> None:
    """A spec that cannot label every encoded column is rejected."""
    spec = ParamSpec(synth_params=[ContinuousParameter("osc_1")], note_params=[])

    with pytest.raises(ValueError, match="ParamSpec width 1, got 2"):
        number_group_swap_per_param_mse(torch.zeros(1, 2), torch.zeros(1, 2), spec)


def test_number_group_swap_per_param_mse_matches_multi_column_params_as_units() -> None:
    """A multi-column parameter uses one coherent assignment for all columns."""
    spec = ParamSpec(
        synth_params=[
            CategoricalParameter("osc_1_shape", values=["saw", "square"], encoding="onehot"),
            CategoricalParameter("osc_2_shape", values=["saw", "square"], encoding="onehot"),
        ],
        note_params=[],
    )
    predicted = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    target = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

    result = number_group_swap_per_param_mse(predicted, target, spec)

    assert torch.equal(result, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_number_group_swap_per_param_mse_multi_column_assignment_minimizes_difference() -> None:
    """Multi-column matching uses squared differences rather than independent columns."""
    spec = ParamSpec(
        synth_params=[
            CategoricalParameter("osc_1_shape", values=["saw", "square"], encoding="onehot"),
            CategoricalParameter("osc_2_shape", values=["saw", "square"], encoding="onehot"),
        ],
        note_params=[],
    )
    predicted = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    result = number_group_swap_per_param_mse(predicted, target, spec)

    assert torch.equal(result, torch.zeros(4))


def test_number_group_swap_per_param_mse_repeated_targets_use_stable_identity() -> None:
    """Equal scalar targets retain deterministic target attribution."""
    spec = ParamSpec(
        synth_params=[ContinuousParameter("osc_1"), ContinuousParameter("osc_2")],
        note_params=[],
    )

    result = number_group_swap_per_param_mse(
        torch.tensor([[0.0, 3.0]]), torch.tensor([[2.0, 2.0]]), spec
    )

    assert torch.equal(result, torch.tensor([4.0, 1.0]))


def test_number_group_swap_per_param_mse_bf16_returns_float32() -> None:
    """Low-precision inputs produce finite float32 errors."""
    spec = ParamSpec(
        synth_params=[ContinuousParameter("osc_1"), ContinuousParameter("osc_2")],
        note_params=[],
    )

    result = number_group_swap_per_param_mse(
        torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16),
        torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        spec,
    )

    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_number_group_swap_per_param_mse_does_not_swap_across_groups() -> None:
    """A low-cost cross-family permutation remains penalized."""
    spec = ParamSpec(
        synth_params=[
            ContinuousParameter("osc_1_gain"),
            ContinuousParameter("osc_2_gain"),
            ContinuousParameter("filter_1_cutoff"),
            ContinuousParameter("filter_2_cutoff"),
        ],
        note_params=[],
    )
    predicted = torch.tensor([[0.0, 1.0, 10.0, 11.0]])
    target = torch.tensor([[10.0, 11.0, 0.0, 1.0]])

    result = number_group_swap_per_param_mse(predicted, target, spec)

    assert torch.equal(result, torch.full((4,), 100.0))


class TestNumberGroupSwapParamMSE:
    """Contract: swaps are limited to number-collapsed parameter-name groups."""

    def test_accumulates_element_mean_over_updates(self) -> None:
        """Compute returns the element mean across batches of different sizes."""
        spec = ParamSpec(
            synth_params=[ContinuousParameter("osc_1"), ContinuousParameter("osc_2")],
            note_params=[],
        )
        metric = NumberGroupSwapParamMSE(spec)
        metric.update(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]]))
        metric.update(
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        )

        assert metric.compute().item() == pytest.approx(2.0 / 3.0)


class TestBestSwapParamMSE:
    """Contract: permutation-invariant floor of the plain elementwise MSE."""

    def test_identical_vectors_scores_zero(self) -> None:
        """Exact agreement is the metric's zero point."""
        params = torch.tensor([[0.3, -0.7, 0.1, 0.9]])
        metric = BestSwapParamMSE()
        metric.update(params, params)
        assert metric.compute().item() == 0.0

    def test_permuted_target_scores_zero_while_mse_is_large(self) -> None:
        """Reorderings cost nothing here while plain MSE pays full price."""
        pred = torch.tensor([[0.9, -0.5, 0.2, 0.0]])
        target = pred[:, torch.tensor([2, 0, 3, 1])]
        metric = BestSwapParamMSE()
        metric.update(pred, target)
        assert metric.compute().item() < 1e-12
        assert _mse(pred, target) > 0.1

    def test_value_error_is_still_penalized(self) -> None:
        """No permutation can hide a genuinely wrong value."""
        pred = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[1.0, 1.0]])
        metric = BestSwapParamMSE()
        metric.update(pred, target)
        assert abs(metric.compute().item() - 1.0) < 1e-6

    def test_never_exceeds_plain_mse(self) -> None:
        """Floor property: sorting is the optimal matching, so it lower-bounds MSE."""
        generator = torch.Generator().manual_seed(7)
        pred = torch.randn(16, 92, generator=generator)
        target = torch.randn(16, 92, generator=generator)
        metric = BestSwapParamMSE()
        metric.update(pred, target)
        assert metric.compute().item() <= _mse(pred, target) + 1e-6

    def test_matching_is_per_sample_not_cross_batch(self) -> None:
        """Sample 1's shift cannot borrow sample 0's values via cross-batch matching."""
        # Sample 0 permuted, sample 1 shifted: a cross-batch matching could hide
        # sample 1's error against sample 0's values; per-sample matching cannot.
        pred = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
        metric = BestSwapParamMSE()
        metric.update(pred, target)
        expected = (0.0 + ((2.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2) / 2
        assert abs(metric.compute().item() - expected) < 1e-6

    def test_accumulates_mean_over_multiple_updates(self) -> None:
        """Compute() returns the element mean across all accumulated updates."""
        metric = BestSwapParamMSE()
        metric.update(torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, 0.0]]))
        metric.update(torch.tensor([[0.0, 0.0]]), torch.tensor([[2.0, 2.0]]))
        assert abs(metric.compute().item() - 2.0) < 1e-6

    def test_bf16_inputs_compute_finite(self) -> None:
        """Bf16 inputs are accumulated in float32 and produce a finite value."""
        pred = torch.randn(4, 92).bfloat16()
        target = torch.randn(4, 92).bfloat16()
        metric = BestSwapParamMSE()
        metric.update(pred, target)
        assert torch.isfinite(metric.compute())

    def test_mismatched_shapes_raise_value_error(self) -> None:
        """The shape guard rejects silently-broadcastable mismatches."""
        metric = BestSwapParamMSE()
        with pytest.raises(ValueError, match="matching 2-D shapes"):
            metric.update(torch.zeros(2, 4), torch.zeros(2, 5))

    def test_non_2d_inputs_raise_value_error(self) -> None:
        """1-D and 3-D inputs are rejected rather than reinterpreted."""
        metric = BestSwapParamMSE()
        with pytest.raises(ValueError, match="matching 2-D shapes"):
            metric.update(torch.zeros(4), torch.zeros(4))
