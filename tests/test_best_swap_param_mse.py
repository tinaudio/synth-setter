"""Behavioral tests for :class:`synth_setter.metrics.BestSwapParamMSE`.

The metric is the optimistic bracket to plain ``param_mse``: the MSE after the
error-minimizing one-to-one matching of predicted scalars to target scalars,
which for squared error is sort-both-and-compare (rearrangement inequality).
"""

from __future__ import annotations

import pytest
import torch

from synth_setter.metrics import BestSwapParamMSE


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Plain elementwise MSE reference.

    :param pred: Predicted vectors.
    :param target: Target vectors.
    :returns: Mean squared error as a float.
    """
    return (pred - target).square().mean().item()


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
