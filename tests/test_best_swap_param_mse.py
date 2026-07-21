"""Behavioral tests for :class:`synth_setter.metrics.BestSwapParamMSE`.

The metric is the optimistic bracket to plain ``param_mse``: the MSE after the
error-minimizing one-to-one matching of predicted scalars to target scalars,
which for squared error is sort-both-and-compare (rearrangement inequality).
"""

from __future__ import annotations

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
        """A prediction equal to the target scores exactly zero."""
        params = torch.tensor([[0.3, -0.7, 0.1, 0.9]])
        metric = BestSwapParamMSE()
        metric.update(params, params)
        assert metric.compute().item() == 0.0

    def test_permuted_target_scores_zero_while_mse_is_large(self) -> None:
        """Any reordering of the target is free for the metric but costly for MSE."""
        pred = torch.tensor([[0.9, -0.5, 0.2, 0.0]])
        target = pred[:, torch.tensor([2, 0, 3, 1])]
        metric = BestSwapParamMSE()
        metric.update(pred, target)
        assert metric.compute().item() < 1e-12
        assert _mse(pred, target) > 0.1

    def test_value_error_is_still_penalized(self) -> None:
        """Wrong values cannot be fixed by swapping and are fully penalized."""
        pred = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[1.0, 1.0]])
        metric = BestSwapParamMSE()
        metric.update(pred, target)
        assert abs(metric.compute().item() - 1.0) < 1e-6

    def test_never_exceeds_plain_mse(self) -> None:
        """Random inputs: the swap-optimal error is a floor under the plain MSE."""
        generator = torch.Generator().manual_seed(7)
        pred = torch.randn(16, 92, generator=generator)
        target = torch.randn(16, 92, generator=generator)
        metric = BestSwapParamMSE()
        metric.update(pred, target)
        assert metric.compute().item() <= _mse(pred, target) + 1e-6

    def test_matching_is_per_sample_not_cross_batch(self) -> None:
        """Scalars are matched within each sample, never across batch rows."""
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
