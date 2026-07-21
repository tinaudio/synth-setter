"""Behavior tests for the partition-aware linear-assignment distance metric.

Covers the pure partition-derivation helper (numbered-prefix grouping over real
param-spec names) and ``PartitionedLinearAssignmentDistance`` itself: block
permutations of interchangeable groups score ~0 while order-fixed parameters
keep their plain squared error.
"""

from __future__ import annotations

import pytest
import torch

from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.metrics import (
    PartitionedLinearAssignmentDistance,
    derive_interchangeable_groups,
)
from synth_setter.param_spec_name import ParamSpecName


def _surge_simple_names() -> list[str]:
    """Return the surge_simple encoded param names (synth then note order).

    :returns: Registry spec names in encoded-vector order.
    """
    return resolve_param_spec(ParamSpecName("surge_simple")).names


class TestDeriveInterchangeableGroups:
    """Partition derivation from spec names."""

    def test_derive_groups_surge_simple_spec_finds_osc_filter_lfo_families(self) -> None:
        """surge_simple yields the osc/filter/lfo families with their block shapes."""
        names = _surge_simple_names()

        groups = derive_interchangeable_groups(names)

        block_shapes = sorted((len(group), len(group[0])) for group in groups)
        # lfo_6 lacks phase/rate, so only lfo_1..5 group.
        assert block_shapes == [(2, 3), (3, 7), (5, 9)]

    def test_derive_groups_surge_simple_spec_aligns_blocks_by_suffix(self) -> None:
        """Position ``i`` addresses the same suffix in every block of a group."""
        names = _surge_simple_names()

        groups = derive_interchangeable_groups(names)

        osc_group = next(g for g in groups if len(g) == 3)
        position = osc_group[0].index(names.index("a_osc_1_pitch"))
        assert osc_group[1][position] == names.index("a_osc_2_pitch")
        assert osc_group[2][position] == names.index("a_osc_3_pitch")

    def test_derive_groups_surge_simple_spec_excludes_lfo_6_and_note_params(self) -> None:
        """Layout-mismatched lfo_6 and unnumbered note params stay order-fixed."""
        names = _surge_simple_names()

        grouped = {
            index
            for group in derive_interchangeable_groups(names)
            for block in group
            for index in block
        }

        assert names.index("a_lfo_6_amplitude") not in grouped
        assert names.index("pitch") not in grouped
        assert names.index("note_start_and_end") not in grouped

    def test_derive_groups_surge_4_spec_has_no_interchangeable_blocks(self) -> None:
        """surge_4's singleton params derive an empty partition."""
        names = resolve_param_spec(ParamSpecName("surge_4")).names

        assert derive_interchangeable_groups(names) == []

    def test_derive_groups_singleton_numbered_family_returns_no_group(self) -> None:
        """A family with one block is not interchangeable with anything."""
        assert derive_interchangeable_groups(["osc_1_a", "osc_1_b", "cutoff"]) == []

    def test_derive_groups_mismatched_suffix_block_excluded_from_group(self) -> None:
        """Only blocks sharing the exact suffix layout group together."""
        names = ["f_1_a", "f_1_b", "f_2_a", "f_2_b", "f_3_a"]

        groups = derive_interchangeable_groups(names)

        assert groups == [[[0, 1], [2, 3]]]

    def test_derive_groups_onehot_widths_expand_to_encoded_indices(self) -> None:
        """Multi-index (one-hot) params expand to their full encoded index span."""
        names = ["f_1_a", "f_1_b", "f_2_a", "f_2_b"]
        widths = [1, 2, 1, 2]

        groups = derive_interchangeable_groups(names, widths)

        assert groups == [[[0, 1, 2], [3, 4, 5]]]


class TestPartitionedLinearAssignmentDistance:
    """Permutation-optimal MSE semantics of the metric."""

    def _two_block_metric(self) -> PartitionedLinearAssignmentDistance:
        """Build a 6-param metric with one interchangeable 2-block group.

        :returns: Metric instance; the last two params stay order-fixed.
        """
        return PartitionedLinearAssignmentDistance(groups=[[[0, 1], [2, 3]]], num_params=6)

    def test_update_block_swapped_prediction_scores_zero_while_mse_large(self) -> None:
        """Swapping interchangeable blocks costs nothing despite a large plain MSE."""
        metric = self._two_block_metric()
        target = torch.tensor([[1.0, 2.0, -1.0, -2.0, 0.5, 0.5]])
        prediction = torch.tensor([[-1.0, -2.0, 1.0, 2.0, 0.5, 0.5]])

        metric.update(prediction, target)

        assert (prediction - target).square().mean() > 1.0
        assert metric.compute() == pytest.approx(0.0)

    def test_update_error_on_fixed_param_equals_plain_mse(self) -> None:
        """Errors outside any group are penalized exactly like plain MSE."""
        metric = self._two_block_metric()
        target = torch.tensor([[1.0, 2.0, -1.0, -2.0, 0.0, 0.0]])
        prediction = torch.tensor([[1.0, 2.0, -1.0, -2.0, 3.0, 0.0]])

        metric.update(prediction, target)

        assert metric.compute() == pytest.approx(9.0 / 6.0)

    def test_update_never_exceeds_plain_mse_on_random_inputs(self) -> None:
        """The optimal matching never scores above the identity matching."""
        metric = self._two_block_metric()
        generator = torch.Generator().manual_seed(7)
        target = torch.randn(8, 6, generator=generator)
        prediction = torch.randn(8, 6, generator=generator)

        metric.update(prediction, target)

        assert metric.compute() <= (prediction - target).square().mean() + 1e-6

    def test_update_no_groups_matches_plain_mse(self) -> None:
        """An empty partition degenerates to plain elementwise MSE."""
        metric = PartitionedLinearAssignmentDistance(groups=[], num_params=3)
        target = torch.tensor([[0.0, 0.0, 0.0]])
        prediction = torch.tensor([[1.0, 2.0, 2.0]])

        metric.update(prediction, target)

        assert metric.compute() == pytest.approx(3.0)

    def test_compute_accumulates_element_mean_over_updates(self) -> None:
        """``compute`` averages squared error over all elements seen so far."""
        metric = PartitionedLinearAssignmentDistance(groups=[], num_params=2)

        metric.update(torch.tensor([[1.0, 1.0]]), torch.zeros(1, 2))
        metric.update(torch.tensor([[0.0, 0.0]]), torch.zeros(1, 2))

        assert metric.compute() == pytest.approx(0.5)

    def test_update_matches_blocks_independently_per_sample(self) -> None:
        """A batch mixing a swapped and an identity sample still scores zero."""
        metric = self._two_block_metric()
        target = torch.tensor(
            [
                [1.0, 2.0, -1.0, -2.0, 0.0, 0.0],
                [1.0, 2.0, -1.0, -2.0, 0.0, 0.0],
            ]
        )
        prediction = torch.tensor(
            [
                [-1.0, -2.0, 1.0, 2.0, 0.0, 0.0],
                [1.0, 2.0, -1.0, -2.0, 0.0, 0.0],
            ]
        )

        metric.update(prediction, target)

        assert metric.compute() == pytest.approx(0.0)

    def test_update_wrong_width_raises_value_error(self) -> None:
        """A width mismatch against ``num_params`` is rejected."""
        metric = self._two_block_metric()

        with pytest.raises(ValueError, match="width"):
            metric.update(torch.zeros(1, 5), torch.zeros(1, 5))

    def test_update_mismatched_shapes_raise_value_error(self) -> None:
        """Unequal prediction/target shapes are rejected."""
        metric = self._two_block_metric()

        with pytest.raises(ValueError, match="shape"):
            metric.update(torch.zeros(2, 6), torch.zeros(1, 6))

    def test_init_duplicate_index_within_block_raises_value_error(self) -> None:
        """A block repeating an index is rejected at construction."""
        with pytest.raises(ValueError, match="repeats"):
            PartitionedLinearAssignmentDistance(groups=[[[0, 0], [1, 2]]], num_params=4)

    def test_init_overlapping_group_indices_raises_value_error(self) -> None:
        """Overlapping block indices are rejected at construction."""
        with pytest.raises(ValueError, match="overlap"):
            PartitionedLinearAssignmentDistance(groups=[[[0, 1], [1, 2]]], num_params=4)

    def test_init_out_of_range_index_raises_value_error(self) -> None:
        """Indices beyond ``num_params`` are rejected at construction."""
        with pytest.raises(ValueError, match="range"):
            PartitionedLinearAssignmentDistance(groups=[[[0], [7]]], num_params=4)

    def test_init_unequal_block_sizes_raises_value_error(self) -> None:
        """Blocks of unequal size within a group are rejected at construction."""
        with pytest.raises(ValueError, match="size"):
            PartitionedLinearAssignmentDistance(groups=[[[0, 1], [2]]], num_params=4)
