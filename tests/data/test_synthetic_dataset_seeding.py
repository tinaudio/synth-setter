"""Regression tests for deterministic online synthetic dataset seeding."""

from collections.abc import Callable
from typing import TypeAlias

import pytest
import torch

from synth_setter.data.kosc_datamodule import KOscDataset
from synth_setter.data.ksin_datamodule import KSinDataset
from synth_setter.data.sample_seed import derive_sample_seed

DatasetFactory: TypeAlias = Callable[[int], KSinDataset | KOscDataset]


def _ksin_dataset(seed: int) -> KSinDataset:
    """Build a small deterministic KSin dataset for seeding tests.

    :param seed: Dataset-level RNG seed.
    :returns: Three-row dataset using the shared minimal audio geometry.
    """
    return KSinDataset(
        k=2,
        signal_length=16,
        num_samples=3,
        sort_frequencies=False,
        break_symmetry=False,
        shift_test_distribution=False,
        is_test=False,
        seed=seed,
    )


def _kosc_dataset(seed: int) -> KOscDataset:
    """Build a small deterministic KOsc dataset for seeding tests.

    :param seed: Dataset-level RNG seed.
    :returns: Three-row dataset using the shared minimal audio geometry.
    """
    return KOscDataset(
        k=2,
        signal_length=16,
        num_samples=3,
        sort_frequencies=False,
        break_symmetry=False,
        is_test=False,
        seed=seed,
    )


@pytest.mark.parametrize("dataset_factory", [_ksin_dataset, _kosc_dataset])
def test_dataset_different_split_seeds_index_zero_produces_distinct_params(
    dataset_factory: DatasetFactory,
) -> None:
    """Protect validation metrics from index-zero train/validation leakage.

    :param dataset_factory: Online dataset constructor under test.
    """
    first_params = dataset_factory(100)[0][1]
    second_params = dataset_factory(200)[0][1]

    assert not torch.equal(first_params, second_params)


@pytest.mark.parametrize("dataset_factory", [_ksin_dataset, _kosc_dataset])
def test_dataset_multiplicative_seed_pair_produces_distinct_params(
    dataset_factory: DatasetFactory,
) -> None:
    """Protect validation metrics when configured split seeds share factors.

    :param dataset_factory: Online dataset constructor under test.
    """
    first_params = dataset_factory(100)[2][1]
    second_params = dataset_factory(200)[1][1]

    assert not torch.equal(first_params, second_params)


@pytest.mark.parametrize("dataset_factory", [_ksin_dataset, _kosc_dataset])
def test_dataset_same_seed_and_index_reproduces_params(
    dataset_factory: DatasetFactory,
) -> None:
    """Preserve reproducible rows across repeated dataset construction.

    :param dataset_factory: Online dataset constructor under test.
    """
    first_params = dataset_factory(100)[2][1]
    second_params = dataset_factory(100)[2][1]

    torch.testing.assert_close(first_params, second_params)


def test_derive_sample_seed_known_inputs_returns_stable_seed() -> None:
    """Pin the cross-platform seed mapping used by persisted experiments."""
    assert derive_sample_seed(123, 456) == 9_373_028_057_125_325_568


def test_derive_sample_seed_affine_collision_pair_returns_distinct_seeds() -> None:
    """Derive distinct seeds for a crafted base/index collision pair."""
    first_seed = derive_sample_seed(0, 0)
    second_seed = derive_sample_seed(7_682_673_210_995_763_517, 1)

    assert first_seed != second_seed
