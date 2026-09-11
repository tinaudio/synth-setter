"""Regression tests for deterministic synthetic sample seeds."""

from synth_setter.data.sample_seed import derive_sample_seed


def test_derive_sample_seed_known_inputs_returns_stable_seed() -> None:
    """Pin the cross-platform mapping used by persisted experiments."""
    assert derive_sample_seed(123, 456) == 9_373_028_057_125_325_568


def test_derive_sample_seed_affine_collision_pair_returns_distinct_seeds() -> None:
    """Separate a crafted base/index collision from the former affine mapping."""
    first_seed = derive_sample_seed(0, 0)
    second_seed = derive_sample_seed(7_682_673_210_995_763_517, 1)

    assert first_seed != second_seed
