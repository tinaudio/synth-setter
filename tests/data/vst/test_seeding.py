"""Tests for deterministic per-sample seed derivation (#884)."""

import numpy as np
from synth_setter.data.vst.seeding import rng_for_sample, seed_for_sample


def test_seed_for_sample_same_inputs_returns_same_seed() -> None:
    assert seed_for_sample(42, 12345, 0) == seed_for_sample(42, 12345, 0)


def test_seed_for_sample_golden_value_is_stable() -> None:
    # Frozen: changing the hash/encoding silently invalidates every existing
    # dataset, so this constant is a load-bearing regression pin.
    assert seed_for_sample(0, 0, 0) == 16774267956234540618


def test_seed_for_sample_adjacent_indices_are_decorrelated() -> None:
    # Rules out a naive ``base_seed + idx`` scheme: adjacent rows must not map
    # to adjacent seeds.
    assert abs(seed_for_sample(0, 0, 0) - seed_for_sample(0, 1, 0)) > 1


def test_seed_for_sample_distinct_index_gives_distinct_seed() -> None:
    assert seed_for_sample(42, 0, 0) != seed_for_sample(42, 1, 0)


def test_seed_for_sample_distinct_attempt_gives_distinct_seed() -> None:
    assert seed_for_sample(42, 12345, 0) != seed_for_sample(42, 12345, 1)


def test_seed_for_sample_distinct_master_gives_distinct_seed() -> None:
    assert seed_for_sample(0, 12345, 0) != seed_for_sample(1, 12345, 0)


def test_seed_for_sample_in_uint64_domain() -> None:
    seed = seed_for_sample(2**31, 9_999_999, 7)
    assert 0 <= seed < 2**64


def test_seed_for_sample_default_attempt_is_zero() -> None:
    assert seed_for_sample(42, 12345) == seed_for_sample(42, 12345, 0)


def test_rng_for_sample_same_inputs_produce_identical_draw_sequence() -> None:
    a = rng_for_sample(42, 12345, 0).integers(0, 2**32, size=5)
    b = rng_for_sample(42, 12345, 0).integers(0, 2**32, size=5)
    assert np.array_equal(a, b)


def test_rng_for_sample_distinct_inputs_diverge() -> None:
    a = rng_for_sample(42, 12345, 0).integers(0, 2**32, size=5)
    b = rng_for_sample(42, 12346, 0).integers(0, 2**32, size=5)
    assert not np.array_equal(a, b)
