"""Tests that parameter sampling draws only from an explicit, passed-in RNG (#884).

The reproducibility guarantee rests on ``ParamSpec.sample`` being a pure function
of the ``numpy`` ``Generator`` it is handed — no global ``np.random`` / ``random``
state may leak in or out, or determinism bleeds between samples under fork.
"""

import random

import numpy as np

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)


def _spec() -> ParamSpec:
    synth_params = [
        CategoricalParameter("osc_type", values=["sine", "saw", "square"]),
        ContinuousParameter("cutoff", 0.0, 1.0),
        ContinuousParameter("maybe_const", 0.1, 0.9, constant_val_p=0.5, constant_val=0.3),
        DiscreteLiteralParameter("steps", 1, 8),
    ]
    note_params = [
        DiscreteLiteralParameter("pitch", 48, 72),
        NoteDurationParameter("note_start_and_end", 2.0),
    ]
    return ParamSpec(synth_params, note_params)


def test_param_spec_sample_same_rng_returns_identical_params() -> None:
    a = _spec().sample(np.random.default_rng(12345))
    b = _spec().sample(np.random.default_rng(12345))
    assert a == b


def test_param_spec_sample_different_seed_returns_different_params() -> None:
    a = _spec().sample(np.random.default_rng(0))
    b = _spec().sample(np.random.default_rng(1))
    assert a != b


def test_param_spec_sample_does_not_touch_global_np_random() -> None:
    # If sampling drew from the global stream, seeding to 7 then sampling would
    # advance it and the next global draw would diverge from a fresh seed-7 stream.
    np.random.seed(7)
    witness = np.random.RandomState(7).randint(0, 2**31)
    _spec().sample(np.random.default_rng(0))
    assert np.random.randint(0, 2**31) == witness


def test_param_spec_sample_does_not_touch_global_stdlib_random() -> None:
    random.seed(7)
    witness = random.Random(7).random()
    _spec().sample(np.random.default_rng(0))
    assert random.random() == witness


def test_param_spec_sample_interleaved_is_order_independent() -> None:
    spec = _spec()
    standalone_a = spec.sample(np.random.default_rng(100))
    standalone_b = spec.sample(np.random.default_rng(200))

    rng_a = np.random.default_rng(100)
    rng_b = np.random.default_rng(200)
    interleaved_a = spec.sample(rng_a)
    interleaved_b = spec.sample(rng_b)

    assert interleaved_a == standalone_a
    assert interleaved_b == standalone_b


def test_param_spec_sample_independent_of_global_seed() -> None:
    np.random.seed(1)
    random.seed(1)
    first = _spec().sample(np.random.default_rng(99))
    np.random.seed(2)
    random.seed(2)
    second = _spec().sample(np.random.default_rng(99))
    assert first == second


def test_categorical_parameter_sample_is_reproducible_from_rng() -> None:
    param = CategoricalParameter("osc_type", values=["sine", "saw", "square"])
    assert param.sample(np.random.default_rng(3)) == param.sample(np.random.default_rng(3))


def test_discrete_literal_parameter_sample_is_reproducible_from_rng() -> None:
    param = DiscreteLiteralParameter("steps", 1, 8)
    assert param.sample(np.random.default_rng(3)) == param.sample(np.random.default_rng(3))


def test_discrete_literal_parameter_sample_returns_native_int() -> None:
    # A sampled pitch reaches mido/pedalboard's MIDI parser, which rejects numpy
    # scalars; the draw must be a native int, not np.int64 (#884 regression guard).
    value = DiscreteLiteralParameter("pitch", 48, 72).sample(np.random.default_rng(3))
    assert type(value) is int


def test_continuous_parameter_sample_is_reproducible_from_rng() -> None:
    param = ContinuousParameter("cutoff", 0.0, 1.0)
    assert param.sample(np.random.default_rng(3)) == param.sample(np.random.default_rng(3))


def test_continuous_parameter_constant_branch_returns_constant_val() -> None:
    # constant_val_p=1.0 always takes the constant branch (which consumes one rng.random
    # draw); pins that branch independently of the uniform-draw branch.
    param = ContinuousParameter("c", 0.1, 0.9, constant_val_p=1.0, constant_val=0.3)
    assert param.sample(np.random.default_rng(0)) == 0.3
    assert param.sample(np.random.default_rng(123)) == 0.3


def test_note_duration_parameter_sample_is_reproducible_from_rng() -> None:
    param = NoteDurationParameter("note_start_and_end", 2.0)
    assert param.sample(np.random.default_rng(3)) == param.sample(np.random.default_rng(3))


def test_param_spec_sample_without_rng_still_produces_valid_params() -> None:
    # Back-compat: a bare ``sample()`` (no rng) draws fresh, non-deterministic params.
    synth, note = _spec().sample()
    assert set(synth) == {"osc_type", "cutoff", "maybe_const", "steps"}
    assert set(note) == {"pitch", "note_start_and_end"}
