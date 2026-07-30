"""Unit coverage for the param_shift embedder's assignment and shift arithmetic."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import ParamSpec
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.data.vst.seeding import rng_for_sample
from synth_setter.pipeline.data.param_shift import (
    assigned_param_index,
    shift_encoded_row,
    shift_rng,
)

_SPEC_NAME = ParamSpecName("torchsynth_adsr")


@pytest.fixture(name="spec")
def _spec() -> ParamSpec:
    """Resolve the small real ADSR spec every case shifts against.

    :returns: Registered torchsynth ADSR param spec.
    """
    return resolve_param_spec(_SPEC_NAME)


def _encoded_row(spec: ParamSpec, seed: int) -> np.ndarray:
    """Draw one real encoded row from the spec's own sampler.

    :param spec: Param spec to sample.
    :param seed: Seed making the draw reproducible.
    :returns: Encoded row of width ``spec.encoded_width``.
    """
    synth_params, note_params = spec.sample(np.random.default_rng(seed))
    return spec.encode(synth_params, note_params)


def test_assigned_param_index_over_a_fragments_row_ids_is_balanced() -> None:
    """Consecutive row ids give every parameter the same share, up to one row."""
    num_params = 7

    counts = Counter(assigned_param_index(row_id, num_params) for row_id in range(100))

    assert set(counts) == set(range(num_params))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_assigned_param_index_zero_params_raises() -> None:
    """An empty spec is a configuration error, not a modulo-by-zero crash."""
    with pytest.raises(ValueError, match="at least one parameter"):
        assigned_param_index(0, 0)


def test_shift_encoded_row_changes_only_the_selected_parameters_span(spec: ParamSpec) -> None:
    """The redrawn parameter's columns move and every other column is untouched.

    :param spec: Registered torchsynth ADSR param spec.
    """
    row = _encoded_row(spec, seed=11)
    param_index = 2
    _, span = list(spec.encoded_slices())[param_index]

    shifted = shift_encoded_row(
        row, spec, param_index=param_index, rng=np.random.default_rng(5)
    )

    untouched = np.ones(spec.encoded_width, dtype=bool)
    untouched[span] = False
    assert np.array_equal(shifted.encoded[untouched], row[untouched])
    assert shifted.encoded.shape == row.shape


def test_shift_encoded_row_reports_the_selected_parameters_name(spec: ParamSpec) -> None:
    """The recorded name identifies the parameter that actually moved.

    :param spec: Registered torchsynth ADSR param spec.
    """
    row = _encoded_row(spec, seed=12)
    param_index = 3
    param, _ = list(spec.encoded_slices())[param_index]

    shifted = shift_encoded_row(
        row, spec, param_index=param_index, rng=np.random.default_rng(6)
    )

    assert shifted.param_name == param.name


def test_shift_encoded_row_amount_is_the_encoded_span_distance(spec: ParamSpec) -> None:
    """The reported amount equals the Euclidean move over the parameter's own columns.

    :param spec: Registered torchsynth ADSR param spec.
    """
    row = _encoded_row(spec, seed=13)
    param_index = 1
    _, span = list(spec.encoded_slices())[param_index]

    shifted = shift_encoded_row(
        row, spec, param_index=param_index, rng=np.random.default_rng(7)
    )

    expected = float(np.linalg.norm(shifted.encoded[span] - row[span]))
    assert shifted.amount == pytest.approx(expected)


def test_shift_encoded_row_stays_inside_the_encoded_domain(spec: ParamSpec) -> None:
    """A redrawn value re-encodes into ``[0, 1]`` so the row remains decodable.

    :param spec: Registered torchsynth ADSR param spec.
    """
    row = _encoded_row(spec, seed=14)

    for param_index in range(len(spec.names)):
        shifted = shift_encoded_row(
            row, spec, param_index=param_index, rng=np.random.default_rng(param_index)
        )

        assert np.all((shifted.encoded >= 0.0) & (shifted.encoded <= 1.0))
        spec.decode(shifted.encoded)


def test_shift_encoded_row_same_seed_reproduces_the_same_shift(spec: ParamSpec) -> None:
    """Re-running a row — as a resume replay does — redraws the identical value.

    :param spec: Registered torchsynth ADSR param spec.
    """
    row = _encoded_row(spec, seed=15)

    first = shift_encoded_row(row, spec, param_index=0, rng=np.random.default_rng(99))
    second = shift_encoded_row(row, spec, param_index=0, rng=np.random.default_rng(99))

    assert np.array_equal(first.encoded, second.encoded)
    assert first.amount == second.amount


def test_shift_encoded_row_wrong_width_raises(spec: ParamSpec) -> None:
    """A row encoded against another spec is rejected before it can be rendered.

    :param spec: Registered torchsynth ADSR param spec.
    """
    with pytest.raises(ValueError, match="expected"):
        shift_encoded_row(
            np.zeros(spec.encoded_width + 1, dtype=np.float32),
            spec,
            param_index=0,
            rng=np.random.default_rng(0),
        )


def test_shift_stream_is_independent_of_the_datagen_stream(spec: ParamSpec) -> None:
    """Reusing the dataset's own master seed must not echo each row's parameters back.

    Datagen draws a row from ``rng_for_sample(base_seed, sample_idx, attempt)``. If the
    shift drew from that same stream, its first value would be the row's first parameter —
    making every replacement a function of the patch it is meant to perturb, and a no-op
    whenever the assigned parameter happens to be that first one.

    :param spec: Registered torchsynth ADSR param spec.
    """
    master = 42
    spans = list(spec.encoded_slices())
    first_param_span = spans[0][1]

    echoes = 0
    for row_id in range(64):
        synth_params, note_params = spec.sample(rng_for_sample(master, row_id, 0))
        encoded = spec.encode(synth_params, note_params)
        param_index = assigned_param_index(row_id, len(spec.names))
        shifted = shift_encoded_row(
            encoded, spec, param_index=param_index, rng=shift_rng(master, row_id)
        )
        replacement = shifted.encoded[spans[param_index][1]][0]
        if replacement == pytest.approx(encoded[first_param_span][0], abs=1e-9):
            echoes += 1

    assert echoes == 0


def test_shift_rng_is_reproducible_for_a_row(spec: ParamSpec) -> None:
    """A rerun or resume-cache replay redraws the identical replacement.

    :param spec: Registered torchsynth ADSR param spec.
    """
    row = _encoded_row(spec, seed=21)

    first = shift_encoded_row(row, spec, param_index=1, rng=shift_rng(7, 99))
    second = shift_encoded_row(row, spec, param_index=1, rng=shift_rng(7, 99))

    assert np.array_equal(first.encoded, second.encoded)


def test_shift_rng_differs_across_rows_and_seeds(spec: ParamSpec) -> None:
    """Distinct rows, and distinct run seeds, draw distinct replacements.

    :param spec: Registered torchsynth ADSR param spec.
    """
    row = _encoded_row(spec, seed=22)
    baseline = shift_encoded_row(row, spec, param_index=1, rng=shift_rng(7, 99)).amount

    assert shift_encoded_row(row, spec, param_index=1, rng=shift_rng(7, 100)).amount != baseline
    assert shift_encoded_row(row, spec, param_index=1, rng=shift_rng(8, 99)).amount != baseline
