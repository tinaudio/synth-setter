"""Contracts for the learnable-Householder pyFDN parameter distribution."""

from typing import cast

import numpy as np
import pytest

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC,
    PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC,
    householder_feedback_matrix,
)
from synth_setter.data.vst.param_spec import DirectionArrayParameter


def test_householder_vector_spec_layout_appends_one_eight_vector() -> None:
    """The learned row carries the reflection vector after the shared FDN fields."""
    layout = [
        (parameter.name, span.start, span.stop)
        for parameter, span in PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.encoded_slices()
    ]

    assert PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.encoded_width == 35
    assert layout[-1] == ("householder_vector", 27, 35)


def test_householder_vector_spec_vector_is_a_unit_direction() -> None:
    """The reflection vector is a direction parameter, projected rather than clipped."""
    vector = PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.synth_params[-1]

    assert isinstance(vector, DirectionArrayParameter)
    assert vector.shape == (8,)


def test_householder_vector_spec_samples_unit_norm_vectors() -> None:
    """Sampled reflection vectors are unit length, so the target has no scale redundancy."""
    params, _ = PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.sample(np.random.default_rng(3))

    assert np.isclose(np.linalg.norm(cast(np.ndarray, params["householder_vector"])), 1.0)


def test_householder_feedback_matrix_all_ones_matches_fixed_householder_spec() -> None:
    """The all-ones vector reproduces the fixed spec's constant matrix exactly."""
    fixed, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(0))

    np.testing.assert_allclose(
        householder_feedback_matrix(np.ones(8)), fixed["feedback_matrix"], atol=1e-15
    )


def test_householder_feedback_matrix_is_scale_and_sign_invariant() -> None:
    """Only the reflection direction matters, so ``u``, ``-u`` and ``3u`` agree."""
    vector = np.array([0.3, -0.7, 0.1, 0.9, -0.2, 0.5, 0.0, -0.4])

    reference = householder_feedback_matrix(vector)

    np.testing.assert_allclose(householder_feedback_matrix(-vector), reference, atol=1e-15)
    np.testing.assert_allclose(householder_feedback_matrix(3 * vector), reference, atol=1e-15)


def test_householder_feedback_matrix_axis_vector_flips_one_line() -> None:
    """A single-axis vector negates that delay line and leaves the rest untouched."""
    feedback = householder_feedback_matrix(np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]))

    expected = np.eye(8)
    expected[3, 3] = -1.0
    np.testing.assert_allclose(feedback, expected, atol=1e-15)


def test_householder_feedback_matrix_near_zero_vector_raises() -> None:
    """A vanishing vector has no reflection direction, so it is rejected, not NaN."""
    with pytest.raises(ValueError, match="norm"):
        householder_feedback_matrix(np.full(8, 1e-9))


def test_householder_feedback_matrix_wrong_length_raises() -> None:
    """Only an order-8 vector describes an order-8 reflection."""
    with pytest.raises(ValueError, match="8"):
        householder_feedback_matrix(np.ones(4))


def test_householder_vector_spec_sampled_feedback_is_orthogonal_symmetric_and_seeded() -> None:
    """Every sampled patch is a lossless reflection that varies with the seed."""
    first, _ = PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.sample(np.random.default_rng(1))
    second, _ = PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.sample(np.random.default_rng(2))
    feedback = cast(np.ndarray, first["feedback_matrix"])

    assert (feedback.shape, feedback.dtype) == ((8, 8), np.dtype(np.float64))
    np.testing.assert_allclose(feedback.T @ feedback, np.eye(8), atol=1e-14)
    np.testing.assert_allclose(feedback, feedback.T, atol=1e-15)
    assert not np.allclose(feedback, second["feedback_matrix"])


def test_householder_vector_spec_encoding_round_trips_feedback_through_vector() -> None:
    """Decoding a row rebuilds the reflection from the eight learned coordinates."""
    params, notes = PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.sample(np.random.default_rng(123))

    encoded = PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.encode(params, notes)
    decoded, decoded_notes = PYFDN_N8_MONO_HOUSEHOLDER_VECTOR_PARAM_SPEC.decode(encoded)

    assert (encoded.shape, encoded.dtype) == ((35,), np.dtype(np.float32))
    np.testing.assert_allclose(decoded["householder_vector"], params["householder_vector"], atol=1e-6)
    np.testing.assert_allclose(decoded["feedback_matrix"], params["feedback_matrix"], atol=1e-5)
    assert decoded_notes == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
