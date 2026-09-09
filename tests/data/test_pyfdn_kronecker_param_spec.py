"""Contracts for the Kronecker parametric-feedback pyFDN parameter distribution."""

from typing import cast

import numpy as np
import pytest

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_N8_MONO_KRONECKER_PARAM_SPEC,
    kronecker_feedback_matrix,
)
from synth_setter.data.vst.param_spec import (
    AngleArrayParameter,
    DiscreteArrayParameter,
)

_SCALE = 8**-0.5
_EXPECTED_HADAMARD_8 = _SCALE * np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1],
    ],
    dtype=np.float64,
)


def test_kronecker_spec_layout_appends_three_angle_pairs_and_three_reflect_flags() -> None:
    """The learned row carries three (cos, sin) pairs and three flags after the FDN fields."""
    layout = [
        (parameter.name, span.start, span.stop)
        for parameter, span in PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.encoded_slices()
    ]

    assert PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.encoded_width == 36
    assert layout == [
        ("delays", 0, 8),
        ("input_matrix", 8, 16),
        ("output_matrix", 16, 24),
        ("direct_matrix", 24, 25),
        ("post_delay.rt_dc_seconds", 25, 26),
        ("post_delay.rt_nyquist_seconds", 26, 27),
        ("kronecker_angles", 27, 33),
        ("kronecker_reflect", 33, 36),
    ]


def test_kronecker_spec_kernel_controls_have_exact_domains() -> None:
    """Angles are unit-circle pairs, one per level, and reflect flags are binary."""
    angles, reflect = PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.synth_params[-2:]

    assert isinstance(angles, AngleArrayParameter)
    assert angles.shape == (3,)
    assert isinstance(reflect, DiscreteArrayParameter)
    assert (reflect.shape, reflect.min, reflect.max) == ((3,), 0, 1)


def test_kronecker_feedback_matrix_all_reflect_quarter_pi_is_hadamard_8() -> None:
    """Three quarter-turn reflection kernels recover the Sylvester Hadamard matrix."""
    feedback = kronecker_feedback_matrix(
        np.full(3, np.pi / 4), np.ones(3, dtype=np.int64)
    )

    np.testing.assert_allclose(feedback, _EXPECTED_HADAMARD_8, atol=1e-15)


def test_kronecker_feedback_matrix_all_rotate_zero_is_identity() -> None:
    """Zero-angle rotation kernels leave every delay line isolated."""
    feedback = kronecker_feedback_matrix(np.zeros(3), np.zeros(3, dtype=np.int64))

    np.testing.assert_array_equal(feedback, np.eye(8))


def test_kronecker_feedback_matrix_zero_outer_angle_decouples_halves() -> None:
    """A zero outermost kernel splits lines 0-3 from lines 4-7 (paper section 5.2)."""
    feedback = kronecker_feedback_matrix(
        np.array([0.3, 0.7, 0.0]), np.array([1, 0, 0], dtype=np.int64)
    )

    np.testing.assert_array_equal(feedback[:4, 4:], np.zeros((4, 4)))
    np.testing.assert_array_equal(feedback[4:, :4], np.zeros((4, 4)))
    assert np.count_nonzero(feedback[:4, :4]) == 16


def test_kronecker_feedback_matrix_zero_inner_angle_decouples_even_and_odd_lines() -> None:
    """A zero innermost kernel splits even-indexed from odd-indexed lines (section 5.3)."""
    feedback = kronecker_feedback_matrix(
        np.array([0.0, 0.7, 0.3]), np.array([0, 1, 0], dtype=np.int64)
    )

    rows, cols = np.indices((8, 8))
    np.testing.assert_array_equal(feedback[(rows + cols) % 2 == 1], 0.0)
    assert np.count_nonzero(feedback[(rows + cols) % 2 == 0]) == 32


def test_kronecker_feedback_matrix_reflection_differs_from_rotation_at_same_angle() -> None:
    """Reflect flags select a genuinely different matrix, not a reparameterisation."""
    angles = np.array([0.4, 1.1, -0.9])

    rotated = kronecker_feedback_matrix(angles, np.zeros(3, dtype=np.int64))
    reflected = kronecker_feedback_matrix(angles, np.array([0, 1, 0], dtype=np.int64))

    assert not np.allclose(rotated, reflected)


def test_kronecker_spec_sampled_feedback_is_orthogonal_and_seed_dependent() -> None:
    """Every sampled patch is lossless by construction yet varies with the seed."""
    first, _ = PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.sample(np.random.default_rng(1))
    second, _ = PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.sample(np.random.default_rng(2))
    first_feedback = cast(np.ndarray, first["feedback_matrix"])
    second_feedback = cast(np.ndarray, second["feedback_matrix"])

    assert (first_feedback.shape, first_feedback.dtype) == ((8, 8), np.dtype(np.float64))
    np.testing.assert_allclose(first_feedback.T @ first_feedback, np.eye(8), atol=1e-14)
    assert not np.allclose(first_feedback, second_feedback)


def test_kronecker_spec_sampled_feedback_matches_its_own_kernel_controls() -> None:
    """The stored matrix is exactly the one the sampled angles and flags describe."""
    params, _ = PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.sample(np.random.default_rng(7))

    np.testing.assert_array_equal(
        params["feedback_matrix"],
        kronecker_feedback_matrix(
            cast(np.ndarray, params["kronecker_angles"]),
            cast(np.ndarray, params["kronecker_reflect"]),
        ),
    )


def test_kronecker_spec_encoding_round_trips_feedback_through_kernel_controls() -> None:
    """Decoding a row rebuilds the feedback matrix from the six learned coordinates."""
    params, notes = PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.sample(np.random.default_rng(123))

    encoded = PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.encode(params, notes)
    decoded, decoded_notes = PYFDN_N8_MONO_KRONECKER_PARAM_SPEC.decode(encoded)

    assert (encoded.shape, encoded.dtype) == ((36,), np.dtype(np.float32))
    np.testing.assert_array_equal(decoded["kronecker_reflect"], params["kronecker_reflect"])
    np.testing.assert_allclose(decoded["kronecker_angles"], params["kronecker_angles"], atol=1e-6)
    np.testing.assert_allclose(decoded["feedback_matrix"], params["feedback_matrix"], atol=1e-6)
    assert decoded_notes == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}


@pytest.mark.parametrize("levels", [2, 4])
def test_kronecker_feedback_matrix_wrong_control_length_raises(levels: int) -> None:
    """Only exactly three kernels describe an order-8 matrix; other counts are rejected.

    :param levels: Number of kernel controls supplied instead of three.
    """
    with pytest.raises(ValueError, match="3"):
        kronecker_feedback_matrix(np.zeros(levels), np.zeros(levels, dtype=np.int64))
