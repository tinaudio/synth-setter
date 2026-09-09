"""Contracts for the unit-circle angle array parameter."""

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import AngleArrayParameter, ParamSpec


def test_angle_array_owns_two_encoded_columns_per_angle() -> None:
    """Each angle is carried as a (cos, sin) pair, so the encoded width doubles the shape."""
    parameter = AngleArrayParameter(name="phase", shape=(3,))

    assert len(parameter) == 6
    assert parameter.encoded_names() == (
        "phase.0.cos",
        "phase.0.sin",
        "phase.1.cos",
        "phase.1.sin",
        "phase.2.cos",
        "phase.2.sin",
    )


def test_angle_array_sample_draws_radians_over_one_full_turn() -> None:
    """Sampling covers the whole circle as float64 radians in the native shape."""
    parameter = AngleArrayParameter(name="phase", shape=(4096,))

    sampled = parameter.sample(np.random.default_rng(0))

    assert (sampled.shape, sampled.dtype) == ((4096,), np.dtype(np.float64))
    assert sampled.min() >= -np.pi
    assert sampled.max() < np.pi
    assert sampled.min() < -3.0
    assert sampled.max() > 3.0


def test_angle_array_encode_maps_cos_sin_into_unit_interval() -> None:
    """``theta`` encodes as ``((cos+1)/2, (sin+1)/2)`` so model space sees ``(cos, sin)``."""
    parameter = AngleArrayParameter(name="phase", shape=(2,))

    encoded = parameter.encode(np.array([0.0, np.pi / 2]))

    assert encoded.dtype == np.float32
    np.testing.assert_allclose(encoded, [1.0, 0.5, 0.5, 1.0], atol=1e-7)


def test_angle_array_encode_treats_plus_and_minus_pi_identically() -> None:
    """The seam at ±π disappears: both ends of the range share one encoding."""
    parameter = AngleArrayParameter(name="phase", shape=(1,))

    np.testing.assert_array_equal(
        parameter.encode(np.array([np.pi])), parameter.encode(np.array([-np.pi]))
    )


def test_angle_array_roundtrip_recovers_angles_within_float32_precision() -> None:
    """Decoding an encoded row returns the original radians in ``[-π, π]``."""
    parameter = AngleArrayParameter(name="phase", shape=(5,))
    angles = np.array([-3.0, -1.0, 0.0, 0.7, 2.9])

    decoded = parameter.decode(parameter.encode(angles))

    assert decoded.dtype == np.float64
    np.testing.assert_allclose(decoded, angles, atol=2e-7)


def test_angle_array_decode_projects_off_circle_pairs_onto_the_unit_circle() -> None:
    """A model pair with the wrong norm decodes to the angle of its direction."""
    parameter = AngleArrayParameter(name="phase", shape=(1,))
    # Encoded (0.65, 0.65) is model-space (0.3, 0.3): norm 0.42 but direction π/4.
    decoded = parameter.decode(np.array([0.65, 0.65]))

    np.testing.assert_allclose(decoded, [np.pi / 4], atol=1e-7)


def test_angle_array_decode_zero_pair_falls_back_to_zero_angle() -> None:
    """A directionless (0, 0) pair decodes deterministically to angle 0 instead of raising."""
    parameter = AngleArrayParameter(name="phase", shape=(1,))

    decoded = parameter.decode(np.array([0.5, 0.5]))

    assert decoded.tolist() == [0.0]


@pytest.mark.parametrize("bad", [np.array([0.5, 0.5, 0.5]), np.array([[0.5, 0.5]])])
def test_angle_array_decode_rejects_wrong_encoded_width(bad: np.ndarray) -> None:
    """Decode consumes exactly two columns per angle.

    :param bad: Encoded input whose width does not match the parameter.
    """
    parameter = AngleArrayParameter(name="phase", shape=(1,))

    with pytest.raises(ValueError, match="shape"):
        parameter.decode(bad)


def test_angle_array_decode_rejects_out_of_unit_interval_value() -> None:
    """Encoded values outside [0, 1] are a caller bug, not something to silently clip."""
    parameter = AngleArrayParameter(name="phase", shape=(1,))

    with pytest.raises(ValueError, match="within"):
        parameter.decode(np.array([1.5, 0.5]))


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_angle_array_encode_rejects_nonfinite_angle(value: float) -> None:
    """Non-finite radians cannot be placed on the circle.

    :param value: Non-finite native angle.
    """
    parameter = AngleArrayParameter(name="phase", shape=(1,))

    with pytest.raises(ValueError, match="finite"):
        parameter.encode(np.array([value]))


def test_angle_array_encode_rejects_wrong_native_shape() -> None:
    """The native array must match the declared shape exactly."""
    parameter = AngleArrayParameter(name="phase", shape=(2,))

    with pytest.raises(ValueError, match="shape"):
        parameter.encode(np.array([0.0, 0.0, 0.0]))


def test_angle_array_encode_wraps_angles_outside_the_principal_range() -> None:
    """Angles are periodic, so ``3π`` encodes exactly like ``π``."""
    parameter = AngleArrayParameter(name="phase", shape=(1,))

    np.testing.assert_allclose(
        parameter.encode(np.array([3 * np.pi])), parameter.encode(np.array([np.pi])), atol=1e-7
    )


def test_angle_array_rejects_empty_or_nonpositive_shape() -> None:
    """The shape must describe at least one angle."""
    with pytest.raises(ValueError, match="shape"):
        AngleArrayParameter(name="phase", shape=(0,))


def test_param_spec_model_space_pair_is_cos_sin_of_the_angle() -> None:
    """Through ParamSpec's [-1, 1] rescale the model regresses ``(cos θ, sin θ)`` directly."""
    spec = ParamSpec(synth_params=[AngleArrayParameter(name="phase", shape=(1,))], note_params=[])
    theta = 2.0

    model_space = spec.encoded_to_model(spec.encode({"phase": np.array([theta])}, {}))

    np.testing.assert_allclose(model_space, [np.cos(theta), np.sin(theta)], atol=1e-7)


def test_param_spec_seam_neighbours_are_close_in_model_space() -> None:
    """Angles just either side of ±π are neighbours in model space, not opposite extremes."""
    spec = ParamSpec(synth_params=[AngleArrayParameter(name="phase", shape=(1,))], note_params=[])

    left = spec.encoded_to_model(spec.encode({"phase": np.array([np.pi - 0.01])}, {}))
    right = spec.encoded_to_model(spec.encode({"phase": np.array([-np.pi + 0.01])}, {}))

    assert float(np.sum((left - right) ** 2)) < 1e-3
