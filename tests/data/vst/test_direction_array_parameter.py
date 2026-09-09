"""Contracts for the unit-sphere direction array parameter."""

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import (
    ContinuousParameter,
    DirectionArrayParameter,
    ParamSpec,
)


def test_direction_array_owns_one_encoded_column_per_component() -> None:
    """A direction is stored component-wise, so the width equals the native length."""
    parameter = DirectionArrayParameter(name="axis", shape=(3,))

    assert len(parameter) == 3
    assert parameter.encoded_names() == ("axis.0", "axis.1", "axis.2")


def test_direction_array_sample_draws_unit_vectors_over_the_sphere() -> None:
    """Samples are unit-norm float64 vectors whose directions cover the sphere."""
    parameter = DirectionArrayParameter(name="axis", shape=(8,))

    samples = np.stack([parameter.sample(np.random.default_rng(seed)) for seed in range(512)])

    assert samples.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(samples, axis=-1), 1.0, atol=1e-12)
    assert samples.min() < -0.8
    assert samples.max() > 0.8
    assert np.abs(samples.mean(axis=0)).max() < 0.1


def test_direction_array_encode_normalises_scale_and_maps_into_unit_interval() -> None:
    """Only direction is stored: ``3u`` encodes exactly like ``u``, as ``(u/|u| + 1) / 2``."""
    parameter = DirectionArrayParameter(name="axis", shape=(2,))

    encoded = parameter.encode(np.array([3.0, 0.0]))

    assert encoded.dtype == np.float32
    np.testing.assert_allclose(encoded, [1.0, 0.5])
    np.testing.assert_array_equal(encoded, parameter.encode(np.array([0.5, 0.0])))


def test_direction_array_roundtrip_recovers_unit_direction() -> None:
    """Decoding an encoded row returns the normalised direction to float32 precision."""
    parameter = DirectionArrayParameter(name="axis", shape=(4,))
    vector = np.array([0.3, -0.7, 0.1, 0.9])

    decoded = parameter.decode(parameter.encode(vector))

    assert decoded.dtype == np.float64
    np.testing.assert_allclose(decoded, vector / np.linalg.norm(vector), atol=2e-7)


def test_direction_array_decode_projects_off_sphere_vectors() -> None:
    """A stored vector with the wrong norm decodes to its direction on the unit sphere."""
    parameter = DirectionArrayParameter(name="axis", shape=(2,))
    # Encoded (0.65, 0.65) is model-space (0.3, 0.3): norm 0.42, direction (1, 1)/sqrt 2.
    decoded = parameter.decode(np.array([0.65, 0.65]))

    np.testing.assert_allclose(decoded, [2**-0.5, 2**-0.5], atol=1e-7)


def test_direction_array_decode_zero_vector_falls_back_to_first_axis() -> None:
    """A directionless vector decodes deterministically to the first basis vector."""
    parameter = DirectionArrayParameter(name="axis", shape=(3,))

    decoded = parameter.decode(np.array([0.5, 0.5, 0.5]))

    assert decoded.tolist() == [1.0, 0.0, 0.0]


def test_direction_array_model_to_encoded_normalises_whole_vector_instead_of_clipping() -> None:
    """An overshooting prediction keeps its direction: (1.4, 0.2) decodes along (1.4, 0.2)."""
    parameter = DirectionArrayParameter(name="axis", shape=(2,))
    expected = np.array([1.4, 0.2]) / np.linalg.norm([1.4, 0.2])

    decoded = parameter.decode(parameter.model_to_encoded(np.array([1.4, 0.2])))

    np.testing.assert_allclose(decoded, expected, atol=1e-7)


def test_direction_array_model_to_encoded_zero_vector_passes_through() -> None:
    """A (0, 0) prediction is passed through so decode applies its own fallback."""
    parameter = DirectionArrayParameter(name="axis", shape=(2,))

    np.testing.assert_allclose(parameter.model_to_encoded(np.zeros(2)), [0.5, 0.5])


def test_param_spec_model_to_encoded_clips_scalars_but_projects_directions() -> None:
    """A full-width row saturates the scalar and projects the direction as a whole."""
    spec = ParamSpec(
        synth_params=[
            ContinuousParameter(name="gain", min=0.0, max=10.0),
            DirectionArrayParameter(name="axis", shape=(2,)),
        ],
        note_params=[],
    )

    decoded, _ = spec.decode(spec.model_to_encoded(np.array([1.5, 1.4, 0.2], dtype=np.float32)))

    assert decoded["gain"] == 10.0
    np.testing.assert_allclose(
        decoded["axis"], np.array([1.4, 0.2]) / np.linalg.norm([1.4, 0.2]), atol=1e-6
    )


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_direction_array_encode_rejects_nonfinite_component(value: float) -> None:
    """Non-finite components have no direction.

    :param value: Non-finite native component.
    """
    parameter = DirectionArrayParameter(name="axis", shape=(2,))

    with pytest.raises(ValueError, match="finite"):
        parameter.encode(np.array([value, 0.0]))


def test_direction_array_encode_rejects_vanishing_norm() -> None:
    """A native zero vector cannot be encoded because it carries no direction."""
    parameter = DirectionArrayParameter(name="axis", shape=(2,))

    with pytest.raises(ValueError, match="norm"):
        parameter.encode(np.zeros(2))


def test_direction_array_encode_rejects_wrong_native_shape() -> None:
    """The native array must match the declared shape exactly."""
    parameter = DirectionArrayParameter(name="axis", shape=(2,))

    with pytest.raises(ValueError, match="shape"):
        parameter.encode(np.zeros(3))


def test_direction_array_decode_rejects_wrong_encoded_width() -> None:
    """Decode consumes exactly one column per component."""
    parameter = DirectionArrayParameter(name="axis", shape=(2,))

    with pytest.raises(ValueError, match="shape"):
        parameter.decode(np.array([0.5, 0.5, 0.5]))


def test_direction_array_rejects_single_component_or_empty_shape() -> None:
    """A direction needs at least two components."""
    with pytest.raises(ValueError, match="shape"):
        DirectionArrayParameter(name="axis", shape=(1,))
