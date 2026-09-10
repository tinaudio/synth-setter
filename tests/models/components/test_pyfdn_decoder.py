"""Behavior tests for tensor-native pyFDN parameter decoding."""

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from torch import Tensor

from synth_setter.data.pyfdn_param_spec import (
    PYFDN_FEEDBACK_GIVENS_ANGLES_NAME,
    PYFDN_FEEDBACK_SKEW_NAME,
    PYFDN_HOUSEHOLDER_VECTOR_NAME,
    PYFDN_KRONECKER_ANGLES_NAME,
    PYFDN_KRONECKER_REFLECT_NAME,
    PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME,
)
from synth_setter.data.vst.param_spec import ParameterValue, decode_model_output
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.pyfdn_decoder import PyFDNParameterDecoder

PYFDN_SPEC_NAMES = tuple(name for name in param_specs if name.startswith("pyfdn_"))


def _assert_native_mapping_matches(
    actual: Mapping[str, Tensor],
    expected: Mapping[str, ParameterValue],
    *,
    dtype: torch.dtype,
) -> None:
    assert set(actual) == set(expected)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    for name, value in expected.items():
        np.testing.assert_allclose(
            actual[name].detach().cpu().numpy(),
            value,
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )


def _span(decoder: PyFDNParameterDecoder, name: str) -> slice:
    return next(
        span for parameter, span in decoder.spec.encoded_slices() if parameter.name == name
    )


@pytest.mark.parametrize("param_spec", PYFDN_SPEC_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decoder_arbitrary_out_of_range_row_matches_offline_mapping(
    param_spec: str, dtype: torch.dtype
) -> None:
    """Every registered pyFDN schema matches the native NumPy decoder.

    :param param_spec: Registered pyFDN schema under test.
    :param dtype: Model prediction precision.
    """
    decoder = PyFDNParameterDecoder(param_spec)
    row = torch.linspace(-2.25, 2.25, decoder.spec.encoded_width, dtype=dtype)

    actual = decoder(row)
    expected, _ = decode_model_output(row.numpy(), decoder.spec)

    _assert_native_mapping_matches(actual, expected, dtype=dtype)


@pytest.mark.parametrize("prediction, delay", [(-0.90875, 437.0), (-0.9987499713897705, 401.0)])
def test_decoder_float32_delay_half_boundary_uses_offline_precision(
    prediction: float, delay: float
) -> None:
    """Promotion before affine decoding preserves an unstable half boundary.

    :param prediction: Float32 model coordinate adjacent to a rounding boundary.
    :param delay: Native delay selected by the canonical promoted calculation.
    """
    decoder = PyFDNParameterDecoder("pyfdn_n8_mono_householder")
    row = torch.zeros(decoder.spec.encoded_width, dtype=torch.float32)
    row[0] = prediction

    actual = decoder(row)
    expected, _ = decode_model_output(row.numpy(), decoder.spec)

    assert actual["delays"][0].item() == delay
    _assert_native_mapping_matches(actual, expected, dtype=row.dtype)


def test_decoder_kronecker_negative_and_reflection_branches_match_offline() -> None:
    """Rounded flags select both kernel branches without changing product order."""
    decoder = PyFDNParameterDecoder("pyfdn_n8_mono_kronecker")
    row = torch.zeros(decoder.spec.encoded_width, dtype=torch.float64)
    row[_span(decoder, PYFDN_KRONECKER_ANGLES_NAME)] = torch.tensor(
        [0.8, -0.6, -0.3, 0.9, -0.4, 0.2], dtype=row.dtype
    )
    row[_span(decoder, PYFDN_KRONECKER_REFLECT_NAME)] = torch.tensor(
        [-1.0, 1.0, -1.0], dtype=row.dtype
    )

    actual = decoder(row)
    expected, _ = decode_model_output(row.numpy(), decoder.spec)

    assert actual[PYFDN_KRONECKER_REFLECT_NAME].tolist() == [0.0, 1.0, 0.0]
    _assert_native_mapping_matches(actual, expected, dtype=row.dtype)


@pytest.mark.parametrize("sine", [-1e-7, 1e-7])
def test_decoder_givens_periodic_seam_matches_offline(sine: float) -> None:
    """Directions on either side of the angle seam retain their signed radians.

    :param sine: Signed sine coordinate selecting one side of the seam.
    """
    decoder = PyFDNParameterDecoder("pyfdn_gotz_n8_mono_fixed_delays_givens")
    row = torch.zeros(decoder.spec.encoded_width, dtype=torch.float64)
    angle_span = _span(decoder, PYFDN_FEEDBACK_GIVENS_ANGLES_NAME)
    row[angle_span.start] = -1.0
    row[angle_span.start + 1] = sine

    actual = decoder(row)
    expected, _ = decode_model_output(row.numpy(), decoder.spec)

    assert torch.sign(actual[PYFDN_FEEDBACK_GIVENS_ANGLES_NAME][0]).item() == np.sign(sine)
    _assert_native_mapping_matches(actual, expected, dtype=row.dtype)


@pytest.mark.parametrize(
    ("param_spec", "coordinate_name"),
    [
        ("pyfdn_n8_mono_householder_vector", PYFDN_HOUSEHOLDER_VECTOR_NAME),
        ("pyfdn_n8_mono_kronecker", PYFDN_KRONECKER_ANGLES_NAME),
        ("pyfdn_gotz_n8_mono_fixed_delays", PYFDN_FEEDBACK_SKEW_NAME),
        (
            "pyfdn_gotz_n8_mono_fixed_delays_givens",
            PYFDN_FEEDBACK_GIVENS_ANGLES_NAME,
        ),
    ],
)
def test_decoder_continuous_feedback_matrix_probe_has_finite_nonzero_gradient(
    param_spec: str, coordinate_name: str
) -> None:
    """A matrix-entry probe depends on each learned feedback coordinate system.

    :param param_spec: Feedback topology under test.
    :param coordinate_name: Span carrying its continuous coordinates.
    """
    decoder = PyFDNParameterDecoder(param_spec)
    row = torch.linspace(-0.73, 0.91, decoder.spec.encoded_width, dtype=torch.float64)
    row.requires_grad_(True)

    feedback = decoder(row)["feedback_matrix"]
    (feedback[0, 1] + 0.37 * feedback[2, 5]).backward()

    assert row.grad is not None
    gradient = row.grad[_span(decoder, coordinate_name)]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


@pytest.mark.parametrize(
    ("field", "expected_width"),
    [("delays", 8), (PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME, 1)],
)
def test_decoder_discrete_fields_have_zero_gradients(field: str, expected_width: int) -> None:
    """Rounded renderer controls intentionally stop useful gradients.

    :param field: Discrete field to probe.
    :param expected_width: Number of encoded coordinates owned by the field.
    """
    decoder = PyFDNParameterDecoder("pyfdn_pitchshift_n8_mono_householder")
    row = torch.linspace(-0.8, 0.8, decoder.spec.encoded_width, dtype=torch.float64)
    row.requires_grad_(True)

    decoder(row)[field].sum().backward()

    assert row.grad is not None
    gradient = row.grad[_span(decoder, field)]
    assert gradient.numel() == expected_width
    assert torch.count_nonzero(gradient).item() == 0


def test_decoder_zero_householder_direction_uses_basis_with_zero_gradients() -> None:
    """A directionless Householder vector becomes the first basis vector safely."""
    decoder = PyFDNParameterDecoder("pyfdn_n8_mono_householder_vector")
    row = torch.full((decoder.spec.encoded_width,), 0.25, dtype=torch.float64)
    field_span = _span(decoder, PYFDN_HOUSEHOLDER_VECTOR_NAME)
    row[field_span] = 0.0
    row.requires_grad_(True)

    decoded = decoder(row)
    decoded["feedback_matrix"][0, 0].backward()

    assert decoded[PYFDN_HOUSEHOLDER_VECTOR_NAME].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert row.grad is not None
    gradient = row.grad[field_span]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() == 0


def test_decoder_zero_angle_pairs_use_zero_radians_with_zero_gradients() -> None:
    """Directionless Givens pairs become zero angles without NaN gradients."""
    decoder = PyFDNParameterDecoder("pyfdn_gotz_n8_mono_fixed_delays_givens")
    row = torch.full((decoder.spec.encoded_width,), 0.25, dtype=torch.float64)
    field_span = _span(decoder, PYFDN_FEEDBACK_GIVENS_ANGLES_NAME)
    row[field_span] = 0.0
    row.requires_grad_(True)

    decoded = decoder(row)
    decoded["feedback_matrix"][0, 1].backward()

    assert torch.count_nonzero(decoded[PYFDN_FEEDBACK_GIVENS_ANGLES_NAME]).item() == 0
    assert row.grad is not None
    gradient = row.grad[field_span]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() == 0


@pytest.mark.parametrize("width_delta", [-1, 1])
def test_decoder_wrong_width_raises(width_delta: int) -> None:
    """Malformed rows fail before positional decoding.

    :param width_delta: Missing or extra coordinate count.
    """
    decoder = PyFDNParameterDecoder("pyfdn_n8_mono_householder")
    with pytest.raises(ValueError, match=f"width {decoder.spec.encoded_width}"):
        decoder(torch.zeros(decoder.spec.encoded_width + width_delta))


def test_decoder_nonfinite_row_raises() -> None:
    """Non-finite model predictions fail before matrix construction."""
    decoder = PyFDNParameterDecoder("pyfdn_n8_mono_householder")
    with pytest.raises(ValueError, match="finite"):
        decoder(torch.full((decoder.spec.encoded_width,), float("nan")))
