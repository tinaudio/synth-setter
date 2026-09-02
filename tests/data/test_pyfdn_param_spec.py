"""Contracts for the fixed pyFDN order-8 mono parameter distribution."""

from typing import cast

import numpy as np

from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.vst.param_spec import ContinuousParameter

_EXPECTED_COORDINATE_NAMES = (
    "delays.0",
    "delays.1",
    "delays.2",
    "delays.3",
    "delays.4",
    "delays.5",
    "delays.6",
    "delays.7",
    "feedback_matrix.0.0",
    "feedback_matrix.0.1",
    "feedback_matrix.0.2",
    "feedback_matrix.0.3",
    "feedback_matrix.0.4",
    "feedback_matrix.0.5",
    "feedback_matrix.0.6",
    "feedback_matrix.0.7",
    "feedback_matrix.1.0",
    "feedback_matrix.1.1",
    "feedback_matrix.1.2",
    "feedback_matrix.1.3",
    "feedback_matrix.1.4",
    "feedback_matrix.1.5",
    "feedback_matrix.1.6",
    "feedback_matrix.1.7",
    "feedback_matrix.2.0",
    "feedback_matrix.2.1",
    "feedback_matrix.2.2",
    "feedback_matrix.2.3",
    "feedback_matrix.2.4",
    "feedback_matrix.2.5",
    "feedback_matrix.2.6",
    "feedback_matrix.2.7",
    "feedback_matrix.3.0",
    "feedback_matrix.3.1",
    "feedback_matrix.3.2",
    "feedback_matrix.3.3",
    "feedback_matrix.3.4",
    "feedback_matrix.3.5",
    "feedback_matrix.3.6",
    "feedback_matrix.3.7",
    "feedback_matrix.4.0",
    "feedback_matrix.4.1",
    "feedback_matrix.4.2",
    "feedback_matrix.4.3",
    "feedback_matrix.4.4",
    "feedback_matrix.4.5",
    "feedback_matrix.4.6",
    "feedback_matrix.4.7",
    "feedback_matrix.5.0",
    "feedback_matrix.5.1",
    "feedback_matrix.5.2",
    "feedback_matrix.5.3",
    "feedback_matrix.5.4",
    "feedback_matrix.5.5",
    "feedback_matrix.5.6",
    "feedback_matrix.5.7",
    "feedback_matrix.6.0",
    "feedback_matrix.6.1",
    "feedback_matrix.6.2",
    "feedback_matrix.6.3",
    "feedback_matrix.6.4",
    "feedback_matrix.6.5",
    "feedback_matrix.6.6",
    "feedback_matrix.6.7",
    "feedback_matrix.7.0",
    "feedback_matrix.7.1",
    "feedback_matrix.7.2",
    "feedback_matrix.7.3",
    "feedback_matrix.7.4",
    "feedback_matrix.7.5",
    "feedback_matrix.7.6",
    "feedback_matrix.7.7",
    "input_matrix.0.0",
    "input_matrix.1.0",
    "input_matrix.2.0",
    "input_matrix.3.0",
    "input_matrix.4.0",
    "input_matrix.5.0",
    "input_matrix.6.0",
    "input_matrix.7.0",
    "output_matrix.0.0",
    "output_matrix.0.1",
    "output_matrix.0.2",
    "output_matrix.0.3",
    "output_matrix.0.4",
    "output_matrix.0.5",
    "output_matrix.0.6",
    "output_matrix.0.7",
    "direct_matrix.0.0",
    "post_delay.rt_dc_seconds",
    "post_delay.rt_nyquist_seconds",
)


def test_pyfdn_spec_layout_has_exact_91_columns_and_slices() -> None:
    """The model boundary preserves the issue-defined field order and widths."""
    layout = [
        (parameter.name, span.start, span.stop)
        for parameter, span in PYFDN_N8_MONO_PARAM_SPEC.encoded_slices()
    ]

    assert PYFDN_N8_MONO_PARAM_SPEC.encoded_width == 91
    assert layout == [
        ("delays", 0, 8),
        ("feedback_matrix", 8, 72),
        ("input_matrix", 72, 80),
        ("output_matrix", 80, 88),
        ("direct_matrix", 88, 89),
        ("post_delay.rt_dc_seconds", 89, 90),
        ("post_delay.rt_nyquist_seconds", 90, 91),
    ]
    assert PYFDN_N8_MONO_PARAM_SPEC.note_params == []


def test_pyfdn_spec_coordinate_names_match_exact_c_order_layout() -> None:
    """Every model coordinate has the stable field-and-index label required by metrics."""
    assert tuple(PYFDN_N8_MONO_PARAM_SPEC.encoded_names) == _EXPECTED_COORDINATE_NAMES


def test_pyfdn_spec_rt_controls_have_exact_positive_bounds() -> None:
    """Both predicted decay controls use the approved reverberation-time domain."""
    rt_dc, rt_nyquist = PYFDN_N8_MONO_PARAM_SPEC.synth_params[-2:]

    assert isinstance(rt_dc, ContinuousParameter)
    assert (rt_dc.name, rt_dc.min, rt_dc.max) == (
        "post_delay.rt_dc_seconds",
        0.1,
        4.0,
    )
    assert isinstance(rt_nyquist, ContinuousParameter)
    assert (rt_nyquist.name, rt_nyquist.min, rt_nyquist.max) == (
        "post_delay.rt_nyquist_seconds",
        0.1,
        4.0,
    )


def test_pyfdn_spec_same_seed_samples_same_patch() -> None:
    """A local RNG seed fully determines every native pyFDN field."""
    first, first_notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(123))
    second, second_notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(123))

    assert first_notes == second_notes == {}
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_pyfdn_spec_different_seeds_change_sampled_patch() -> None:
    """Distinct local RNG seeds do not collapse onto one fixed patch."""
    first, _ = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(123))
    second, _ = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(456))

    assert any(not np.array_equal(first[name], second[name]) for name in first)


def test_pyfdn_spec_delay_sampling_preserves_rng_draw_order() -> None:
    """Delay-line identity follows the generator's unsorted integer draw order."""
    params, _ = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(123))

    np.testing.assert_array_equal(
        params["delays"],
        np.array([412, 946, 874, 443, 1128, 576, 604, 547], dtype=np.int64),
    )


def test_pyfdn_spec_sampled_feedback_matrix_is_orthogonal() -> None:
    """The feedback target is Haar-orthogonal without post-sample repair."""
    params, _ = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(123))
    feedback = cast(np.ndarray, params["feedback_matrix"])

    np.testing.assert_allclose(feedback.T @ feedback, np.eye(8), atol=1e-12)


def test_pyfdn_spec_sampled_fields_have_exact_native_shapes_and_dtypes() -> None:
    """Sampling emits the arrays consumed directly by the native build codec."""
    params, _ = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(123))
    delays = cast(np.ndarray, params["delays"])
    feedback = cast(np.ndarray, params["feedback_matrix"])
    inputs = cast(np.ndarray, params["input_matrix"])
    outputs = cast(np.ndarray, params["output_matrix"])
    direct = cast(np.ndarray, params["direct_matrix"])

    assert (delays.shape, delays.dtype) == ((8,), np.dtype(np.int64))
    assert (feedback.shape, feedback.dtype) == ((8, 8), np.dtype(np.float64))
    assert (inputs.shape, inputs.dtype) == ((8, 1), np.dtype(np.float64))
    assert (outputs.shape, outputs.dtype) == ((1, 8), np.dtype(np.float64))
    assert (direct.shape, direct.dtype) == ((1, 1), np.dtype(np.float64))


def test_pyfdn_spec_sampled_rt_controls_are_bounded_python_floats() -> None:
    """Sampled reverberation times stay positive and inside the model contract."""
    params, _ = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(123))
    rt_dc = params["post_delay.rt_dc_seconds"]
    rt_nyquist = params["post_delay.rt_nyquist_seconds"]

    assert isinstance(rt_dc, float)
    assert 0.1 <= rt_dc <= 4.0
    assert isinstance(rt_nyquist, float)
    assert 0.1 <= rt_nyquist <= 4.0


def test_pyfdn_spec_encoding_is_float32_and_round_trips_native_fields() -> None:
    """Encoded rows preserve every sampled native field."""
    params, notes = PYFDN_N8_MONO_PARAM_SPEC.sample(np.random.default_rng(123))

    encoded = PYFDN_N8_MONO_PARAM_SPEC.encode(params, notes)
    decoded, decoded_notes = PYFDN_N8_MONO_PARAM_SPEC.decode(encoded)

    assert encoded.shape == (91,)
    assert encoded.dtype == np.float32
    assert decoded_notes == {}
    np.testing.assert_array_equal(decoded["delays"], params["delays"])
    for name in (
        "feedback_matrix",
        "input_matrix",
        "output_matrix",
        "direct_matrix",
        "post_delay.rt_dc_seconds",
        "post_delay.rt_nyquist_seconds",
    ):
        np.testing.assert_allclose(decoded[name], params[name], atol=1.2e-7)
