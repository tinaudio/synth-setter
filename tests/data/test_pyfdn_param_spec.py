"""Contracts for the fixed-Householder pyFDN parameter distribution."""

from typing import cast

import numpy as np

from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC
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

_EXPECTED_HOUSEHOLDER_FEEDBACK = np.array(
    [
        [0.75, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
        [-0.25, 0.75, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
        [-0.25, -0.25, 0.75, -0.25, -0.25, -0.25, -0.25, -0.25],
        [-0.25, -0.25, -0.25, 0.75, -0.25, -0.25, -0.25, -0.25],
        [-0.25, -0.25, -0.25, -0.25, 0.75, -0.25, -0.25, -0.25],
        [-0.25, -0.25, -0.25, -0.25, -0.25, 0.75, -0.25, -0.25],
        [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, 0.75, -0.25],
        [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, 0.75],
    ],
    dtype=np.float64,
)


def test_pyfdn_spec_layout_has_exact_27_columns_and_slices() -> None:
    """The model boundary excludes the fixed feedback matrix."""
    layout = [
        (parameter.name, span.start, span.stop)
        for parameter, span in PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.encoded_slices()
    ]

    assert PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.encoded_width == 27
    assert layout == [
        ("delays", 0, 8),
        ("input_matrix", 8, 16),
        ("output_matrix", 16, 24),
        ("direct_matrix", 24, 25),
        ("post_delay.rt_dc_seconds", 25, 26),
        ("post_delay.rt_nyquist_seconds", 26, 27),
    ]
    assert PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.note_params == []


def test_pyfdn_spec_coordinate_names_match_exact_c_order_layout() -> None:
    """Every learned coordinate has the stable label required by metrics."""
    assert (
        tuple(PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.encoded_names)
        == _EXPECTED_COORDINATE_NAMES
    )


def test_pyfdn_spec_rt_controls_have_exact_positive_bounds() -> None:
    """Both predicted decay controls use the approved reverberation-time domain."""
    rt_dc, rt_nyquist = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.synth_params[-2:]

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
    first, first_notes = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(
        np.random.default_rng(123)
    )
    second, second_notes = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(
        np.random.default_rng(123)
    )

    assert first_notes == second_notes == {
        "pitch": 0,
        "note_start_and_end": (0.0, 0.0),
    }
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_pyfdn_spec_different_seeds_change_sampled_patch() -> None:
    """Distinct local RNG seeds do not collapse onto one fixed learned patch."""
    first, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(123))
    second, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(456))

    learned_names = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.synth_param_names
    assert any(not np.array_equal(first[name], second[name]) for name in learned_names)


def test_pyfdn_spec_delay_sampling_preserves_rng_draw_order() -> None:
    """Delay-line identity follows the generator's unsorted integer draw order."""
    params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(123))

    np.testing.assert_array_equal(
        params["delays"],
        np.array([412, 946, 874, 443, 1128, 576, 604, 547], dtype=np.int64),
    )


def test_pyfdn_spec_sampled_fields_have_exact_native_shapes_and_dtypes() -> None:
    """Sampling emits the arrays consumed directly by the native build codec."""
    params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(123))
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
    params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(123))
    rt_dc = params["post_delay.rt_dc_seconds"]
    rt_nyquist = params["post_delay.rt_nyquist_seconds"]

    assert isinstance(rt_dc, float)
    assert 0.1 <= rt_dc <= 4.0
    assert isinstance(rt_nyquist, float)
    assert 0.1 <= rt_nyquist <= 4.0


def test_pyfdn_spec_samples_all_ones_householder_reflection() -> None:
    """Every patch uses pyFDN's order-8 Householder reflection."""
    params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(123))
    feedback = cast(np.ndarray, params["feedback_matrix"])

    np.testing.assert_allclose(
        feedback, _EXPECTED_HOUSEHOLDER_FEEDBACK, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(feedback.T @ feedback, np.eye(8), atol=1e-15)


def test_pyfdn_spec_encoding_round_trips_learned_fields() -> None:
    """Encoding and decoding preserve every learned native field."""
    params, notes = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(
        np.random.default_rng(123)
    )

    encoded = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.encode(params, notes)
    decoded, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.decode(encoded)

    assert (encoded.shape, encoded.dtype) == ((27,), np.dtype(np.float32))
    np.testing.assert_array_equal(decoded["delays"], params["delays"])
    for name in (
        "input_matrix",
        "output_matrix",
        "direct_matrix",
        "post_delay.rt_dc_seconds",
        "post_delay.rt_nyquist_seconds",
    ):
        np.testing.assert_allclose(decoded[name], params[name], atol=1.2e-7)


def test_pyfdn_spec_decode_restores_fixed_renderer_values() -> None:
    """Decoded rows restore feedback and MIDI fields outside model coordinates."""
    params, notes = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(
        np.random.default_rng(123)
    )
    encoded = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.encode(params, notes)

    decoded, decoded_notes = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.decode(encoded)

    assert decoded_notes == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
    np.testing.assert_allclose(
        decoded["feedback_matrix"], _EXPECTED_HOUSEHOLDER_FEEDBACK, rtol=0.0, atol=1e-15
    )


def test_pyfdn_spec_samples_return_independent_feedback_arrays() -> None:
    """Mutating one patch cannot alter the fixed matrix in later patches."""
    first, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(123))
    first_feedback = cast(np.ndarray, first["feedback_matrix"])
    first_feedback[0, 0] = 0.0

    second, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(123))

    np.testing.assert_allclose(
        second["feedback_matrix"],
        _EXPECTED_HOUSEHOLDER_FEEDBACK,
        rtol=0.0,
        atol=1e-15,
    )
