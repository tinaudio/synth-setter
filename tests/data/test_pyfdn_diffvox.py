"""Contracts for the DiffVox vocal effects chain rendered through pyFDN."""

from typing import cast

import numpy as np
import pytest

from synth_setter.data.pyfdn_diffvox import (
    diffvox_feedback_matrix,
    render_diffvox_chain,
)
from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import (
    PYFDN_DIFFVOX_DELAY_EVEN_PAN_NAME,
    PYFDN_DIFFVOX_DELAY_FEEDBACK_NAME,
    PYFDN_DIFFVOX_DELAY_GAIN_NAME,
    PYFDN_DIFFVOX_DELAY_LP_FREQ_NAME,
    PYFDN_DIFFVOX_DELAY_ODD_PAN_NAME,
    PYFDN_DIFFVOX_DELAY_TIME_NAME,
    PYFDN_DIFFVOX_DIRECT_PAN_NAME,
    PYFDN_DIFFVOX_PARAM_SPEC,
    PYFDN_DIFFVOX_REVERB_INPUT_NAME,
    PYFDN_DIFFVOX_REVERB_OUTPUT_NAME,
    PYFDN_DIFFVOX_REVERB_RT_NAME,
    PYFDN_DIFFVOX_REVERB_SKEW_NAME,
    PYFDN_DIFFVOX_SEND_NAME,
)
from synth_setter.data.vst.param_spec import ParameterValues
from synth_setter.param_spec_name import ParamSpecName

_SAMPLE_RATE = 44_100.0
_FRAMES = 176_400
_DIFFVOX = ParamSpecName("pyfdn_diffvox")


def _impulse() -> np.ndarray:
    source = np.zeros(_FRAMES, dtype=np.float64)
    source[0] = 1.0
    return source


def _dry_params() -> ParameterValues:
    """Build a flat-EQ, centre-panned patch with silent delay and reverb sends.

    :returns: Native controls whose only audible path is the direct signal.
    """
    params, _ = PYFDN_DIFFVOX_PARAM_SPEC.sample(np.random.default_rng(0))
    for name in PYFDN_DIFFVOX_PARAM_SPEC.synth_param_names:
        if name.endswith(".gain_db"):
            params[name] = 0.0
    params["peq.lp.freq_hz"] = 18_000.0
    params["peq.lp.q"] = 0.707
    params["peq.hp.freq_hz"] = 16.0
    params["peq.hp.q"] = 0.707
    params[PYFDN_DIFFVOX_DIRECT_PAN_NAME] = 0.5
    params[PYFDN_DIFFVOX_DELAY_GAIN_NAME] = 0.0
    params[PYFDN_DIFFVOX_DELAY_TIME_NAME] = 0.25
    params[PYFDN_DIFFVOX_DELAY_FEEDBACK_NAME] = 0.5
    params[PYFDN_DIFFVOX_DELAY_LP_FREQ_NAME] = 16_000.0
    params[PYFDN_DIFFVOX_SEND_NAME] = 0.0
    params[PYFDN_DIFFVOX_REVERB_OUTPUT_NAME] = np.zeros((2, 6), dtype=np.float64)
    params[PYFDN_DIFFVOX_REVERB_INPUT_NAME] = np.full((6, 2), 0.5, dtype=np.float64)
    params[PYFDN_DIFFVOX_REVERB_RT_NAME] = np.full(10, 1.0, dtype=np.float64)
    return params


def _magnitude_at(audio: np.ndarray, freq_hz: float) -> float:
    spectrum = np.fft.rfft(audio)
    bin_index = int(round(freq_hz * audio.shape[0] / _SAMPLE_RATE))
    return float(np.abs(spectrum[bin_index]))


def test_diffvox_spec_has_82_learned_coordinates_and_fixed_midi_stubs() -> None:
    """The chain exposes every DiffVox control except the skipped compressor."""
    spec = PYFDN_DIFFVOX_PARAM_SPEC

    _, note_params = spec.sample(np.random.default_rng(3))

    assert spec.encoded_width == 82
    assert note_params == {"pitch": 0, "note_start_and_end": (0.0, 0.0)}
    assert "feedback_matrix" not in spec.synth_param_names


def test_diffvox_spec_sampled_patch_round_trips_through_codec() -> None:
    """Sampled native values survive encode and decode."""
    params, note = PYFDN_DIFFVOX_PARAM_SPEC.sample(np.random.default_rng(11))

    decoded, _ = PYFDN_DIFFVOX_PARAM_SPEC.decode(PYFDN_DIFFVOX_PARAM_SPEC.encode(params, note))

    for name, value in params.items():
        np.testing.assert_allclose(decoded[name], value, rtol=1e-5, atol=1e-4, err_msg=name)


def test_diffvox_feedback_matrix_zero_skew_is_identity() -> None:
    """The matrix exponential of a zero skew matrix is the identity."""
    np.testing.assert_allclose(diffvox_feedback_matrix(np.zeros(15)), np.eye(6), atol=1e-15)


def test_diffvox_feedback_matrix_is_orthogonal_for_any_skew() -> None:
    """Every skew vector maps to a unilossless orthogonal feedback matrix."""
    matrix = diffvox_feedback_matrix(np.linspace(-3.0, 3.0, 15))

    np.testing.assert_allclose(matrix.T @ matrix, np.eye(6), atol=1e-12)


def test_diffvox_render_dry_centre_pan_passes_impulse_to_both_channels() -> None:
    """Flat EQ with silent sends leaves a centre-panned unit impulse."""
    output = render_diffvox_chain(_dry_params(), _impulse(), sample_rate=_SAMPLE_RATE)

    assert output.shape == (_FRAMES, 2)
    np.testing.assert_allclose(_magnitude_at(output[:, 0], 1_000.0), 1.0, rtol=0.02)
    np.testing.assert_allclose(_magnitude_at(output[:, 1], 1_000.0), 1.0, rtol=0.02)


def test_diffvox_render_hard_left_direct_pan_silences_right_channel() -> None:
    """A direct pan of zero routes the dry signal entirely to the left channel."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_DIRECT_PAN_NAME] = 0.0

    output = render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)

    np.testing.assert_allclose(_magnitude_at(output[:, 0], 1_000.0), np.sqrt(2.0), rtol=0.02)
    np.testing.assert_array_equal(output[:, 1], 0.0)


def test_diffvox_render_peak_band_boosts_its_centre_frequency() -> None:
    """A 12 dB peak at 1 kHz multiplies that bin's magnitude by 3.98."""
    params = _dry_params()
    params["peq.pk1.freq_hz"] = 1_000.0
    params["peq.pk1.gain_db"] = 12.0
    params["peq.pk1.q"] = 4.0

    output = render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)

    np.testing.assert_allclose(_magnitude_at(output[:, 0], 1_000.0), 3.981, rtol=0.03)


def test_diffvox_render_high_pass_removes_direct_current() -> None:
    """Raising the high-pass cutoff drives the DC magnitude to zero."""
    params = _dry_params()
    params["peq.hp.freq_hz"] = 5_000.0

    output = render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)

    assert abs(output[:, 0].sum()) < 1e-6
    assert _magnitude_at(output[:, 0], 200.0) < 0.01


def test_diffvox_render_ping_pong_delay_alternates_channels() -> None:
    """Odd echoes land on the odd panner and even echoes on the even panner."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_DELAY_GAIN_NAME] = 1.0
    params[PYFDN_DIFFVOX_DELAY_ODD_PAN_NAME] = 0.0
    params[PYFDN_DIFFVOX_DELAY_EVEN_PAN_NAME] = 1.0
    delay_samples = 11_025

    output = render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)

    first_echo = output[delay_samples - 2 : delay_samples + 200]
    second_echo = output[2 * delay_samples - 2 : 2 * delay_samples + 200]
    assert np.abs(first_echo[:, 0]).max() > 0.5
    assert np.abs(first_echo[:, 1]).max() < 1e-9
    assert np.abs(second_echo[:, 0]).max() < 1e-6
    assert np.abs(second_echo[:, 1]).max() > 0.2


def test_diffvox_render_delay_feedback_scales_successive_echoes() -> None:
    """Each echo's DC sum carries one more feedback factor than the previous one."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_DELAY_GAIN_NAME] = 1.0
    params[PYFDN_DIFFVOX_DELAY_FEEDBACK_NAME] = 0.25
    params[PYFDN_DIFFVOX_DELAY_ODD_PAN_NAME] = 0.5
    params[PYFDN_DIFFVOX_DELAY_EVEN_PAN_NAME] = 0.5
    delay_samples = 11_025

    output = render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)

    first = output[delay_samples : delay_samples + 200, 0].sum()
    second = output[2 * delay_samples : 2 * delay_samples + 200, 0].sum()
    assert second / first == pytest.approx(0.25, rel=0.05)


def test_diffvox_render_reverb_output_gains_add_a_decaying_tail() -> None:
    """A non-zero output matrix produces late energy that a silent reverb lacks."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_REVERB_OUTPUT_NAME] = np.full((2, 6), 0.3, dtype=np.float64)
    params[PYFDN_DIFFVOX_REVERB_RT_NAME] = np.full(10, 2.0, dtype=np.float64)

    output = render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)

    late = output[22_050:44_100]
    later = output[44_100:66_150]
    assert np.abs(late).max() > 1e-3
    assert np.abs(later).max() < 0.5 * np.abs(late).max()


def test_diffvox_render_longer_reverb_time_keeps_more_late_energy() -> None:
    """Raising every band's RT slows the tail decay."""
    short = _dry_params()
    short[PYFDN_DIFFVOX_REVERB_OUTPUT_NAME] = np.full((2, 6), 0.3, dtype=np.float64)
    short[PYFDN_DIFFVOX_REVERB_RT_NAME] = np.full(10, 0.5, dtype=np.float64)
    long = dict(short)
    long[PYFDN_DIFFVOX_REVERB_RT_NAME] = np.full(10, 4.0, dtype=np.float64)

    short_tail = render_diffvox_chain(short, _impulse(), sample_rate=_SAMPLE_RATE)[88_200:]
    long_tail = render_diffvox_chain(long, _impulse(), sample_rate=_SAMPLE_RATE)[88_200:]

    assert np.square(long_tail).sum() > 10.0 * np.square(short_tail).sum()


def test_diffvox_render_steep_reverb_profile_stays_bounded() -> None:
    """A short RT band beside long ones must not push the FDN loop past unity gain."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_REVERB_OUTPUT_NAME] = np.full((2, 6), 0.3, dtype=np.float64)
    params[PYFDN_DIFFVOX_REVERB_RT_NAME] = np.array(
        [0.11, 3.66, 1.6, 2.79, 2.74, 3.12, 2.69, 4.97, 4.42, 4.79], dtype=np.float64
    )

    output = render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)

    assert np.abs(output).max() < 10.0
    assert np.abs(output[132_300:]).max() < np.abs(output[44_100:88_200]).max()


def test_diffvox_render_send_routes_delay_output_into_reverb() -> None:
    """The delay-to-reverb send only matters when the delay is audible."""
    silent_delay = _dry_params()
    silent_delay[PYFDN_DIFFVOX_REVERB_OUTPUT_NAME] = np.full((2, 6), 0.3, dtype=np.float64)
    sent = dict(silent_delay)
    sent[PYFDN_DIFFVOX_SEND_NAME] = 1.0
    audible = dict(sent)
    audible[PYFDN_DIFFVOX_DELAY_GAIN_NAME] = 0.5

    unsent_render = render_diffvox_chain(silent_delay, _impulse(), sample_rate=_SAMPLE_RATE)
    sent_render = render_diffvox_chain(sent, _impulse(), sample_rate=_SAMPLE_RATE)
    audible_render = render_diffvox_chain(audible, _impulse(), sample_rate=_SAMPLE_RATE)

    np.testing.assert_array_equal(sent_render, unsent_render)
    assert not np.array_equal(audible_render, sent_render)


def test_diffvox_render_rejects_missing_control() -> None:
    """A patch lacking one chain control is refused before any processing."""
    params = _dry_params()
    del params[PYFDN_DIFFVOX_SEND_NAME]

    with pytest.raises(ValueError, match="must contain exactly"):
        render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)


def test_diffvox_render_rejects_wrong_reverb_matrix_shape() -> None:
    """A mis-shaped reverb input matrix is refused."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_REVERB_INPUT_NAME] = np.zeros((2, 6), dtype=np.float64)

    with pytest.raises(ValueError, match="must have shape"):
        render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)


def test_diffvox_render_rejects_out_of_range_feedback() -> None:
    """A feedback beyond the ParamSpec bound is refused."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_DELAY_FEEDBACK_NAME] = 1.5

    with pytest.raises(ValueError, match="delay.feedback"):
        render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)


def test_diffvox_render_rejects_sample_rate_below_twice_the_highest_eq_bound() -> None:
    """A rate whose Nyquist limit an 18 kHz cutoff could exceed is refused."""
    with pytest.raises(ValueError, match="Nyquist"):
        render_diffvox_chain(_dry_params(), _impulse(), sample_rate=32_000.0)


def test_diffvox_render_rejects_float32_reverb_matrix() -> None:
    """Array controls must arrive as float64 like every other pyFDN native array."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_REVERB_OUTPUT_NAME] = np.zeros((2, 6), dtype=np.float32)

    with pytest.raises(TypeError, match="dtype float64"):
        render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)


def test_diffvox_render_rejects_list_for_array_control() -> None:
    """A Python list is not accepted where a native array is required."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_REVERB_RT_NAME] = [1.0] * 10  # type: ignore[assignment]

    with pytest.raises(TypeError, match="NumPy array"):
        render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)


def test_diffvox_render_rejects_non_real_scalar_control() -> None:
    """A string where a real scalar is required is refused before processing."""
    params = _dry_params()
    params[PYFDN_DIFFVOX_DIRECT_PAN_NAME] = "centre"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="real scalar"):
        render_diffvox_chain(params, _impulse(), sample_rate=_SAMPLE_RATE)


def test_diffvox_renderer_returns_stereo_float32_and_is_repeatable() -> None:
    """Fresh filter state per render makes the stereo output identical each call."""
    renderer = PyFDNRenderer(param_spec_name=_DIFFVOX, channels=2)
    params, _ = PYFDN_DIFFVOX_PARAM_SPEC.sample(np.random.default_rng(5))

    first = renderer.render(params)
    second = renderer.render(params)

    assert first.shape == (2, _FRAMES)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(second, first)


def test_diffvox_renderer_rejects_mono_channel_contract() -> None:
    """The stereo chain cannot be constructed with the mono FDN geometry."""
    with pytest.raises(ValueError, match="fixed render contract"):
        PyFDNRenderer(param_spec_name=_DIFFVOX, channels=1)


def test_diffvox_renderer_chirp_excitation_returns_finite_stereo_audio() -> None:
    """The canonical chirp traverses the chain end to end."""
    renderer = PyFDNRenderer(param_spec_name=_DIFFVOX, channels=2, excitation="chirp")
    params, _ = PYFDN_DIFFVOX_PARAM_SPEC.sample(np.random.default_rng(9))

    audio = renderer.render(params)

    assert audio.shape == (2, _FRAMES)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() > 0.0


def test_diffvox_renderer_provenance_names_process_fdn() -> None:
    """Impulse provenance identifies the block-processing path used by the chain."""
    renderer = PyFDNRenderer(param_spec_name=_DIFFVOX, channels=2)

    assert renderer.source_provenance["implementation"] == "pyFDN.process_fdn"
    assert renderer.source_provenance["channels"] == 1


def test_diffvox_sampled_patch_renders_within_amplitude_budget() -> None:
    """A random sampled patch stays finite after the full chain."""
    renderer = PyFDNRenderer(param_spec_name=_DIFFVOX, channels=2)
    params, _ = PYFDN_DIFFVOX_PARAM_SPEC.sample(np.random.default_rng(21))
    skew = cast(np.ndarray, params[PYFDN_DIFFVOX_REVERB_SKEW_NAME])
    assert skew.shape == (15,)

    audio = renderer.render(params)

    assert np.isfinite(audio).all()
