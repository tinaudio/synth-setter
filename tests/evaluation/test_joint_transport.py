"""Tests for joint time-frequency optimal transport."""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.evaluation.joint_transport import (
    compute_grid_wasserstein_distance,
    compute_joint_time_frequency_ot,
)

_SAMPLE_RATE = 44100
_NUM_SAMPLES = 4096


def _impulse(sample: int, amplitude: float = 1.0) -> np.ndarray:
    """Return a one-channel point impulse.

    :param sample: Zero-based location within the fixed-length test response.
    :param amplitude: Signed linear gain of the sole nonzero sample.
    :returns: Float64 audio with shape ``(1, _NUM_SAMPLES)``.
    """
    audio = np.zeros((1, _NUM_SAMPLES), dtype=np.float64)
    audio[0, sample] = amplitude
    return audio


def test_grid_transport_point_mass_time_shift_returns_scaled_displacement() -> None:
    """A 0.2-second shift costs two default time units."""
    target = np.array([[1.0, 0.0]])
    pred = np.array([[0.0, 1.0]])

    distance = compute_grid_wasserstein_distance(
        target,
        pred,
        time_coordinates=np.array([0.0, 0.2]),
        log_frequency_coordinates=np.array([0.0]),
    )

    assert distance == pytest.approx(2.0)


def test_grid_transport_point_mass_frequency_shift_returns_scaled_displacement() -> None:
    """A two-octave shift costs two default frequency units."""
    target = np.array([[1.0], [0.0]])
    pred = np.array([[0.0], [1.0]])

    distance = compute_grid_wasserstein_distance(
        target,
        pred,
        time_coordinates=np.array([0.0]),
        log_frequency_coordinates=np.array([0.0, 2.0]),
    )

    assert distance == pytest.approx(2.0)


def test_grid_transport_custom_axis_scale_changes_displacement_units() -> None:
    """The public time scale controls the displacement unit."""
    target = np.array([[1.0, 0.0]])
    pred = np.array([[0.0, 1.0]])

    distance = compute_grid_wasserstein_distance(
        target,
        pred,
        time_coordinates=np.array([0.0, 0.2]),
        log_frequency_coordinates=np.array([0.0]),
        time_scale_seconds=0.4,
    )

    assert distance == pytest.approx(0.5)


def test_grid_transport_nonpositive_axis_scale_raises_value_error() -> None:
    """Axis scales must define positive physical cost units."""
    mass = np.array([[1.0]])

    with pytest.raises(ValueError, match="time_scale_seconds"):
        compute_grid_wasserstein_distance(
            mass,
            mass,
            time_coordinates=np.array([0.0]),
            log_frequency_coordinates=np.array([0.0]),
            time_scale_seconds=0.0,
        )


def test_grid_transport_checkerboards_with_same_marginals_have_positive_distance() -> None:
    """Joint transport distinguishes layouts with identical axis marginals."""
    target = np.array([[0.5, 0.0], [0.0, 0.5]])
    pred = np.array([[0.0, 0.5], [0.5, 0.0]])

    distance = compute_grid_wasserstein_distance(
        target,
        pred,
        time_coordinates=np.array([0.0, 0.1]),
        log_frequency_coordinates=np.array([0.0, 1.0]),
    )

    assert distance == pytest.approx(1.0)


def test_grid_transport_nonzero_origins_preserve_joint_displacement() -> None:
    """Ground cost depends on displacement, not absolute coordinate origins."""
    target = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pred = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    distance = compute_grid_wasserstein_distance(
        target,
        pred,
        time_coordinates=np.array([1.0, 1.1, 1.4]),
        log_frequency_coordinates=np.array([10.0, 12.0]),
    )

    assert distance == pytest.approx(6.0)


def test_joint_ot_shifted_impulse_measures_physical_time_displacement() -> None:
    """A 1024-sample delay at 44.1 kHz costs approximately 0.2322 units."""
    distance = compute_joint_time_frequency_ot(_impulse(1024), _impulse(2048), _SAMPLE_RATE)

    assert distance == pytest.approx(0.23219954648526078)


def test_joint_ot_pooled_long_response_retains_late_energy_timing() -> None:
    """Pooling a four-second response retains a two-second reflection displacement."""
    target = np.zeros((1, 176400))
    pred = np.zeros_like(target)
    target[0, 44100] = 1.0
    pred[0, 132300] = 1.0

    distance = compute_joint_time_frequency_ot(target, pred, _SAMPLE_RATE)

    assert distance == pytest.approx(20.0, abs=0.7)


def test_joint_ot_identical_audio_returns_zero() -> None:
    """Identity of indiscernibles holds on actual audio."""
    audio = _impulse(1024)

    assert compute_joint_time_frequency_ot(audio, audio, _SAMPLE_RATE) == pytest.approx(0.0)


def test_joint_ot_audio_order_is_symmetric() -> None:
    """Swapping actual audio inputs preserves the distance."""
    target = _impulse(1024)
    pred = _impulse(2048)

    forward = compute_joint_time_frequency_ot(target, pred, _SAMPLE_RATE)
    reverse = compute_joint_time_frequency_ot(pred, target, _SAMPLE_RATE)

    assert forward == pytest.approx(reverse)


def test_joint_ot_independent_audio_gains_leave_distance_unchanged() -> None:
    """Global normalization removes each input's nonzero gain."""
    target = _impulse(1024)
    pred = _impulse(2048)

    reference = compute_joint_time_frequency_ot(target, pred, _SAMPLE_RATE)
    rescaled = compute_joint_time_frequency_ot(3.0 * target, 0.25 * pred, _SAMPLE_RATE)

    assert rescaled == pytest.approx(reference)


def test_joint_ot_extreme_finite_gains_do_not_underflow_or_overflow() -> None:
    """Pre-STFT peak scaling protects finite signals at dtype extremes."""
    target = _impulse(1024)
    pred = _impulse(2048)

    reference = compute_joint_time_frequency_ot(target, pred, _SAMPLE_RATE)
    rescaled = compute_joint_time_frequency_ot(1e-200 * target, 1e200 * pred, _SAMPLE_RATE)

    assert rescaled == pytest.approx(reference)


def test_joint_ot_one_sided_power_gives_dc_and_tone_equal_linear_energy() -> None:
    """A unit tone carries the same energy as DC with amplitude one over root two."""
    sample_indices = np.arange(65536)
    dc = np.full((1, sample_indices.size), 1.0 / np.sqrt(2.0))
    tone = np.sin(2.0 * np.pi * 128.0 * sample_indices / 2048.0)[None, :]

    full_shift = compute_joint_time_frequency_ot(dc, tone, _SAMPLE_RATE)
    mixed_shift = compute_joint_time_frequency_ot(dc + tone, dc, _SAMPLE_RATE)

    assert mixed_shift / full_shift == pytest.approx(0.5, rel=0.03)


def test_joint_ot_one_sided_power_gives_nyquist_and_tone_equal_linear_energy() -> None:
    """A unit tone carries the same energy as Nyquist at amplitude one over root two."""
    sample_indices = np.arange(65536)
    nyquist = ((-1.0) ** sample_indices / np.sqrt(2.0))[None, :]
    tone = np.sin(2.0 * np.pi * 128.0 * sample_indices / 2048.0)[None, :]

    full_shift = compute_joint_time_frequency_ot(nyquist, tone, _SAMPLE_RATE)
    mixed_shift = compute_joint_time_frequency_ot(nyquist + tone, nyquist, _SAMPLE_RATE)

    assert mixed_shift / full_shift == pytest.approx(0.5, rel=0.03)


def test_joint_ot_opposite_phase_channels_do_not_cancel_energy() -> None:
    """Channel energy is combined without waveform phase cancellation."""
    first = _impulse(1024)[0]
    second = _impulse(2048)[0]
    target = np.stack((first, -first))
    pred = np.stack((second, -second))

    assert compute_joint_time_frequency_ot(target, pred, _SAMPLE_RATE) > 0.0


def test_joint_ot_frequencies_above_eight_kilohertz_are_included() -> None:
    """The fixed log-frequency range extends above 8 kHz."""
    time = np.arange(_NUM_SAMPLES) / _SAMPLE_RATE
    target = np.sin(2.0 * np.pi * 10000.0 * time)[None, :]
    pred = np.sin(2.0 * np.pi * 12000.0 * time)[None, :]

    assert compute_joint_time_frequency_ot(target, pred, _SAMPLE_RATE) > 0.0


@pytest.mark.parametrize("sample_rate", [0, -1, np.inf, np.nan])
def test_joint_ot_invalid_sample_rate_raises_value_error(sample_rate: float) -> None:
    """Nonpositive and nonfinite sample rates are rejected.

    :param sample_rate: Sampling frequency in Hz violating the positive-finite contract.
    """
    audio = _impulse(1024)

    with pytest.raises(ValueError, match="sample_rate"):
        compute_joint_time_frequency_ot(audio, audio, sample_rate)


def test_joint_ot_non_channel_first_audio_raises_value_error() -> None:
    """One-dimensional waveforms violate the channel-first contract."""
    audio = np.zeros(_NUM_SAMPLES)

    with pytest.raises(ValueError, match="channel-first"):
        compute_joint_time_frequency_ot(audio, audio, _SAMPLE_RATE)


def test_joint_ot_mismatched_audio_shapes_raise_value_error() -> None:
    """The two audio tensors must have identical shapes."""
    target = _impulse(1024)
    pred = np.zeros((2, _NUM_SAMPLES))

    with pytest.raises(ValueError, match="same shape"):
        compute_joint_time_frequency_ot(target, pred, _SAMPLE_RATE)


def test_joint_ot_nonfinite_audio_raises_value_error() -> None:
    """Nonfinite sample values are rejected before the STFT."""
    target = _impulse(1024)
    pred = target.copy()
    pred[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_joint_time_frequency_ot(target, pred, _SAMPLE_RATE)


def test_joint_ot_complex_audio_raises_value_error() -> None:
    """Audio samples must be real-valued."""
    audio = _impulse(1024).astype(np.complex128)

    with pytest.raises(ValueError, match="real-valued"):
        compute_joint_time_frequency_ot(audio, audio, _SAMPLE_RATE)


def test_joint_ot_two_silent_signals_return_zero() -> None:
    """Two empty energy measures have explicit distance zero."""
    silence = np.zeros((1, _NUM_SAMPLES))

    assert compute_joint_time_frequency_ot(silence, silence, _SAMPLE_RATE) == 0.0


def test_joint_ot_one_silent_signal_raises_value_error() -> None:
    """Balanced transport is undefined with only one empty measure."""
    silence = np.zeros((1, _NUM_SAMPLES))

    with pytest.raises(ValueError, match="one input is silent"):
        compute_joint_time_frequency_ot(_impulse(1024), silence, _SAMPLE_RATE)
