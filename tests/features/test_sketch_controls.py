"""Behavioral tests for sketch-control extraction (loudness, centroid, pitch)."""

import numpy as np
import pytest
import torch

from synth_setter.data.vst.shapes import mel_hop_length
from synth_setter.features.sketch_controls import (
    NUM_SKETCH_CONTROLS,
    extract_sketch_controls,
    extract_sketch_controls_batch,
    loudness_track,
    pitch_track,
    sketch_num_frames,
    spectral_centroid_track,
)

_SAMPLE_RATE = 44100
_DURATION_S = 1.0


def _sine(freq_hz: float, amplitude: float = 0.5, channels: int = 2) -> torch.Tensor:
    """Build a fixed-length test sine.

    :param freq_hz: Oscillation frequency.
    :param amplitude: Peak amplitude.
    :param channels: Channel count.
    :returns: ``(channels, T)`` float32 waveform.
    """
    t = torch.arange(int(_SAMPLE_RATE * _DURATION_S)) / _SAMPLE_RATE
    wave = amplitude * torch.sin(2 * torch.pi * freq_hz * t)
    return wave.expand(channels, -1).to(torch.float32)


def test_sketch_num_frames_one_second_matches_mel_grid() -> None:
    """Frame counts agree with the mel hop grid."""
    samples = int(_SAMPLE_RATE * _DURATION_S)
    assert sketch_num_frames(samples, _SAMPLE_RATE) == samples // mel_hop_length(_SAMPLE_RATE) + 1


def test_extract_sketch_controls_stereo_sine_returns_frame_grid_shape() -> None:
    """Extraction lands on the (controls, frames) grid as float32."""
    audio = _sine(440.0)
    controls = extract_sketch_controls(audio, _SAMPLE_RATE)
    expected_frames = sketch_num_frames(audio.shape[-1], _SAMPLE_RATE)
    assert controls.shape == (NUM_SKETCH_CONTROLS, expected_frames)
    assert controls.dtype == torch.float32


def test_extract_sketch_controls_sine_stays_within_signed_unit_bounds() -> None:
    """Every control value is finite and within [-1, 1]."""
    controls = extract_sketch_controls(_sine(440.0), _SAMPLE_RATE)
    assert torch.isfinite(controls).all()
    assert controls.min() >= -1.0
    assert controls.max() <= 1.0


def test_loudness_track_silence_returns_floor() -> None:
    """Silence maps to the -1 loudness floor."""
    silence = torch.zeros(2, int(_SAMPLE_RATE * _DURATION_S))
    loudness = loudness_track(silence, _SAMPLE_RATE)
    assert torch.allclose(loudness, torch.full_like(loudness, -1.0))


def test_loudness_track_louder_sine_exceeds_quieter_sine() -> None:
    """Loudness orders correctly with amplitude."""
    loud = loudness_track(_sine(440.0, amplitude=0.9), _SAMPLE_RATE)
    quiet = loudness_track(_sine(440.0, amplitude=0.05), _SAMPLE_RATE)
    assert loud.mean() > quiet.mean()


def test_loudness_track_half_amplitude_sine_matches_expected_db() -> None:
    """A 0.5-amplitude sine lands at its analytic dB value."""
    # RMS of a 0.5-amplitude sine is 0.5/sqrt(2) = -9.03 dBFS -> (80 - 9.03)/80 * 2 - 1.
    loudness = loudness_track(_sine(440.0, amplitude=0.5), _SAMPLE_RATE)
    interior = loudness[2:-2]
    assert interior.mean() == pytest.approx(0.774, abs=0.01)


def test_spectral_centroid_track_high_sine_exceeds_low_sine() -> None:
    """Centroid orders correctly with frequency."""
    high = spectral_centroid_track(_sine(4000.0), _SAMPLE_RATE)
    low = spectral_centroid_track(_sine(200.0), _SAMPLE_RATE)
    assert high[2:-2].mean() > low[2:-2].mean()


def test_spectral_centroid_track_sine_440_near_midi_69() -> None:
    """A 440 Hz sine centers near MIDI 69 on the normalized scale."""
    centroid = spectral_centroid_track(_sine(440.0), _SAMPLE_RATE)
    # MIDI 69 / 127 mapped to [-1, 1]; windowing broadens the mass, so a loose tolerance.
    assert centroid[2:-2].mean() == pytest.approx(69.0 / 127.0 * 2.0 - 1.0, abs=0.1)


def test_spectral_centroid_track_silence_returns_finite_floor() -> None:
    """Silent frames map to the finite MIDI floor, never NaN."""
    silence = torch.zeros(2, int(_SAMPLE_RATE * _DURATION_S))
    centroid = spectral_centroid_track(silence, _SAMPLE_RATE)
    assert torch.isfinite(centroid).all()
    assert torch.allclose(centroid, torch.full_like(centroid, -1.0))


def test_pitch_track_sine_440_near_midi_69() -> None:
    """PESTO tracks a 440 Hz sine to MIDI 69 on the frame grid."""
    pitch = pitch_track(_sine(440.0), _SAMPLE_RATE)
    assert pitch.shape == (sketch_num_frames(int(_SAMPLE_RATE * _DURATION_S), _SAMPLE_RATE),)
    assert pitch[5:-5].mean() == pytest.approx(69.0 / 127.0 * 2.0 - 1.0, abs=0.05)


def test_extract_sketch_controls_batch_rows_match_single_clip_extraction() -> None:
    """Batch rows equal independent single-clip extraction."""
    clips = torch.stack([_sine(440.0), _sine(880.0, amplitude=0.25)])
    batch = extract_sketch_controls_batch(clips, _SAMPLE_RATE)
    single = extract_sketch_controls(clips[1], _SAMPLE_RATE)
    assert batch.shape[0] == 2
    assert np.allclose(batch[1].numpy(), single.numpy(), atol=1e-5)
