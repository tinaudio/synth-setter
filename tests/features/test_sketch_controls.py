"""Behavioral tests for sketch-control extraction (loudness, centroid, pitch)."""

import numpy as np
import pytest
import torch

from synth_setter.data.vst.shapes import mel_hop_length
from synth_setter.features.sketch_controls import (
    NUM_SKETCH_CONTROLS,
    SKETCH_CENTROID_ROW,
    SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_BINS,
    SKETCH_PITCH_SLICE,
    extract_sketch_controls,
    extract_sketch_controls_batch,
    loudness_track,
    pitch_track,
    sketch_num_frames,
    spectral_centroid_track,
)

_SAMPLE_RATE = 44100
_DURATION_S = 1.0
# PESTO bins are 3 per semitone, so MIDI 69 (440 Hz) peaks at bin 69 * 3.
_A4_PITCH_BIN = 207


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


def _silence() -> torch.Tensor:
    """Build a fixed-length silent clip.

    :returns: ``(2, T)`` zero waveform.
    """
    return torch.zeros(2, int(_SAMPLE_RATE * _DURATION_S))


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
    loudness = loudness_track(_silence(), _SAMPLE_RATE)
    assert torch.allclose(loudness, torch.full_like(loudness, -1.0), atol=0.02)


def test_loudness_track_louder_sine_exceeds_quieter_sine() -> None:
    """Loudness orders correctly with amplitude."""
    loud = loudness_track(_sine(1000.0, amplitude=0.9), _SAMPLE_RATE)
    quiet = loudness_track(_sine(1000.0, amplitude=0.05), _SAMPLE_RATE)
    assert loud.mean() > quiet.mean()


def test_loudness_track_crescendo_increases_within_clip() -> None:
    """A linear amplitude ramp yields a rising loudness contour."""
    t = torch.arange(int(_SAMPLE_RATE * _DURATION_S)) / _SAMPLE_RATE
    ramp = (t / _DURATION_S) * torch.sin(2 * torch.pi * 1000.0 * t)
    loudness = loudness_track(ramp.expand(2, -1).to(torch.float32), _SAMPLE_RATE)
    early = loudness[5:25].mean()
    late = loudness[-25:-5].mean()
    assert late > early


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
    centroid = spectral_centroid_track(_silence(), _SAMPLE_RATE)
    assert torch.isfinite(centroid).all()
    assert torch.allclose(centroid, torch.full_like(centroid, -1.0))


def test_pitch_track_sine_440_peaks_at_a4_bin() -> None:
    """PESTO activations for a 440 Hz sine peak at the MIDI-69 bin."""
    activations = pitch_track(_sine(440.0), _SAMPLE_RATE)
    frames = sketch_num_frames(int(_SAMPLE_RATE * _DURATION_S), _SAMPLE_RATE)
    assert activations.shape == (SKETCH_PITCH_BINS, frames)
    interior = activations[:, 5:-5]
    assert interior.argmax(dim=0).float().mean() == pytest.approx(_A4_PITCH_BIN, abs=1.5)
    assert interior.max() > 0.5


def test_pitch_track_silence_stays_below_voiced_threshold() -> None:
    """Silence produces activations below Sketch2Sound's 0.1 zero-bin cut."""
    activations = pitch_track(_silence(), _SAMPLE_RATE)
    assert activations.max() < 0.1


def test_extract_sketch_controls_row_layout_matches_slices() -> None:
    """The stacked rows equal the per-track outputs at the declared slices."""
    audio = _sine(440.0)
    controls = extract_sketch_controls(audio, _SAMPLE_RATE)
    assert torch.allclose(
        controls[SKETCH_LOUDNESS_ROW], loudness_track(audio, _SAMPLE_RATE), atol=1e-5
    )
    assert torch.allclose(
        controls[SKETCH_CENTROID_ROW], spectral_centroid_track(audio, _SAMPLE_RATE), atol=1e-5
    )
    assert torch.allclose(
        controls[SKETCH_PITCH_SLICE], pitch_track(audio, _SAMPLE_RATE), atol=1e-5
    )


def test_extract_sketch_controls_batch_rows_match_single_clip_extraction() -> None:
    """Batch rows equal independent single-clip extraction."""
    clips = torch.stack([_sine(440.0), _sine(880.0, amplitude=0.25)])
    batch = extract_sketch_controls_batch(clips, _SAMPLE_RATE)
    single = extract_sketch_controls(clips[1], _SAMPLE_RATE)
    assert batch.shape[0] == 2
    assert np.allclose(batch[1].numpy(), single.numpy(), atol=1e-5)


def test_extract_sketch_controls_repeat_calls_are_deterministic() -> None:
    """Two extractions of the same clip are bitwise identical."""
    audio = _sine(440.0)
    first = extract_sketch_controls(audio, _SAMPLE_RATE)
    second = extract_sketch_controls(audio, _SAMPLE_RATE)
    assert torch.equal(first, second)
