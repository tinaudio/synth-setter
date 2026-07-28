"""Behavioral tests for time-varying sketch control extraction."""

import pytest
import torch

from synth_setter.data.vst.shapes import (
    NUM_SKETCH_CONTROLS,
    SKETCH_CONTROL_FIELD,
    mel_n_frames,
)
from synth_setter.features.sketch_controls import (
    extract_sketch_controls,
    loudness_track,
    pitch_track,
    spectral_centroid_track,
)

_SAMPLE_RATE = 44100
_DURATION_SECONDS = 1.0


def _sine(frequency_hz: float, amplitude: float = 0.5) -> torch.Tensor:
    """Build a stereo sine test signal.

    :param frequency_hz: Sine frequency in Hz.
    :param amplitude: Peak amplitude.
    :returns: Audio of shape ``(2, samples)``.
    """
    t = torch.arange(int(_SAMPLE_RATE * _DURATION_SECONDS)) / _SAMPLE_RATE
    mono = amplitude * torch.sin(2 * torch.pi * frequency_hz * t)
    return mono.expand(2, -1)


def _n_frames() -> int:
    return mel_n_frames(_SAMPLE_RATE, _DURATION_SECONDS)


def test_sketch_control_field_constants_match_contract() -> None:
    """The stored-column name and control count are fixed by the batch contract."""
    assert SKETCH_CONTROL_FIELD == "sketch_ctrl"
    assert NUM_SKETCH_CONTROLS == 3


def test_loudness_track_sine_matches_mel_frame_grid() -> None:
    """Loudness lands on the 100 fps mel frame grid in [0, 1]."""
    track = loudness_track(_sine(440.0), _SAMPLE_RATE)

    assert track.shape == (_n_frames(),)
    assert torch.isfinite(track).all()
    assert (track >= 0).all() and (track <= 1).all()


def test_loudness_track_silence_maps_to_zero() -> None:
    """Digital silence hits the -80 dB floor, i.e. 0 after (db + 80) / 80."""
    track = loudness_track(torch.zeros(2, _SAMPLE_RATE), _SAMPLE_RATE)

    assert torch.allclose(track, torch.zeros_like(track), atol=1e-6)


def test_spectral_centroid_track_higher_sine_reads_higher() -> None:
    """A 4 kHz sine yields a strictly higher centroid track than a 200 Hz sine."""
    low = spectral_centroid_track(_sine(200.0), _SAMPLE_RATE)
    high = spectral_centroid_track(_sine(4000.0), _SAMPLE_RATE)

    assert low.shape == (_n_frames(),)
    assert torch.isfinite(low).all() and torch.isfinite(high).all()
    assert high.mean() > low.mean()


def test_spectral_centroid_track_silence_is_finite_and_bounded() -> None:
    """Silent frames (0/0 centroid) still produce finite values in [0, 1]."""
    track = spectral_centroid_track(torch.zeros(2, _SAMPLE_RATE), _SAMPLE_RATE)

    assert torch.isfinite(track).all()
    assert (track >= 0).all() and (track <= 1).all()


@pytest.mark.slow
def test_pitch_track_440hz_sine_reads_midi_69() -> None:
    """PESTO reads a 440 Hz sine as MIDI 69, i.e. 69/127 after scaling."""
    track = pitch_track(_sine(440.0), _SAMPLE_RATE)

    assert track.shape == (_n_frames(),)
    assert torch.isfinite(track).all()
    # Edge frames see partial windows; assert on the steady-state interior.
    interior = track[10:-10]
    assert torch.allclose(interior, torch.full_like(interior, 69.0 / 127.0), atol=0.01)


@pytest.mark.slow
def test_extract_sketch_controls_sine_stacks_three_tracks_in_minus_one_one() -> None:
    """The stacked control matrix is (3, n_frames) float32 within [-1, 1]."""
    controls = extract_sketch_controls(_sine(440.0), _SAMPLE_RATE)

    assert controls.shape == (NUM_SKETCH_CONTROLS, _n_frames())
    assert controls.dtype == torch.float32
    assert torch.isfinite(controls).all()
    assert (controls >= -1).all() and (controls <= 1).all()
    pitch_row = controls[2, 10:-10]
    expected = 2.0 * (69.0 / 127.0) - 1.0
    assert torch.allclose(pitch_row, torch.full_like(pitch_row, expected), atol=0.02)


@pytest.mark.slow
def test_extract_sketch_controls_silence_loudness_row_is_minus_one() -> None:
    """Silence maps the loudness row to the -1 end of the [-1, 1] range."""
    controls = extract_sketch_controls(torch.zeros(2, _SAMPLE_RATE), _SAMPLE_RATE)

    assert torch.allclose(controls[0], torch.full_like(controls[0], -1.0), atol=1e-5)
