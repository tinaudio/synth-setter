"""Sketch-control extraction dispatches on the checkpoint's profile."""

import numpy as np
import pytest
import torch

from synth_setter.conditioning import SketchControlSpec
from synth_setter.features import profile_controls
from synth_setter.features.profile_controls import extract_profile_controls
from synth_setter.features.pyfdn_controls import extract_reverb_sketch

_SAMPLE_RATE = 44_100


def _decaying_noise(seed: int, channels: int = 1) -> np.ndarray:
    """Return a synthetic exponentially decaying noise burst that resembles a reverb tail.

    :param seed: Generator seed.
    :param channels: Channel count.
    :returns: Channel-first float32 audio of four seconds.
    """
    rng = np.random.default_rng(seed)
    envelope = np.exp(-np.arange(4 * _SAMPLE_RATE) / (0.4 * _SAMPLE_RATE))
    return (rng.standard_normal((channels, envelope.size)) * envelope).astype(np.float32)


def test_reverb_profile_returns_reverb_sketch_on_control_grid() -> None:
    """The pyFDN profile feeds the mono waveform straight to the reverb-sketch extractor."""
    audio = _decaying_noise(3)
    spec = SketchControlSpec(
        profile="pyfdn_reverb", column="pyfdn_sketch", num_frames=32, num_control_tokens=32
    )
    controls = extract_profile_controls(audio, _SAMPLE_RATE, spec)
    assert controls.shape == (1, 10, 32)
    assert controls.dtype is torch.float32
    np.testing.assert_array_equal(
        controls[0].numpy(), extract_reverb_sketch(audio[0], _SAMPLE_RATE)
    )


def test_reverb_profile_multichannel_audio_rejected() -> None:
    """A stereo file cannot be silently downmixed into an impulse response."""
    spec = SketchControlSpec(
        profile="pyfdn_reverb", column="pyfdn_sketch", num_frames=32, num_control_tokens=32
    )
    with pytest.raises(ValueError, match="mono"):
        extract_profile_controls(_decaying_noise(3, channels=2), _SAMPLE_RATE, spec)


def test_music_profile_pools_tracks_and_zeros_weak_pitch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The music profile keeps the production pooling and pitch zero-binning.

    :param monkeypatch: Replaces the PESTO-backed extractor with a fixed control matrix.
    """
    controls = torch.full((386, 401), 0.05)
    controls[0] = 0.25
    controls[1] = 0.5
    controls[2, 0] = 0.1
    controls[2, 13] = 0.11
    monkeypatch.setattr(profile_controls, "extract_sketch_controls", lambda *args: controls)
    spec = SketchControlSpec(num_frames=32, pitch_zero_threshold=0.1)
    pooled = extract_profile_controls(np.zeros((2, 176400), np.float32), _SAMPLE_RATE, spec)
    assert pooled.shape == (1, 386, 32)
    assert pooled.dtype is torch.float32
    assert torch.equal(pooled[:, 0], torch.full((1, 32), 0.25))
    assert torch.equal(pooled[:, 1], torch.full((1, 32), 0.5))
    assert pooled[0, 2, 0] == pytest.approx(0.1)
    assert pooled[0, 2, 1] == pytest.approx(0.11)
    assert torch.count_nonzero(pooled[:, 2:, 2:]) == 0


def test_tiv_profile_rejected() -> None:
    """Profiles without an offline extractor fail loudly instead of feeding zeros."""
    spec = SketchControlSpec(
        profile="tiv", column="tiv", source="online", sample_rate=_SAMPLE_RATE, num_frames=1
    )
    with pytest.raises(ValueError, match="tiv"):
        extract_profile_controls(_decaying_noise(3), _SAMPLE_RATE, spec)
