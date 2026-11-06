"""Sketch-control extraction dispatches on the checkpoint's profile."""

import numpy as np
import pytest
import torch

from synth_setter.conditioning import SketchControlSpec
from synth_setter.features import profile_controls
from synth_setter.features.profile_controls import extract_profile_controls
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.sketch import prepare_music_sketch_controls

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
    # Pooling 401 source frames to 32 averages source 0-12 into output 0 and 12-25 into
    # output 1, so the spikes are sized by window mean, clear of the threshold either way.
    controls[2, :26] = 0.0
    controls[2, 0] = 0.12 * 13
    controls[2, 13] = 0.08 * 14
    monkeypatch.setattr(profile_controls, "extract_sketch_controls", lambda *args: controls)
    spec = SketchControlSpec(num_frames=32, pitch_zero_threshold=0.1)
    pooled = extract_profile_controls(np.zeros((2, 176400), np.float32), _SAMPLE_RATE, spec)
    assert pooled.shape == (1, 386, 32)
    assert pooled.dtype is torch.float32
    assert torch.equal(pooled[:, 0], torch.full((1, 32), 0.25))
    assert torch.equal(pooled[:, 1], torch.full((1, 32), 0.5))
    assert pooled[0, 2, 0] == pytest.approx(0.12)
    assert pooled[0, 2, 1] == 0.0
    assert torch.count_nonzero(pooled[:, 2:, 2:]) == 0


def test_music_preparation_thresholds_after_frame_resampling() -> None:
    """Pitch thresholding applies on the configured frame grid before tokenization."""
    controls = torch.zeros(1, 386, 5)
    controls[0, 2, 2] = 0.3
    spec = SketchControlSpec(
        num_frames=2,
        num_control_tokens=1,
        pitch_zero_threshold=0.15,
    )

    prepared = prepare_music_sketch_controls(controls, spec)

    assert prepared.dtype is torch.float32
    assert prepared.shape == (1, 386, 2)
    assert torch.count_nonzero(prepared) == 0


def test_music_preparation_sparser_extraction_than_frame_grid_rejected() -> None:
    """A frame grid denser than the extractor emits is refused instead of fabricated."""
    controls = torch.zeros(1, 386, 3)
    spec = SketchControlSpec(num_frames=4, num_control_tokens=2)

    with pytest.raises(ValueError, match="at least 4 frames"):
        prepare_music_sketch_controls(controls, spec)


def test_music_preparation_returns_float32_independent_storage() -> None:
    """Preparation normalizes dtype without retaining writable source storage."""
    controls = torch.ones(1, 386, 2, dtype=torch.float64)
    spec = SketchControlSpec(num_frames=2, num_control_tokens=2)

    prepared = prepare_music_sketch_controls(controls, spec)
    prepared.zero_()

    assert controls.dtype is torch.float64
    assert torch.count_nonzero(controls) == controls.numel()


def test_music_preparation_output_supports_trainable_projection_backward() -> None:
    """Prepared tensors remain usable by trainable projections under autograd."""
    controls = torch.ones(1, 386, 4, requires_grad=True)
    spec = SketchControlSpec(num_frames=2, num_control_tokens=2)
    projection = torch.nn.Linear(386, 1, bias=False)

    prepared = prepare_music_sketch_controls(controls, spec)
    projection(prepared.transpose(1, 2)).sum().backward()

    assert projection.weight.grad is not None
    assert controls.grad is not None


@pytest.mark.gpu
def test_music_preparation_preserves_cuda_device() -> None:
    """Preparation keeps extracted controls on their source device."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    controls = torch.ones(1, 386, 4, device="cuda")
    spec = SketchControlSpec(num_frames=2, num_control_tokens=2)

    prepared = prepare_music_sketch_controls(controls, spec)

    assert prepared.device.type == "cuda"


def test_tiv_profile_rejected() -> None:
    """Profiles without an offline extractor fail loudly instead of feeding zeros."""
    spec = SketchControlSpec(
        profile="tiv", column="tiv", source="online", sample_rate=_SAMPLE_RATE, num_frames=1
    )
    with pytest.raises(ValueError, match="tiv"):
        extract_profile_controls(_decaying_noise(3), _SAMPLE_RATE, spec)
