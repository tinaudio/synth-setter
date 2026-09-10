"""Channel-aware audio losses cannot cancel or broadcast waveform channels."""

import torch

from synth_setter.models.components.audio_shape import canonical_audio


def test_canonical_audio_stereo_preserves_antiphase_channels() -> None:
    """Shape normalization cannot average away a nonzero stereo signal."""
    first = torch.linspace(-1.0, 1.0, 4096)
    audio = torch.stack((first, -first)).unsqueeze(0)

    torch.testing.assert_close(canonical_audio(audio), audio)
