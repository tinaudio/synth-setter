"""Channel-aware audio losses cannot cancel or broadcast waveform channels."""

import torch

from synth_setter.models.components.audio_feedback import canonical_target_audio


def test_canonical_audio_stereo_preserves_antiphase_channels() -> None:
    """Shape normalization cannot average away a nonzero stereo signal."""
    first = torch.linspace(-1.0, 1.0, 4096)
    audio = torch.stack((first, -first)).unsqueeze(0)

    torch.testing.assert_close(canonical_target_audio(audio), audio)
