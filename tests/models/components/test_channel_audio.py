"""Channel-aware audio losses cannot cancel or broadcast waveform channels."""

import pytest
import torch

from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
from synth_setter.models.components.audio_feedback import (
    canonical_target_audio,
    channelwise_distance,
)


def test_canonical_audio_stereo_preserves_antiphase_channels() -> None:
    """Shape normalization cannot average away a nonzero stereo signal."""
    first = torch.linspace(-1.0, 1.0, 4096)
    audio = torch.stack((first, -first)).unsqueeze(0)

    torch.testing.assert_close(canonical_target_audio(audio), audio)


def test_channelwise_distance_antiphase_target_has_nonzero_loss() -> None:
    """Opposite channels are compared separately before losses are reduced."""
    first = torch.sin(torch.arange(4096) * 0.1)
    target = torch.stack((first, -first)).unsqueeze(0)
    rendered = torch.zeros_like(target)
    distance = MultiScaleSpectralDistance(sample_rate=44_100)

    assert channelwise_distance(distance, rendered, target).item() > 0.0


def test_channelwise_distance_channel_count_mismatch_rejected() -> None:
    """Broadcasting a single waveform across multiple target channels is not matching."""
    distance = MultiScaleSpectralDistance(sample_rate=44_100)

    with pytest.raises(ValueError, match="shape"):
        channelwise_distance(distance, torch.zeros(1, 4096), torch.ones(1, 2, 4096))


def test_channelwise_distance_second_channel_gradient_stays_on_that_channel() -> None:
    """A channel-specific error cannot disappear into a mean waveform."""
    first = torch.sin(torch.arange(4096) * 0.1)
    target = torch.stack((first, -first)).unsqueeze(0)
    rendered = target.clone()
    rendered[:, 1] *= 0.5
    rendered.requires_grad_()
    distance = MultiScaleSpectralDistance(sample_rate=44_100)

    channelwise_distance(distance, rendered, target).sum().backward()

    assert rendered.grad is not None
    assert torch.count_nonzero(rendered.grad[:, 0]) == 0
    assert rendered.grad[:, 1].abs().sum() > 0
