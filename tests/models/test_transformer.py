"""Behavioral tests for transformer model components."""

import pytest
import torch

from synth_setter.models.components.transformer import PatchEmbed


@pytest.mark.xfail(
    strict=True, reason="#3388 padding temporarily reverted for #3483 checkpoint validation"
)
def test_patch_embed_production_grid_includes_highest_mel_bins() -> None:
    """Include signal from the final mel bins in at least one patch token."""
    patch_embed = PatchEmbed(
        patch_size=16,
        stride=10,
        in_channels=1,
        d_model=1,
        spec_shape=(128, 401),
    )
    with torch.no_grad():
        patch_embed.projection.weight.fill_(1.0)
        assert patch_embed.projection.bias is not None
        patch_embed.projection.bias.zero_()
    spectrogram = torch.zeros(1, 1, 128, 401)
    spectrogram[:, :, 126:, :] = 1.0

    tokens = patch_embed(spectrogram)

    assert torch.count_nonzero(tokens) > 0


@pytest.mark.xfail(
    strict=True, reason="#3388 padding temporarily reverted for #3483 checkpoint validation"
)
def test_patch_embed_short_time_grid_returns_tokens() -> None:
    """Pad a time axis shorter than one patch before convolution."""
    patch_embed = PatchEmbed(
        patch_size=16,
        stride=10,
        in_channels=1,
        d_model=3,
        spec_shape=(2500, 9),
    )

    tokens = patch_embed(torch.randn(2, 1, 2500, 9))

    assert tokens.shape == (2, 250, 3)
