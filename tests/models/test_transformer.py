"""Behavioral tests for transformer model components."""

import pytest
import torch

from synth_setter.models.components.transformer import (
    ASTWithProjectionHead,
    AudioSpectrogramTransformer,
    PatchEmbed,
)


def test_patch_embed_legacy_production_grid_preserves_checkpoint_token_count() -> None:
    """Keep the pre-fix token geometry used by existing checkpoints."""
    patch_embed = PatchEmbed(
        patch_size=16,
        stride=10,
        in_channels=1,
        d_model=3,
        spec_shape=(128, 401),
        use_fixed_ast_padding=False,
    )

    tokens = patch_embed(torch.randn(2, 1, 128, 401))

    assert tokens.shape == (2, 480, 3)


def test_patch_embed_default_production_grid_uses_corrected_token_count() -> None:
    """Default direct construction to corrected matching-axis padding."""
    patch_embed = PatchEmbed(
        patch_size=16,
        stride=10,
        in_channels=1,
        d_model=3,
        spec_shape=(128, 401),
    )

    tokens = patch_embed(torch.randn(2, 1, 128, 401))

    assert tokens.shape == (2, 520, 3)


def test_patch_embed_fixed_production_grid_includes_highest_mel_bins() -> None:
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


def test_patch_embed_short_time_grid_returns_tokens() -> None:
    """Pad a time axis shorter than one patch before convolution by default."""
    patch_embed = PatchEmbed(
        patch_size=16,
        stride=10,
        in_channels=1,
        d_model=3,
        spec_shape=(2500, 9),
    )

    tokens = patch_embed(torch.randn(2, 1, 2500, 9))

    assert tokens.shape == (2, 250, 3)


def test_audio_spectrogram_transformer_default_padding_uses_corrected_token_count() -> None:
    """Size default positional state for the corrected patch grid."""
    transformer = AudioSpectrogramTransformer(
        d_model=8,
        n_heads=2,
        n_layers=1,
        n_conditioning_outputs=2,
    )

    assert transformer.positional_encoding.pe.shape == (1, 522, 8)


def test_audio_spectrogram_transformer_default_forward_returns_conditioning_tokens() -> None:
    """Forward the default grid and return one token per conditioning output."""
    transformer = AudioSpectrogramTransformer(
        d_model=8,
        n_heads=2,
        n_layers=1,
        n_conditioning_outputs=2,
    )

    outputs = transformer(torch.randn(2, 2, 128, 401))

    assert outputs.shape == (2, 2, 8)


def test_ast_projection_head_legacy_padding_preserves_checkpoint_shape() -> None:
    """Keep projection-head positional state compatible with legacy checkpoints."""
    model = ASTWithProjectionHead(
        d_model=8, d_out=4, n_heads=2, n_layers=1, use_fixed_ast_padding=False
    )

    assert model.positional_encoding.pe.shape == (1, 481, 8)


def test_ast_projection_head_default_forward_returns_embedding() -> None:
    """Forward the default grid through the head to one vector per clip."""
    model = ASTWithProjectionHead(d_model=8, d_out=4, n_heads=2, n_layers=1)

    outputs = model(torch.randn(2, 2, 128, 401))

    assert outputs.shape == (2, 4)


def test_ast_corrected_padding_rejects_legacy_checkpoint_shape() -> None:
    """Reject incompatible positional state instead of silently resizing it."""
    legacy = ASTWithProjectionHead(
        d_model=8, d_out=4, n_heads=2, n_layers=1, use_fixed_ast_padding=False
    )
    corrected = ASTWithProjectionHead(d_model=8, d_out=4, n_heads=2, n_layers=1)

    with pytest.raises(RuntimeError, match="size mismatch for positional_encoding.pe"):
        corrected.load_state_dict(legacy.state_dict(), strict=True)


def test_ast_projection_head_default_padding_uses_corrected_checkpoint_shape() -> None:
    """Size default projection-head positional state for the corrected grid."""
    model = ASTWithProjectionHead(d_model=8, d_out=4, n_heads=2, n_layers=1)

    assert model.positional_encoding.pe.shape == (1, 521, 8)
