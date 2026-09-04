"""Behavior tests for parameter-token embedding inside the token transformer encoder."""

import torch

from synth_setter.models.components.transformer import (
    AudioSpectrogramTransformer,
    LearntProjection,
    ParamTokenEmbed,
)


def _param_token_embed(
    num_params: int = 7, num_tokens: int = 16, d_model: int = 32
) -> ParamTokenEmbed:
    projection = LearntProjection(
        d_model=d_model,
        d_token=d_model,
        num_params=num_params,
        num_tokens=num_tokens,
        initial_ffn=True,
        final_ffn=False,
    )
    return ParamTokenEmbed(projection=projection)


def test_param_token_embed_maps_flat_params_to_token_sequence() -> None:
    """Produce a finite (batch, num_tokens, d_model) sequence from flat parameters."""
    embed = _param_token_embed(num_params=7, num_tokens=16, d_model=32)

    tokens = embed(torch.rand(3, 7))

    assert embed.num_tokens == 16
    assert tokens.shape == (3, 16, 32)
    assert torch.isfinite(tokens).all()


def test_param_token_embed_freezes_decoder_half_and_trains_encoder_half() -> None:
    """Route gradients into the encoder half while the decoder half stays frozen."""
    embed = _param_token_embed()

    embed(torch.rand(2, 7)).sum().backward()

    assert not embed.projection.out_projection.requires_grad
    assert embed.projection.initial_ffn is not None
    encoder_side = [
        embed.projection.in_projection,
        embed.projection.assignment,
        *embed.projection.initial_ffn.parameters(),
    ]
    assert all(
        weight.grad is not None and torch.count_nonzero(weight.grad) for weight in encoder_side
    )


def test_param_token_embed_keeps_batch_rows_isolated() -> None:
    """Give one row's tokens zero input gradient with respect to any other row."""
    embed = _param_token_embed()
    params = torch.rand(3, 7, requires_grad=True)

    embed(params)[0].sum().backward()

    assert params.grad is not None
    assert torch.count_nonzero(params.grad[0])
    assert torch.all(params.grad[1:] == 0)


def test_param_token_embed_with_final_ffn_freezes_it_too() -> None:
    """Freeze a configured final FFN alongside the out projection."""
    projection = LearntProjection(
        d_model=32, d_token=32, num_params=7, num_tokens=16, initial_ffn=True, final_ffn=True
    )

    embed = ParamTokenEmbed(projection=projection)

    assert embed.projection.final_ffn is not None
    assert all(not weight.requires_grad for weight in embed.projection.final_ffn.parameters())


def test_token_transformer_with_param_embed_returns_conditioning_tokens() -> None:
    """Encode parameter vectors into class tokens through the injected embed."""
    encoder = AudioSpectrogramTransformer(
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_conditioning_outputs=1,
        token_embed=_param_token_embed(num_params=7, num_tokens=16, d_model=32),
    )

    encoded = encoder(torch.rand(3, 7))

    assert encoded.shape == (3, 1, 32)
    assert torch.isfinite(encoded).all()


def test_token_transformer_default_construction_still_encodes_spectrograms() -> None:
    """Keep the default patch-embed spectrogram path working without token_embed."""
    encoder = AudioSpectrogramTransformer(
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_conditioning_outputs=1,
    )

    encoded = encoder(torch.randn(2, 2, 128, 401))

    assert encoded.shape == (2, 1, 32)
