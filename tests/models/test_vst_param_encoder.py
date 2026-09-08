"""Parameter embeddings reuse the feed-forward AST backbone and projection head."""

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from synth_setter.data.vst.param_spec_registry import resolve_param_spec_width
from synth_setter.models.components.transformer import (
    ASTWithProjectionHead,
    LearntProjection,
    ParamTokenEmbed,
    PatchEmbed,
)
from synth_setter.models.components.vst_param_encoder import VSTFeedForwardParamEncoder
from synth_setter.utils.utils import register_resolvers


@pytest.mark.parametrize("batch_size", [1, 3])
def test_param_encoder_flat_batch_returns_finite_embeddings(batch_size: int) -> None:
    """Retain the default embedding width for singleton and multi-row batches.

    :param batch_size: Number of independent parameter vectors.
    """
    encoder = VSTFeedForwardParamEncoder(
        LearntProjection(16, 16, 7, 4, final_ffn=False),
        d_model=16,
        n_heads=2,
        n_layers=2,
    )

    output = encoder(torch.rand(batch_size, 7))

    assert output.shape == (batch_size, 512)
    assert torch.isfinite(output).all()


@pytest.fixture
def encoder() -> VSTFeedForwardParamEncoder:
    """Use a small real backbone with an unused decoder FFN.

    :returns: Train-mode encoder with a compact output head.
    """
    return VSTFeedForwardParamEncoder(
        LearntProjection(16, 16, 7, 4, final_ffn=True),
        d_model=16,
        d_out=8,
        n_heads=2,
        n_layers=2,
    )


def test_param_encoder_backward_reaches_every_trainable_weight(
    encoder: VSTFeedForwardParamEncoder,
) -> None:
    """Train the full encoder while leaving the decoder frozen.

    :param encoder: Small encoder with a configured decoder FFN.
    """
    params = torch.rand(3, 7, requires_grad=True)

    encoder(params).square().mean().backward()

    assert params.grad is not None
    assert torch.isfinite(params.grad).all()
    assert torch.count_nonzero(params.grad)
    for name, weight in encoder.named_parameters():
        if weight.requires_grad:
            assert weight.grad is not None, name
            assert torch.isfinite(weight.grad).all(), name
            assert torch.count_nonzero(weight.grad), name
        else:
            assert weight.grad is None, name
    assert isinstance(encoder.patch_embed, ParamTokenEmbed)
    projection = encoder.patch_embed.projection
    assert not projection.out_projection.requires_grad
    assert projection.final_ffn is not None
    assert not any(p.requires_grad for p in projection.final_ffn.parameters())


def test_param_encoder_single_row_loss_keeps_other_input_gradients_zero(
    encoder: VSTFeedForwardParamEncoder,
) -> None:
    """Backpropagate only into the example selected by the loss.

    :param encoder: Small train-mode encoder.
    """
    params = torch.rand(3, 7, requires_grad=True)

    encoder(params)[1].square().sum().backward()

    assert params.grad is not None
    assert torch.count_nonzero(params.grad[1])
    assert torch.count_nonzero(params.grad[[0, 2]]) == 0


def test_param_encoder_other_rows_changed_preserves_first_output(
    encoder: VSTFeedForwardParamEncoder,
) -> None:
    """Unrelated examples cannot alter another example's embedding.

    :param encoder: Small train-mode encoder.
    """
    params = torch.rand(3, 7)
    expected = encoder(params)[0]
    params[1:] += 10

    torch.testing.assert_close(encoder(params)[0], expected)


def test_param_encoder_shared_head_matches_injected_ast(
    encoder: VSTFeedForwardParamEncoder,
) -> None:
    """Load identical weights into the shared head and obtain identical embeddings.

    :param encoder: Parameter encoder supplying the reference checkpoint.
    """
    reference = ASTWithProjectionHead(
        d_model=16,
        d_out=8,
        n_heads=2,
        n_layers=2,
        token_embed=ParamTokenEmbed(LearntProjection(16, 16, 7, 4, final_ffn=True)),
    )
    reference.load_state_dict(encoder.state_dict(), strict=True)
    params = torch.rand(3, 7)

    torch.testing.assert_close(encoder(params), reference(params), rtol=0, atol=0)
    assert VSTFeedForwardParamEncoder.forward is ASTWithProjectionHead.forward


def test_ast_without_token_embed_strict_roundtrip_preserves_spectrogram_output() -> None:
    """The optional tokenizer leaves the default checkpoint namespace unchanged."""
    original = ASTWithProjectionHead(d_model=16, n_heads=2, n_layers=1)
    restored = ASTWithProjectionHead(d_model=16, n_heads=2, n_layers=1, token_embed=None)
    restored.load_state_dict(original.state_dict(), strict=True)
    audio = torch.rand(2, 2, 128, 401)

    assert isinstance(restored.patch_embed, PatchEmbed)
    assert "patch_embed.projection.weight" in restored.state_dict()
    torch.testing.assert_close(restored(audio), original(audio), rtol=0, atol=0)
    assert restored(audio).shape == (2, 16)


def test_param_encoder_composed_config_resolves_real_dimensions_and_runs() -> None:
    """Resolve the active synth width and exercise the production-sized shape graph."""
    register_resolvers()
    with initialize_config_module("synth_setter.configs", version_base="1.3"):
        cfg = compose(
            "train.yaml",
            overrides=[
                "model=vst_ffn",
                "model/encoder=vst_ff_param",
                "synth=obxf",
                "datamodule=surge_lance",
                "trainer=cpu",
            ],
        )
    assert cfg.model.encoder.projection.num_params == resolve_param_spec_width("obxf")
    assert cfg.model.encoder.projection.d_model == cfg.model.net.d_model == 768
    assert cfg.model.encoder.projection.d_token == 768
    assert cfg.model.encoder.projection.num_tokens == 128
    assert cfg.model.encoder.projection.initial_ffn is True
    assert cfg.model.encoder.projection.final_ffn is False
    assert cfg.model.encoder.n_heads == cfg.model.net.n_heads == 16
    assert cfg.model.encoder.n_layers == cfg.model.net.n_layers == 12
    assert cfg.model.net.d_out == cfg.model.encoder_output_dim == resolve_param_spec_width("obxf")
    with torch.device("meta"):
        model = instantiate(cfg.model.encoder)
        output = model(torch.empty(2, cfg.model.encoder.projection.num_params))
    assert output.shape == (2, 512)


def test_param_encoder_composed_small_backbone_trains_on_real_params() -> None:
    """Drive Hydra's real encoder through an optimizer step without model doubles."""
    register_resolvers()
    with initialize_config_module("synth_setter.configs", version_base="1.3"):
        cfg = compose(
            "train.yaml",
            overrides=[
                "model=vst_ffn",
                "model/encoder=vst_ff_param",
                "synth=obxf",
                "datamodule=surge_lance",
                "trainer=cpu",
                "model.encoder.d_model=16",
                "model.encoder.n_heads=2",
                "model.encoder.n_layers=1",
                "model.encoder.projection.num_tokens=4",
            ],
        )
    model = instantiate(cfg.model.encoder)
    params = torch.rand(2, resolve_param_spec_width("obxf"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = model(params).square().mean()
    initial.backward()
    optimizer.step()

    assert model(params).shape == (2, 512)
    assert model(params).square().mean() < initial
