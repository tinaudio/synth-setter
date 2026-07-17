"""Behavioral tests for the pretrained-AST conditioning encoder bridge.

Tiny random backbones keep every test offline while exercising the real HF model.
"""

from __future__ import annotations

import pytest
import torch
from transformers import ASTConfig, ASTModel

from synth_setter.models.components.pretrained_ast import (
    PretrainedASTEncoder,
    interpolate_time_position_embeddings,
)

TINY_BACKBONE = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
}


def _tiny_encoder(**overrides: object) -> PretrainedASTEncoder:
    """Build an offline encoder over a tiny backbone.

    :param **overrides: Constructor arguments replacing the tiny defaults.
    :returns: Encoder ready for CPU forward passes.
    """
    kwargs: dict[str, object] = {
        "d_model": 32,
        "n_conditioning_outputs": 8,
        "n_pool_heads": 2,
        "pretrained": False,
        "backbone_config": TINY_BACKBONE,
    }
    kwargs.update(overrides)
    return PretrainedASTEncoder(**kwargs)  # type: ignore[arg-type]


class TestForwardContract:
    """The flow module's mel-to-conditioning contract."""

    def test_forward_stereo_mel_returns_one_token_per_conditioning_output(self) -> None:
        """A stereo mel batch yields finite float32 conditioning tokens."""
        encoder = _tiny_encoder()

        out = encoder(torch.randn(2, 2, 128, 401))

        assert out.shape == (2, 8, 32)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_forward_respects_configured_conditioning_output_count(self) -> None:
        """The configured conditioning count controls the output token count."""
        encoder = _tiny_encoder(n_conditioning_outputs=3)

        out = encoder(torch.randn(1, 2, 128, 401))

        assert out.shape == (1, 3, 32)

    def test_forward_channel_mix_uses_stereo_mean(self) -> None:
        """Stereo pairs with the same mean produce identical conditioning."""
        encoder = _tiny_encoder().eval()
        left_heavy = torch.stack((torch.ones(128, 401), torch.full((128, 401), 3.0)))
        centered = torch.full((2, 128, 401), 2.0)

        with torch.no_grad():
            mixed = encoder(torch.stack((left_heavy, centered)))

        torch.testing.assert_close(mixed[0], mixed[1], rtol=1e-5, atol=1e-6)

    def test_forward_is_trainable_end_to_end_by_default(self) -> None:
        """Unfrozen construction backpropagates through the backbone."""
        encoder = _tiny_encoder()

        encoder(torch.randn(1, 2, 128, 401)).sum().backward()

        backbone_grads = [p.grad for p in encoder.backbone.parameters() if p.requires_grad]
        assert backbone_grads and all(g is not None for g in backbone_grads)


class TestFreeze:
    """Backbone freezing leaves the adaptation layers trainable."""

    def test_freeze_stops_backbone_grads_but_trains_bridge(self) -> None:
        """Frozen backbone parameters stop gradients without freezing the bridge."""
        encoder = _tiny_encoder(freeze=True)

        encoder(torch.randn(1, 2, 128, 401)).sum().backward()

        assert all(not p.requires_grad for p in encoder.backbone.parameters())
        assert encoder.queries.grad is not None
        assert encoder.input_scale.grad is not None


class TestOfflineConstruction:
    """Random initialization must remain network-independent."""

    def test_pretrained_false_never_calls_from_pretrained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offline construction must not touch the checkpoint loader.

        :param monkeypatch: Replaces ``from_pretrained`` with a tripwire.
        """

        def _forbid(*args: object, **kwargs: object) -> None:
            raise AssertionError("from_pretrained called during offline construction")

        monkeypatch.setattr(ASTModel, "from_pretrained", _forbid)

        _tiny_encoder()

    @pytest.mark.parametrize("backbone_config", [{}, {"hidden_size": 32}])
    def test_pretrained_true_with_backbone_config_raises_before_download(
        self, monkeypatch: pytest.MonkeyPatch, backbone_config: dict[str, int]
    ) -> None:
        """Checkpoint mode rejects offline-only overrides before network access.

        :param monkeypatch: Replaces ``from_pretrained`` with a tripwire.
        :param backbone_config: Explicit offline override, empty or populated.
        """

        def _forbid(*args: object, **kwargs: object) -> None:
            raise AssertionError("checkpoint download attempted")

        monkeypatch.setattr(ASTModel, "from_pretrained", _forbid)

        with pytest.raises(ValueError, match="backbone_config requires pretrained=False"):
            PretrainedASTEncoder(backbone_config=backbone_config)

    @pytest.mark.parametrize("reserved_key", ["max_length", "num_mel_bins"])
    def test_offline_reserved_geometry_override_raises(self, reserved_key: str) -> None:
        """Offline overrides cannot conflict with geometry derived from spec_shape.

        :param reserved_key: ``ASTConfig`` key owned by ``spec_shape``.
        """
        config = {**TINY_BACKBONE, reserved_key: 1}

        with pytest.raises(ValueError, match=rf"{reserved_key}.*spec_shape"):
            _tiny_encoder(backbone_config=config)

    def test_d_model_mismatching_backbone_hidden_size_raises(self) -> None:
        """A backbone-width mismatch fails before training."""
        with pytest.raises(ValueError, match="d_model"):
            _tiny_encoder(d_model=64)


class TestPositionEmbeddingInterpolation:
    """Checkpoint position grids adapt to the configured frame count."""

    def _tiny_backbone(self, max_length: int) -> ASTModel:
        """Build a tiny ASTModel for the requested frame count.

        :param max_length: Input frame count represented by the position grid.
        :returns: Randomly initialized tiny backbone.
        """
        return ASTModel(
            ASTConfig(
                num_mel_bins=128,
                max_length=max_length,
                patch_size=16,
                frequency_stride=10,
                time_stride=10,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
            )
        )

    def test_interpolation_resizes_grid_and_updates_config(self) -> None:
        """The resized patch grid and backbone config remain consistent."""
        backbone = self._tiny_backbone(max_length=1024)

        interpolate_time_position_embeddings(backbone, max_length=401)

        f_dim, t_dim = backbone.embeddings.get_shape(backbone.config)
        assert backbone.config.max_length == 401
        assert (f_dim, t_dim) == (12, 39)
        assert backbone.embeddings.position_embeddings.shape == (1, f_dim * t_dim + 2, 32)

    def test_interpolation_preserves_special_tokens_and_constant_fields(self) -> None:
        """Interpolation preserves special tokens and constant patch fields."""
        backbone = self._tiny_backbone(max_length=1024)
        with torch.no_grad():
            backbone.embeddings.position_embeddings[:, :2] = torch.randn(1, 2, 32)
            backbone.embeddings.position_embeddings[:, 2:] = 0.5
        special_before = backbone.embeddings.position_embeddings[:, :2].clone()

        interpolate_time_position_embeddings(backbone, max_length=401)

        pos = backbone.embeddings.position_embeddings
        assert torch.equal(pos[:, :2], special_before)
        assert torch.allclose(pos[:, 2:], torch.full_like(pos[:, 2:], 0.5))

    def test_interpolated_backbone_accepts_the_new_frame_count(self) -> None:
        """The adapted backbone accepts the target frame count end to end."""
        backbone = self._tiny_backbone(max_length=1024)

        interpolate_time_position_embeddings(backbone, max_length=401)
        out = backbone(input_values=torch.randn(1, 401, 128))

        assert out.last_hidden_state.shape[0] == 1
