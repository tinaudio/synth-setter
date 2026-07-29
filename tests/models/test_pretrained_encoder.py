"""Behavior tests for frozen online CLAP conditioning."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import ClapFeatureExtractor

from synth_setter.models.components.pretrained_encoder import (
    ClapAudioEncoder,
    PretrainedConditioningEncoder,
)
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.components.vector_projection import VectorProjection
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.pipeline.data.add_embeddings import (
    DEFAULT_CLAP_CHECKPOINT,
    _resolve_clap_checkpoint,
)

_SAMPLE_RATE = 48_000
_PROJECTION_DIM = 8
_TINY_CLAP_CONFIG: dict[str, Any] = {
    "projection_dim": _PROJECTION_DIM,
    "text_config": {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "max_position_embeddings": 16,
        "projection_dim": _PROJECTION_DIM,
    },
    "audio_config": {
        "num_mel_bins": 64,
        "spec_size": 256,
        "hidden_size": 32,
        "projection_dim": _PROJECTION_DIM,
        "depths": [1, 1, 1, 1],
        "num_attention_heads": [1, 2, 4, 8],
        "patch_embeds_hidden_size": 4,
        "patch_size": 4,
        "patch_stride": [4, 4],
        "num_classes": 8,
        "window_size": 4,
    },
}


@pytest.fixture(scope="module")
def clap_checkpoint() -> str:
    """Materialize the mirrored production checkpoint in the shared local cache.

    :returns: Local CLAP checkpoint directory.
    """
    return _resolve_clap_checkpoint(DEFAULT_CLAP_CHECKPOINT)


@pytest.fixture(scope="module")
def clap_encoder(clap_checkpoint: str) -> ClapAudioEncoder:
    """Build a small random CLAP network with the production feature geometry.

    :param clap_checkpoint: Local R2-mirrored feature-extractor checkpoint.
    :returns: Frozen waveform encoder suitable for CPU behavior tests.
    """
    return ClapAudioEncoder(
        sample_rate=_SAMPLE_RATE,
        checkpoint=clap_checkpoint,
        pretrained=False,
        backbone_config=_TINY_CLAP_CONFIG,
    )


def test_clap_features_match_huggingface_short_audio_path(
    clap_encoder: ClapAudioEncoder, clap_checkpoint: str
) -> None:
    """Online torch features agree with the extractor used for stored CLAP vectors.

    :param clap_encoder: Differentiable online encoder under test.
    :param clap_checkpoint: Local R2-mirrored reference extractor checkpoint.
    """
    audio = torch.sin(torch.arange(_SAMPLE_RATE, dtype=torch.float32) * 0.01).unsqueeze(0)
    extractor = ClapFeatureExtractor.from_pretrained(clap_checkpoint)
    expected = extractor(list(audio.numpy()), sampling_rate=_SAMPLE_RATE, return_tensors="pt")[
        "input_features"
    ]

    actual = clap_encoder.features(audio)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


def test_clap_forward_backpropagates_to_waveform(clap_encoder: ClapAudioEncoder) -> None:
    """The frozen backbone retains the graph from embedding to input waveform.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    audio = torch.sin(torch.arange(4_800, dtype=torch.float32) * 0.01).unsqueeze(0)
    audio.requires_grad_()

    embedding = clap_encoder(audio)
    (gradient,) = torch.autograd.grad(embedding.square().sum(), audio)

    assert embedding.shape == (1, _PROJECTION_DIM)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def test_clap_backbone_stays_frozen_and_in_eval_mode(clap_encoder: ClapAudioEncoder) -> None:
    """Parent training-mode changes cannot move CLAP weights or stochastic state.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    clap_encoder.train()

    assert not clap_encoder.clap.training
    assert all(not parameter.requires_grad for parameter in clap_encoder.clap.parameters())


def test_pretrained_conditioning_encoder_exposes_metric_and_conditioning_taps(
    clap_encoder: ClapAudioEncoder,
) -> None:
    """The frozen embedding and trained projection are usable from one forward pass.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    encoder = PretrainedConditioningEncoder(
        backbone=clap_encoder,
        head=VectorProjection(input_dim=_PROJECTION_DIM, d_model=6),
        out_dim=6,
    )
    audio = torch.sin(torch.arange(4_800, dtype=torch.float32) * 0.01).unsqueeze(0)

    embedding = encoder.embed(audio)
    conditioning = encoder.project(embedding)

    assert embedding.shape == (1, _PROJECTION_DIM)
    assert conditioning.shape == (1, 6)
    assert torch.equal(conditioning, encoder(audio))


def test_optimizer_excludes_frozen_clap_backbone(clap_encoder: ClapAudioEncoder) -> None:
    """Lightning optimizes the projection and field without registering CLAP parameters.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    encoder = PretrainedConditioningEncoder(
        backbone=clap_encoder,
        head=VectorProjection(input_dim=_PROJECTION_DIM, d_model=6),
        out_dim=6,
    )
    module = VSTFlowMatchingModule(
        encoder=encoder,
        vector_field=VectorField(field_dim=4, hidden_dim=8, conditioning_dim=6, num_blocks=1),
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=4,
        conditioning="audio",
        warmup_steps=0,
    )
    module._trainer = SimpleNamespace(model=module)  # pyright: ignore[reportAttributeAccessIssue]

    optimizer = module.configure_optimizers()["optimizer"]
    optimized = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }

    assert optimized
    assert all(id(parameter) not in optimized for parameter in clap_encoder.parameters())
    assert all(
        id(parameter) in optimized for parameter in module.parameters() if parameter.requires_grad
    )


def test_clap_features_with_empty_audio_raises(clap_encoder: ClapAudioEncoder) -> None:
    """An empty waveform fails at the input boundary instead of dividing by zero.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    with pytest.raises(ValueError, match="empty"):
        clap_encoder.features(torch.empty(1, 0))
