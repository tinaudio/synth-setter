"""Behavior tests for frozen online CLAP conditioning."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torchaudio.functional as audio_fn
from transformers import ClapFeatureExtractor

from synth_setter.models.components.pretrained_encoder import (
    ClapAudioEncoder,
    PretrainedConditioningEncoder,
)
from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.components.vector_projection import VectorProjection
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

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
def clap_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create an offline feature-extractor checkpoint with production geometry.

    :param tmp_path_factory: Module-scoped temporary directory factory.
    :returns: Local CLAP feature-extractor directory.
    """
    checkpoint = tmp_path_factory.mktemp("clap-feature-extractor")
    ClapFeatureExtractor(
        feature_size=64,
        sampling_rate=_SAMPLE_RATE,
        hop_length=480,
        fft_window_size=1024,
        max_length_s=10,
        frequency_min=50,
        frequency_max=14_000,
        truncation="rand_trunc",
        padding="repeatpad",
    ).save_pretrained(checkpoint)
    return str(checkpoint)


@pytest.fixture(scope="module")
def clap_encoder(clap_checkpoint: str) -> ClapAudioEncoder:
    """Build a deterministic small random CLAP network.

    :param clap_checkpoint: Self-contained feature-extractor checkpoint.
    :returns: Frozen waveform encoder suitable for CPU behavior tests.
    """
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return ClapAudioEncoder(
            sample_rate=_SAMPLE_RATE,
            checkpoint=clap_checkpoint,
            pretrained=False,
            backbone_config=_TINY_CLAP_CONFIG,
        )


def test_clap_checkpoint_with_wrong_digest_raises(clap_checkpoint: str) -> None:
    """A mutable checkpoint cannot silently replace the configured artifact.

    :param clap_checkpoint: Self-contained feature-extractor checkpoint.
    """
    with pytest.raises(ValueError, match="digest mismatch"):
        ClapAudioEncoder(
            sample_rate=_SAMPLE_RATE,
            checkpoint=clap_checkpoint,
            checkpoint_sha256="0" * 64,
            pretrained=False,
            backbone_config=_TINY_CLAP_CONFIG,
        )


def test_clap_offline_backbone_with_unknown_config_key_raises(clap_checkpoint: str) -> None:
    """Transformers rejects malformed offline config at construction.

    :param clap_checkpoint: Self-contained feature-extractor checkpoint.
    """
    with pytest.raises((TypeError, ValueError), match="unknown_option"):
        ClapAudioEncoder(
            sample_rate=_SAMPLE_RATE,
            checkpoint=clap_checkpoint,
            pretrained=False,
            backbone_config={"unknown_option": True},
        )


def test_clap_features_match_huggingface_short_audio_path(
    clap_encoder: ClapAudioEncoder, clap_checkpoint: str
) -> None:
    """Online torch features agree with the extractor used for stored CLAP vectors.

    :param clap_encoder: Differentiable online encoder under test.
    :param clap_checkpoint: Self-contained reference extractor checkpoint.
    """
    audio = torch.sin(torch.arange(_SAMPLE_RATE, dtype=torch.float32) * 0.01).unsqueeze(0)
    extractor = ClapFeatureExtractor.from_pretrained(clap_checkpoint)
    expected = extractor(list(audio.numpy()), sampling_rate=_SAMPLE_RATE, return_tensors="pt")[
        "input_features"
    ]

    actual = clap_encoder.features(audio)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


def test_clap_features_match_huggingface_after_44100_hz_resampling(
    clap_checkpoint: str,
) -> None:
    """The shipped TorchSynth rate matches HF after the documented resampling.

    :param clap_checkpoint: Self-contained reference extractor checkpoint.
    """
    sample_rate = 44_100
    audio = torch.sin(torch.arange(sample_rate, dtype=torch.float32) * 0.01).unsqueeze(0)
    encoder = ClapAudioEncoder(
        sample_rate=sample_rate,
        checkpoint=clap_checkpoint,
        pretrained=False,
        backbone_config=_TINY_CLAP_CONFIG,
    )
    extractor = ClapFeatureExtractor.from_pretrained(clap_checkpoint)
    resampled = audio_fn.resample(audio, sample_rate, _SAMPLE_RATE)
    expected = extractor(list(resampled.numpy()), sampling_rate=_SAMPLE_RATE, return_tensors="pt")[
        "input_features"
    ]

    actual = encoder.features(audio)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


def test_clap_stereo_features_match_mono_channel_mean(clap_encoder: ClapAudioEncoder) -> None:
    """Stored stereo audio uses the same mono downmix as offline CLAP embeddings.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    left = torch.sin(torch.arange(4_800, dtype=torch.float32) * 0.01)
    right = torch.cos(torch.arange(4_800, dtype=torch.float32) * 0.01)
    stereo = torch.stack((left, right)).unsqueeze(0)

    assert torch.equal(clap_encoder.features(stereo), clap_encoder.features(stereo.mean(dim=1)))


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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1.01, -1.01])
def test_clap_features_with_invalid_audio_raises(
    clap_encoder: ClapAudioEncoder, invalid: float
) -> None:
    """Non-finite and out-of-range waveforms fail before feature extraction.

    :param clap_encoder: Small frozen CLAP encoder under test.
    :param invalid: Invalid sample value under test.
    """
    audio = torch.zeros(1, 4_800)
    audio[0, 0] = invalid

    with pytest.raises(ValueError, match="finite|range"):
        clap_encoder.features(audio)


def test_clap_features_with_empty_audio_raises(clap_encoder: ClapAudioEncoder) -> None:
    """An empty waveform fails at the input boundary instead of dividing by zero.

    :param clap_encoder: Small frozen CLAP encoder under test.
    """
    with pytest.raises(ValueError, match="empty"):
        clap_encoder.features(torch.empty(1, 0))
