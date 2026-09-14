"""Behavior tests for online CQT conditioning."""

import pytest
import torch

from synth_setter.models.components.cqt_encoder import CqtAudioEncoder
from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.pretrained_encoder import PretrainedConditioningEncoder
from synth_setter.pipeline.data.cqt import CQT_EMBEDDING_DIM, cqt_num_frames


def _tones(*, rows: int, channels: int, samples: int, sample_rate: int) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    mono = torch.stack(
        [torch.sin(2 * torch.pi * (220.0 + 110.0 * row) * time) for row in range(rows)]
    )
    return mono[:, None, :].repeat(1, channels, 1)


def test_cqt_audio_encoder_real_waveforms_returns_canonical_features() -> None:
    """Real waveform extraction emits finite detached log-magnitude features."""
    sample_rate = 16_000
    samples = 4_000
    audio = _tones(rows=2, channels=2, samples=samples, sample_rate=sample_rate).requires_grad_()

    features = CqtAudioEncoder(sample_rate=sample_rate, max_batch_size=2)(audio)

    assert features.shape == (2, CQT_EMBEDDING_DIM, cqt_num_frames(samples, sample_rate))
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert torch.all(features >= 0)
    assert not features.requires_grad


def test_cqt_conditioning_backpropagates_only_through_trainable_pool() -> None:
    """Frozen extraction detaches audio while preserving projection gradients."""
    sample_rate = 16_000
    audio = _tones(rows=2, channels=1, samples=4_000, sample_rate=sample_rate).requires_grad_()
    head = EmbeddingPool(
        embed_dim=CQT_EMBEDDING_DIM,
        d_model=8,
        num_heads=2,
        max_seq_len=cqt_num_frames(4_000, sample_rate),
    )
    encoder = PretrainedConditioningEncoder(
        backbone=CqtAudioEncoder(sample_rate=sample_rate), head=head, out_dim=8
    )

    encoder(audio).square().mean().backward()

    assert audio.grad is None
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_cqt_audio_encoder_true_low_precision_matches_input_dtype() -> None:
    """Float32 extraction casts features back for true low-precision heads."""
    encoder = CqtAudioEncoder(sample_rate=16_000).to(dtype=torch.bfloat16)
    audio = _tones(rows=1, channels=1, samples=4_000, sample_rate=16_000).bfloat16()

    features = encoder(audio)

    assert features.dtype == torch.bfloat16


def test_cqt_audio_encoder_device_transition_clears_transform_cache() -> None:
    """Module device transitions release cached upstream transform tensors."""
    encoder = CqtAudioEncoder(sample_rate=16_000)
    audio = _tones(rows=1, channels=1, samples=4_000, sample_rate=16_000)
    encoder(audio)

    assert len(encoder._transforms) == 1

    encoder.cpu()

    assert not encoder._transforms


def test_cqt_audio_encoder_chunked_batch_matches_single_pass() -> None:
    """Chunking bounds transform batches without changing their features."""
    sample_rate = 16_000
    audio = _tones(rows=3, channels=1, samples=4_000, sample_rate=sample_rate)

    chunked = CqtAudioEncoder(sample_rate=sample_rate, max_batch_size=1)(audio)
    single_pass = CqtAudioEncoder(sample_rate=sample_rate, max_batch_size=3)(audio)

    torch.testing.assert_close(chunked, single_pass)


def test_cqt_audio_encoder_channel_mean_matches_mono_input() -> None:
    """Stereo conditioning follows the cached CQT channel-mean policy."""
    sample_rate = 16_000
    stereo = _tones(rows=1, channels=2, samples=4_000, sample_rate=sample_rate)

    stereo_features = CqtAudioEncoder(sample_rate=sample_rate)(stereo)
    mono_features = CqtAudioEncoder(sample_rate=sample_rate)(stereo.mean(dim=1))

    torch.testing.assert_close(stereo_features, mono_features)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (torch.empty(0, 400), "batch"),
        (torch.empty(1, 0, 400), "channels"),
        (torch.empty(1, 1, 0), "samples"),
        (torch.empty(1, 1, 1, 400), "shape"),
    ],
)
def test_cqt_audio_encoder_invalid_waveform_raises(audio: torch.Tensor, message: str) -> None:
    """Invalid waveform geometry fails before entering the upstream transform.

    :param audio: Malformed waveform batch.
    :param message: Expected failing geometry name.
    """
    encoder = CqtAudioEncoder(sample_rate=16_000)

    with pytest.raises(ValueError, match=message):
        encoder(audio)


@pytest.mark.parametrize("field", ["sample_rate", "max_batch_size"])
def test_cqt_audio_encoder_nonpositive_configuration_raises(field: str) -> None:
    """Nonpositive extraction configuration is rejected.

    :param field: Constructor argument set to zero.
    """
    kwargs = {"sample_rate": 16_000, "max_batch_size": 32, field: 0}

    with pytest.raises(ValueError, match=field):
        CqtAudioEncoder(**kwargs)
