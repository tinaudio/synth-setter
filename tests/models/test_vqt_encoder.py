"""Behavior tests for online nnAudio2 VQT conditioning."""

from typing import TypedDict, cast

import pytest
import torch

from synth_setter.models.components.embed_pool import EmbeddingPool
from synth_setter.models.components.pretrained_encoder import PretrainedConditioningEncoder
from synth_setter.models.components.vqt_encoder import VqtAudioEncoder


class _VqtEncoderKwargs(TypedDict):
    sample_rate: int
    hop_length: int
    fmin: float
    n_bins: int
    bins_per_octave: int
    gamma: float
    max_batch_size: int
    pad_mode: str


def _tones(*, rows: int, channels: int, samples: int, sample_rate: int) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    mono = torch.stack(
        [torch.sin(2 * torch.pi * (220.0 + 110.0 * row) * time) for row in range(rows)]
    )
    return mono[:, None, :].repeat(1, channels, 1)


def test_vqt_audio_encoder_real_waveforms_returns_log_magnitude_features() -> None:
    """Real waveform extraction emits finite detached VQT features."""
    sample_rate = 16_000
    audio = _tones(rows=2, channels=2, samples=4_000, sample_rate=sample_rate).requires_grad_()
    encoder = VqtAudioEncoder(
        sample_rate=sample_rate,
        hop_length=160,
        fmin=55.0,
        n_bins=24,
        bins_per_octave=12,
        gamma=20.0,
        max_batch_size=2,
    )

    features = encoder(audio)

    assert features.shape == (2, 24, 26)
    assert features.dtype == torch.float32
    assert features.device == audio.device
    assert torch.isfinite(features).all()
    assert torch.all(features >= 0)
    assert not features.requires_grad


def test_vqt_audio_encoder_production_policy_returns_256_by_401() -> None:
    """Canonical four-second audio emits the documented conditioning geometry."""
    audio = _tones(rows=1, channels=2, samples=176_400, sample_rate=44_100)
    encoder = VqtAudioEncoder(
        sample_rate=44_100,
        hop_length=441,
        fmin=32.7,
        n_bins=256,
        bins_per_octave=32,
        gamma=20.0,
    )

    features = encoder(audio)

    assert features.shape == (1, 256, 401)
    assert torch.isfinite(features).all()


@pytest.mark.gpu
def test_vqt_audio_encoder_cuda_returns_features_on_cuda() -> None:
    """The transform and output follow waveform placement onto CUDA."""
    audio = _tones(rows=1, channels=1, samples=4_000, sample_rate=16_000).cuda()
    encoder = VqtAudioEncoder(
        sample_rate=16_000,
        hop_length=160,
        fmin=55.0,
        n_bins=24,
        bins_per_octave=12,
        gamma=20.0,
    ).cuda()

    features = encoder(audio)

    assert features.device.type == "cuda"
    assert features.shape == (1, 24, 26)
    assert torch.isfinite(features).all()


def test_vqt_audio_encoder_disables_ambient_autocast() -> None:
    """Mixed-precision training preserves the float32 extraction policy."""
    audio = _tones(rows=1, channels=1, samples=4_000, sample_rate=16_000)
    encoder = VqtAudioEncoder(
        sample_rate=16_000,
        hop_length=160,
        fmin=55.0,
        n_bins=24,
        bins_per_octave=12,
        gamma=20.0,
    )
    expected = encoder(audio)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = encoder(audio)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_vqt_audio_encoder_low_precision_matches_input_dtype() -> None:
    """Float32 extraction casts features back for low-precision heads."""
    audio = _tones(rows=1, channels=1, samples=4_000, sample_rate=16_000).bfloat16()
    encoder = VqtAudioEncoder(
        sample_rate=16_000,
        hop_length=160,
        fmin=55.0,
        n_bins=24,
        bins_per_octave=12,
        gamma=20.0,
    ).to(dtype=torch.bfloat16)

    features = encoder(audio)

    assert features.dtype == torch.bfloat16


def test_vqt_conditioning_backpropagates_only_through_trainable_pool() -> None:
    """Frozen extraction detaches audio while preserving projection gradients."""
    sample_rate = 16_000
    audio = _tones(rows=2, channels=1, samples=4_000, sample_rate=sample_rate).requires_grad_()
    backbone = VqtAudioEncoder(
        sample_rate=sample_rate,
        hop_length=160,
        fmin=55.0,
        n_bins=24,
        bins_per_octave=12,
        gamma=20.0,
    )
    head = EmbeddingPool(embed_dim=24, d_model=8, num_heads=2, max_seq_len=26)
    encoder = PretrainedConditioningEncoder(backbone=backbone, head=head, out_dim=8)

    encoder(audio).square().mean().backward()

    assert audio.grad is None
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_vqt_audio_encoder_distinct_tones_shift_peak_frequency() -> None:
    """The VQT representation responds directionally to waveform pitch."""
    sample_rate = 16_000
    time = torch.arange(4_000, dtype=torch.float32) / sample_rate
    audio = torch.stack(
        [
            torch.sin(2 * torch.pi * 220.0 * time),
            torch.sin(2 * torch.pi * 440.0 * time),
        ]
    )
    encoder = VqtAudioEncoder(
        sample_rate=sample_rate,
        hop_length=160,
        fmin=55.0,
        n_bins=48,
        bins_per_octave=12,
        gamma=20.0,
    )

    peak_bins = encoder(audio).mean(dim=-1).argmax(dim=-1)

    assert peak_bins[1] > peak_bins[0]


def test_vqt_audio_encoder_chunked_batch_matches_single_pass() -> None:
    """Chunking bounds transform batches without changing their features."""
    sample_rate = 16_000
    audio = _tones(rows=3, channels=1, samples=4_000, sample_rate=sample_rate)
    kwargs = {
        "sample_rate": sample_rate,
        "hop_length": 160,
        "fmin": 55.0,
        "n_bins": 24,
        "bins_per_octave": 12,
        "gamma": 20.0,
    }

    chunked = VqtAudioEncoder(max_batch_size=1, **kwargs)(audio)
    single_pass = VqtAudioEncoder(max_batch_size=3, **kwargs)(audio)

    torch.testing.assert_close(chunked, single_pass)


def test_vqt_audio_encoder_unlimited_batch_matches_single_pass() -> None:
    """The unlimited sentinel sends the complete batch through unchanged."""
    sample_rate = 16_000
    audio = _tones(rows=3, channels=1, samples=4_000, sample_rate=sample_rate)
    kwargs = {
        "sample_rate": sample_rate,
        "hop_length": 160,
        "fmin": 55.0,
        "n_bins": 24,
        "bins_per_octave": 12,
        "gamma": 20.0,
    }

    unlimited = VqtAudioEncoder(max_batch_size=-1, **kwargs)(audio)
    single_pass = VqtAudioEncoder(max_batch_size=3, **kwargs)(audio)

    torch.testing.assert_close(unlimited, single_pass)


def test_vqt_audio_encoder_channel_mean_matches_mono_input() -> None:
    """Stereo conditioning uses the documented channel-mean policy."""
    sample_rate = 16_000
    stereo = _tones(rows=1, channels=2, samples=4_000, sample_rate=sample_rate)
    encoder = VqtAudioEncoder(
        sample_rate=sample_rate,
        hop_length=160,
        fmin=55.0,
        n_bins=24,
        bins_per_octave=12,
        gamma=20.0,
    )

    stereo_features = encoder(stereo)
    mono_features = encoder(stereo.mean(dim=1))

    torch.testing.assert_close(stereo_features, mono_features)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (torch.empty(0, 400), "batch"),
        (torch.empty(1, 0, 400), "channels"),
        (torch.empty(1, 1, 0), "samples"),
        (torch.empty(1, 1, 159), "hop_length"),
        (torch.empty(1, 1, 1, 400), "shape"),
    ],
)
def test_vqt_audio_encoder_invalid_waveform_raises(audio: torch.Tensor, message: str) -> None:
    """Invalid waveform geometry fails before entering nnAudio2.

    :param audio: Malformed waveform batch.
    :param message: Expected failing geometry name.
    """
    encoder = VqtAudioEncoder(
        sample_rate=16_000,
        hop_length=160,
        fmin=55.0,
        n_bins=24,
        bins_per_octave=12,
        gamma=20.0,
    )

    with pytest.raises(ValueError, match=message):
        encoder(audio)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"sample_rate": 100}, "Nyquist"),
        ({"hop_length": 0}, "hop_length"),
        ({"fmin": 0.0}, "fmin"),
        ({"n_bins": 0}, "n_bins"),
        ({"bins_per_octave": 0}, "bins_per_octave"),
        ({"gamma": 0.0}, "gamma"),
        ({"gamma": float("nan")}, "gamma"),
        ({"fmin": float("inf")}, "fmin"),
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_batch_size": -2}, "max_batch_size"),
        ({"pad_mode": "zeros"}, "pad_mode"),
    ],
)
def test_vqt_audio_encoder_invalid_configuration_raises(
    kwargs: dict[str, float | int | str], message: str
) -> None:
    """Invalid extraction configuration is rejected.

    :param kwargs: Constructor values containing one invalid field.
    :param message: Expected invalid field name.
    """
    values: dict[str, object] = {
        "sample_rate": 16_000,
        "hop_length": 160,
        "fmin": 55.0,
        "n_bins": 24,
        "bins_per_octave": 12,
        "gamma": 20.0,
        "max_batch_size": 32,
        "pad_mode": "reflect",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        VqtAudioEncoder(**cast(_VqtEncoderKwargs, values))
