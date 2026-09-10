"""Tests for GPU-capable CQT feature extraction."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import torch
from pydantic import ValidationError

from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.pipeline.data.add_embeddings import _encode_cqt_column
from synth_setter.pipeline.data.cqt import (
    CQT_BINS_PER_OCTAVE,
    CQT_EMBEDDING_DIM,
    CQT_NUM_OCTAVES,
    cqt_artifact_digest,
    cqt_num_frames,
    load_cqt_audio_encoder,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig
from tests.helpers.run_if import RunIf


def _tone_batch(*, sample_rate: int, seconds: float = 0.25) -> np.ndarray:
    """Return two deterministic mono tones.

    :param sample_rate: Audio sample rate in Hz.
    :param seconds: Tone duration in seconds.
    :returns: ``(2, 1, samples)`` float32 waveform batch.
    """
    samples = int(sample_rate * seconds)
    time = np.arange(samples, dtype=np.float32) / sample_rate
    tones = np.stack(
        [
            np.sin(2 * np.pi * 220.0 * time),
            np.sin(2 * np.pi * 440.0 * time),
        ]
    )
    return tones[:, None, :].astype(np.float32)


def test_cqt_num_frames_for_four_second_audio_matches_conditioning_grid() -> None:
    """Four-second CQT features use the canonical 100 Hz conditioning grid."""
    assert cqt_num_frames(176_400, 44_100) == 401


def test_cqt_artifact_digest_with_checkpoint_override_raises() -> None:
    """Checkpoint overrides cannot mislabel the checkpoint-free transform."""
    with pytest.raises(ValueError, match="checkpoint-free"):
        cqt_artifact_digest("weights.pt")


def test_cqt_config_with_checkpoint_override_raises() -> None:
    """The strict CLI boundary rejects CQT checkpoint overrides."""
    with pytest.raises(ValidationError, match="checkpoint-free"):
        AddEmbeddingsConfig(lance_uri="dataset.lance", checkpoints={"cqt": "weights.pt"})


def test_cqt_encoder_for_real_tones_returns_finite_nonzero_features() -> None:
    """The real upstream transform returns consumable log-magnitude features."""
    sample_rate = 16_000
    audio = _tone_batch(sample_rate=sample_rate)

    features = load_cqt_audio_encoder("cpu")(audio, sample_rate)

    assert features.shape == (2, CQT_EMBEDDING_DIM, cqt_num_frames(4_000, sample_rate))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert np.all(features >= 0)
    assert np.count_nonzero(features) > 0
    assert not np.allclose(features[0], features[1])


def test_cqt_encoder_with_opposite_stereo_channels_returns_zero_features() -> None:
    """The channel-mean policy cancels opposite-polarity stereo audio."""
    sample_rate = 16_000
    mono = _tone_batch(sample_rate=sample_rate)[:1]
    stereo = np.concatenate([mono, -mono], axis=1)

    features = load_cqt_audio_encoder("cpu")(stereo, sample_rate)

    np.testing.assert_array_equal(features, np.zeros_like(features))


def test_encode_cqt_column_with_valid_features_builds_arrow_tensor() -> None:
    """The registry adapter preserves CQT shape and float32 values."""
    sample_rate = 16_000
    audio = _tone_batch(sample_rate=sample_rate)
    frames = cqt_num_frames(audio.shape[-1], sample_rate)
    expected = np.ones((2, CQT_EMBEDDING_DIM, frames), dtype=np.float32)

    encoded = _encode_cqt_column(
        {AUDIO_FIELD: audio}, sample_rate, lambda waveform, rate: expected
    )

    assert isinstance(encoded.type, pa.FixedShapeTensorType)
    assert encoded.type.shape == [CQT_EMBEDDING_DIM, frames]
    np.testing.assert_array_equal(encoded.to_numpy_ndarray(), expected)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_encode_cqt_column_with_nonfinite_feature_raises(value: float) -> None:
    """The registry adapter rejects non-finite persisted values.

    :param value: Invalid feature value returned by the encoder.
    """
    sample_rate = 16_000
    audio = _tone_batch(sample_rate=sample_rate)
    frames = cqt_num_frames(audio.shape[-1], sample_rate)
    features = np.zeros((2, CQT_EMBEDDING_DIM, frames), dtype=np.float32)
    features[0, 0, 0] = value

    with pytest.raises(ValueError, match="non-finite"):
        _encode_cqt_column({AUDIO_FIELD: audio}, sample_rate, lambda waveform, rate: features)


def test_encode_cqt_column_with_wrong_shape_raises() -> None:
    """The registry adapter rejects tensors outside its storage contract."""
    sample_rate = 16_000
    audio = _tone_batch(sample_rate=sample_rate)

    with pytest.raises(ValueError, match="expected"):
        _encode_cqt_column(
            {AUDIO_FIELD: audio},
            sample_rate,
            lambda waveform, rate: np.zeros((2, 128, 3), dtype=np.float32),
        )


def test_cqt_encoder_with_invalid_audio_rank_raises() -> None:
    """The adapter rejects audio outside the public ``(B, C, T)`` contract."""
    encoder = load_cqt_audio_encoder("cpu")

    with pytest.raises(ValueError, match=r"expected audio shaped \(B, C, T\)"):
        encoder(np.zeros((2, 4_000), dtype=np.float32), 16_000)


@RunIf(min_gpus=1)
@pytest.mark.gpu
def test_cqt_encoder_on_cuda_matches_cpu_and_allocates_gpu_memory() -> None:
    """CUDA extraction runs on the GPU and agrees with the CPU transform."""
    sample_rate = 16_000
    audio = _tone_batch(sample_rate=sample_rate)
    expected = load_cqt_audio_encoder("cpu")(audio, sample_rate)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    actual = load_cqt_audio_encoder("cuda")(audio, sample_rate)

    assert torch.cuda.max_memory_allocated() > 0
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_cqt_constants_define_256_log_frequency_bins() -> None:
    """The persisted frequency width is the product of octaves and bins per octave."""
    assert CQT_EMBEDDING_DIM == CQT_NUM_OCTAVES * CQT_BINS_PER_OCTAVE == 256
