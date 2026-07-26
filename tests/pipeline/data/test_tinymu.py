"""Behavioral tests for the pinned TinyMU MATPAC adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synth_setter.data.vst.shapes import TINYMU_FIELD
from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, IndexSpec
from synth_setter.pipeline.data.tinymu import (
    DEFAULT_TINYMU_CHECKPOINT,
    TINYMU_CHECKPOINT_SHA256,
    TINYMU_EMBEDDING_DIM,
    TINYMU_SAMPLE_RATE,
    TINYMU_SOURCE_COMMIT,
    TINYMU_SOURCE_DIR_ENV,
    TINYMU_SOURCE_MODEL_PATH,
    resolve_tinymu_checkpoint,
    resolve_tinymu_source_model,
    tinymu_encoder_input,
    tinymu_num_latent_frames,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

_SAMPLE_RATE = 16_000
_AUDIO_SAMPLES = 16_000


def test_tinymu_registry_spec_pins_checkpoint_and_mean_pooling() -> None:
    """The registry exposes one immutable MATPAC sequence policy."""
    spec = EMBEDDING_REGISTRY["tinymu"]

    assert spec.column == TINYMU_FIELD
    assert spec.default_checkpoint == DEFAULT_TINYMU_CHECKPOINT
    assert spec.requires_extra == "tinymu"
    assert spec.co_resident is False
    assert spec.index == IndexSpec(pool="mean", vector_column=f"{TINYMU_FIELD}_vec")


@pytest.mark.parametrize(
    ("num_samples", "sample_rate", "expected_frames"),
    [
        (2_800, 16_000, 1),
        (16_000, 16_000, 7),
        (64_000, 16_000, 25),
        (480_000, 48_000, 63),
    ],
)
def test_tinymu_num_latent_frames_measured_clips_matches_contract(
    num_samples: int, sample_rate: int, expected_frames: int
) -> None:
    """Measured clip lengths map to the upstream MATPAC token counts.

    :param num_samples: Source clip length.
    :param sample_rate: Source sample rate.
    :param expected_frames: Measured MATPAC token count.
    """
    assert tinymu_num_latent_frames(num_samples, sample_rate) == expected_frames


@given(
    channels=st.integers(min_value=1, max_value=2),
    sample_rate=st.sampled_from([8_000, 16_000, 22_050, 44_100, 48_000]),
)
@settings(max_examples=10, deadline=None)
def test_tinymu_encoder_input_valid_audio_returns_finite_mono_float32(
    channels: int, sample_rate: int
) -> None:
    """Supported channel/sample-rate combinations normalize to MATPAC input.

    :param channels: Supported source channel count.
    :param sample_rate: Supported source sample rate.
    """
    samples = sample_rate
    time = np.arange(samples, dtype=np.float32) / sample_rate
    channel_rows = [np.sin(2 * np.pi * frequency * time) for frequency in (220.0, 330.0)]
    audio = np.stack(channel_rows[:channels])[None]

    prepared = tinymu_encoder_input(audio, sample_rate)

    assert prepared.shape == (1, TINYMU_SAMPLE_RATE)
    assert prepared.dtype == np.float32
    assert np.isfinite(prepared).all()


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.zeros((2, 100), dtype=np.float32), 16_000, r"expected a \(B, C, T\) batch"),
        (np.zeros((1, 3, 3_000), dtype=np.float32), 16_000, "1 or 2 channels"),
        (np.zeros((1, 1, 3_000), dtype=np.float32), 0, "positive sample_rate"),
        (np.full((1, 1, 3_000), np.nan, dtype=np.float32), 16_000, "non-finite"),
        (np.zeros((1, 1, 2_799), dtype=np.float32), 16_000, "at least 2800 samples"),
    ],
)
def test_tinymu_encoder_input_incompatible_audio_raises(
    audio: np.ndarray, sample_rate: int, message: str
) -> None:
    """Malformed, non-finite, and too-short audio fails before inference.

    :param audio: Candidate source batch.
    :param sample_rate: Candidate source sample rate.
    :param message: Expected failure detail.
    """
    with pytest.raises(ValueError, match=message):
        tinymu_encoder_input(audio, sample_rate)


def test_tinymu_registry_encoder_valid_sequence_returns_fixed_shape_tensor() -> None:
    """The registry persists TinyMU sequences in conditioning orientation."""
    audio = np.zeros((2, 1, _AUDIO_SAMPLES), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        assert source.shape == audio.shape
        assert sample_rate == _SAMPLE_RATE
        return np.ones((2, TINYMU_EMBEDDING_DIM, 7), dtype=np.float32)

    encoded = EMBEDDING_REGISTRY["tinymu"].encode_column(audio, _SAMPLE_RATE, encode)

    assert encoded.to_numpy_ndarray().shape == (2, TINYMU_EMBEDDING_DIM, 7)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones((2, 7, TINYMU_EMBEDDING_DIM), dtype=np.float32), "produced shape"),
        (np.full((2, TINYMU_EMBEDDING_DIM, 7), np.inf, dtype=np.float32), "non-finite"),
    ],
)
def test_tinymu_registry_encoder_invalid_output_raises(
    output: np.ndarray, message: str
) -> None:
    """Wrong orientation and non-finite MATPAC outputs cannot reach Lance.

    :param output: Candidate encoder output.
    :param message: Expected failure detail.
    """
    audio = np.zeros((2, 1, _AUDIO_SAMPLES), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del source, sample_rate
        return output

    with pytest.raises(ValueError, match=message):
        EMBEDDING_REGISTRY["tinymu"].encode_column(audio, _SAMPLE_RATE, encode)


def test_resolve_tinymu_checkpoint_wrong_hash_raises(tmp_path: Path) -> None:
    """A local checkpoint override cannot weaken the pinned artifact identity.

    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "matpac.pt"
    checkpoint.write_bytes(b"not the trusted MATPAC checkpoint")

    with pytest.raises(ValueError, match=TINYMU_CHECKPOINT_SHA256):
        resolve_tinymu_checkpoint(str(checkpoint))


def test_resolve_tinymu_checkpoint_unpinned_r2_uri_raises() -> None:
    """A mutable or unrelated R2 object is rejected before hydration."""
    with pytest.raises(ValueError, match="pinned TinyMU checkpoint"):
        resolve_tinymu_checkpoint("r2://intermediate-data/tinymu/main/model.pt")


def test_resolve_tinymu_source_model_wrong_commit_raises(tmp_path: Path) -> None:
    """The adapter refuses a checkout whose source does not match the pinned commit.

    :param tmp_path: Temporary Git checkout location.
    """
    source = tmp_path / "TinyMU"
    model = source / TINYMU_SOURCE_MODEL_PATH
    model.parent.mkdir(parents=True)
    model.write_text("class matpac_wrapper: pass\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=TinyMU test",
            "-c",
            "user.email=tinymu@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    with pytest.raises(ValueError, match=TINYMU_SOURCE_COMMIT):
        resolve_tinymu_source_model(source)


def test_add_embeddings_config_tinymu_source_string_coerces_path() -> None:
    """Hydra source-directory strings remain strict Path values."""
    config = AddEmbeddingsConfig(
        lance_uri="dataset.lance",
        embeddings=("tinymu",),
        tinymu_source_dir="/opt/TinyMU",  # type: ignore[arg-type]
    )

    assert config.tinymu_source_dir == Path("/opt/TinyMU")


def test_add_embeddings_config_tinymu_source_env_name_is_stable() -> None:
    """The documented environment fallback remains a single explicit boundary."""
    assert TINYMU_SOURCE_DIR_ENV == "TINYMU_SOURCE_DIR"
    assert TINYMU_EMBEDDING_DIM == 3_840
