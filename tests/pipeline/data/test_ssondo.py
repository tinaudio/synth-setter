"""Behavioral tests for the pinned S-SONDO audio embedding adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.data.vst.shapes import SSONDO_FIELD
from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, IndexSpec, build_index
from synth_setter.pipeline.data.ssondo import (
    SSONDO_CHECKPOINT_SHA256,
    SSONDO_EMBEDDING_DIM,
    SSONDO_INPUT_SAMPLES,
    SSONDO_SAMPLE_RATE,
    resolve_ssondo_checkpoint,
    ssondo_encoder_input,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig


def test_ssondo_config_incompatible_pq_subvectors_raises() -> None:
    """Invalid S-SONDO PQ splits fail before embedding work starts."""
    with pytest.raises(ValueError, match=r"must divide the ssondo dim \(960\)"):
        AddEmbeddingsConfig(
            lance_uri="dataset.lance",
            embeddings=("ssondo",),
            num_sub_vectors=7,
        )


def test_ssondo_encoder_input_stereo_short_clip_downmixes_and_pads() -> None:
    """A short stereo clip becomes one float32 ten-second model window."""
    left = np.ones(32_000, dtype=np.float32)
    right = np.full(32_000, 3.0, dtype=np.float32)
    audio = np.stack((left, right))[None]

    prepared = ssondo_encoder_input(audio, SSONDO_SAMPLE_RATE)

    assert prepared.shape == (1, SSONDO_INPUT_SAMPLES)
    assert prepared.dtype == np.float32
    np.testing.assert_array_equal(prepared[0, :32_000], 2.0)
    np.testing.assert_array_equal(prepared[0, 32_000:], 0.0)


def test_ssondo_encoder_input_half_rate_resamples_to_target_window() -> None:
    """Source audio is resampled to 32 kHz before right-padding."""
    audio = np.zeros((2, 1, 16_000), dtype=np.float32)

    prepared = ssondo_encoder_input(audio, 16_000)

    assert prepared.shape == (2, SSONDO_INPUT_SAMPLES)
    assert prepared.dtype == np.float32
    assert np.isfinite(prepared).all()


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.zeros((1, 32_000), dtype=np.float32), 32_000, r"expected a \(B, C, T\) batch"),
        (np.zeros((1, 3, 32_000), dtype=np.float32), 32_000, "1 or 2 channels"),
        (np.zeros((1, 1, 32_000), dtype=np.float32), 0, "positive sample_rate"),
        (np.zeros((1, 1, 0), dtype=np.float32), 32_000, "non-empty"),
        (np.full((1, 1, 32_000), np.nan, dtype=np.float32), 32_000, "non-finite"),
        (np.zeros((1, 1, 320_001), dtype=np.float32), 32_000, "at most 10 seconds"),
    ],
)
def test_ssondo_encoder_input_incompatible_audio_raises(
    audio: np.ndarray, sample_rate: int, message: str
) -> None:
    """Malformed, non-finite, empty, and overlong clips fail before inference.

    :param audio: Candidate source batch.
    :param sample_rate: Candidate source rate.
    :param message: Expected failure detail.
    """
    with pytest.raises(ValueError, match=message):
        ssondo_encoder_input(audio, sample_rate)


def test_ssondo_registry_encoder_valid_output_returns_fixed_vector() -> None:
    """The registry persists one float32 vector per audio row."""
    audio = np.zeros((2, 1, 32_000), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        assert source.shape == audio.shape
        assert sample_rate == SSONDO_SAMPLE_RATE
        return np.ones((2, SSONDO_EMBEDDING_DIM), dtype=np.float32)

    encoded = EMBEDDING_REGISTRY["ssondo"].encode_column(
        audio, SSONDO_SAMPLE_RATE, encode
    )

    assert encoded.type == pa.list_(pa.float32(), SSONDO_EMBEDDING_DIM)
    assert np.asarray(encoded.to_pylist()).shape == (2, SSONDO_EMBEDDING_DIM)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones((2, 1, SSONDO_EMBEDDING_DIM), dtype=np.float32), "produced shape"),
        (np.full((2, SSONDO_EMBEDDING_DIM), np.inf, dtype=np.float32), "non-finite"),
    ],
)
def test_ssondo_registry_encoder_invalid_output_raises(
    output: np.ndarray, message: str
) -> None:
    """Wrong-shaped and non-finite outputs cannot reach Lance.

    :param output: Candidate encoder output.
    :param message: Expected failure detail.
    """
    audio = np.zeros((2, 1, 32_000), dtype=np.float32)

    def encode(source: np.ndarray, sample_rate: int) -> np.ndarray:
        del source, sample_rate
        return output

    with pytest.raises(ValueError, match=message):
        EMBEDDING_REGISTRY["ssondo"].encode_column(
            audio, SSONDO_SAMPLE_RATE, encode
        )


def test_resolve_ssondo_checkpoint_wrong_hash_raises(tmp_path: Path) -> None:
    """A local override cannot weaken the pinned checkpoint identity.

    :param tmp_path: Temporary checkpoint location.
    """
    checkpoint = tmp_path / "matpac_mobilenetv3.ckpt"
    checkpoint.write_bytes(b"not the trusted S-SONDO checkpoint")

    with pytest.raises(ValueError, match=SSONDO_CHECKPOINT_SHA256):
        resolve_ssondo_checkpoint(str(checkpoint))


def test_resolve_ssondo_checkpoint_unpinned_repo_raises() -> None:
    """A mutable or unrelated Hugging Face repository is rejected."""
    with pytest.raises(ValueError, match="pinned S-SONDO checkpoint"):
        resolve_ssondo_checkpoint("somebody/other-model")


@pytest.mark.slow
def test_ssondo_index_builds_and_returns_stored_query_row(tmp_path: Path) -> None:
    """The 960-dimensional policy builds a searchable IVF_PQ index.

    :param tmp_path: Scratch directory for the Lance dataset.
    """
    uri = tmp_path / "ssondo-index.lance"
    vectors = np.random.default_rng(7).standard_normal((300, SSONDO_EMBEDDING_DIM)).astype(
        np.float32
    )
    lance.write_dataset(
        pa.table(
            {
                "row_id": pa.array(np.arange(300)),
                SSONDO_FIELD: pa.array(vectors.tolist(), type=pa.list_(pa.float32(), 960)),
            }
        ),
        str(uri),
    )
    dataset = lance.dataset(str(uri))
    config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("ssondo",),
        num_partitions=4,
        num_sub_vectors=16,
        metric="l2",
    )

    built = build_index(
        dataset,
        SSONDO_FIELD,
        index=cast(IndexSpec, EMBEDDING_REGISTRY["ssondo"].index),
        config=config,
    )

    indices = cast("list[dict[str, Any]]", lance.dataset(str(uri)).list_indices())
    assert built is True
    assert [entry["fields"] for entry in indices] == [[SSONDO_FIELD]]
    hits = lance.dataset(str(uri)).to_table(
        nearest={"column": SSONDO_FIELD, "q": vectors[137], "k": 1},
        columns=["row_id"],
    )
    assert hits.column("row_id")[0].as_py() == 137
