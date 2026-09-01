"""Behavioral coverage for branch-safe distributed embedding backfills."""

from __future__ import annotations

import pickle
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.pipeline.data.backfill_embeddings import (
    EmbeddingPromotionConfig,
    _transform_fragment,
    promote_embedding_candidate,
)
from synth_setter.pipeline.data.lance_shard import tensor_array


def test_transform_fragment_writes_uncommitted_clap_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker must write fragment data without publishing a branch manifest.

    :param tmp_path: Temporary directory for the real Lance dataset.
    :param monkeypatch: Fixture replacing only the heavyweight encoder loader.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill

    audio = np.arange(64, dtype=np.float32).reshape(8, 2, 4) / 64
    uri = tmp_path / "worker.lance"
    dataset = lance.write_dataset(
        pa.table({"audio": tensor_array(audio, np.dtype("float32"), (2, 4))}),
        uri,
        max_rows_per_file=8,
    )
    candidate = dataset.create_branch("embeddings-2985", (None, dataset.version))

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.repeat(mono.mean(axis=1, keepdims=True), 512, axis=1)

    monkeypatch.setattr(backfill, "_worker_encoder", lambda *args: encode)
    fragment_id = candidate.get_fragments()[0].metadata.id
    metadata_bytes, schema_bytes, rows, _metrics = _transform_fragment(
        str(uri),
        None,
        "embeddings-2985",
        candidate.version,
        fragment_id,
        "clap",
        "unused-checkpoint",
        44_100,
        2,
        b"clap:test-artifact",
    )

    assert candidate.version == 1
    operation = lance.LanceOperation.Merge(
        [pickle.loads(metadata_bytes)],  # noqa: S301
        pickle.loads(schema_bytes),  # noqa: S301
    )
    committed = lance.LanceDataset.commit(candidate, operation, read_version=candidate.version)
    assert rows == 8
    assert committed.version == 2
    assert committed.schema.field("clap").metadata == {
        b"synth_setter.embedding.name": b"clap",
        b"synth_setter.embedding.artifact": b"clap:test-artifact",
    }
    expected = np.repeat(audio.mean(axis=(1, 2), keepdims=False)[:, None], 512, axis=1)
    actual = committed.take(range(8), columns=["clap"]).column(0).combine_chunks()
    np.testing.assert_allclose(actual.values.to_numpy().reshape(8, 512), expected)


def test_promote_embedding_candidate_reuses_branch_fragments_in_one_main_commit(
    tmp_path: Path,
) -> None:
    """Promotion must preserve source rows and publish both candidate columns atomically.

    :param tmp_path: Temporary directory for the real Lance dataset.
    """
    uri = tmp_path / "embeddings.lance"
    source = pa.table(
        {
            "row_id": np.arange(8, dtype=np.int64),
            "audio": pa.array(np.arange(32, dtype=np.float32).reshape(8, 4).tolist()),
        }
    ).replace_schema_metadata({b"source-contract": b"preserve-me"})
    dataset = lance.write_dataset(source, uri, max_rows_per_file=4)
    dataset.tags.create("pre-embeddings-2985", (None, dataset.version))
    candidate = dataset.create_branch("embeddings-2985", (None, dataset.version))
    candidate.add_columns(
        pa.table(
            {
                "clap": pa.array(np.arange(16, dtype=np.float32).reshape(8, 2).tolist()),
                "meanaudio_16k": pa.array(np.arange(24, dtype=np.float32).reshape(8, 3).tolist()),
            }
        )
    )
    candidate_version = candidate.version

    result = promote_embedding_candidate(
        EmbeddingPromotionConfig(
            lance_uri=str(uri),
            candidate_branch="embeddings-2985",
            rollback_tag="pre-embeddings-2985",
            columns=("clap", "meanaudio_16k"),
        )
    )

    main = lance.dataset(uri)
    assert result.source_version == 1
    assert result.candidate_version == candidate_version
    assert result.committed_version == 2
    assert main.version == 2
    assert main.schema.names == ["row_id", "audio", "clap", "meanaudio_16k"]
    assert main.schema.metadata == source.schema.metadata
    assert main.take(range(8), columns=["row_id", "audio"]).equals(
        source.select(["row_id", "audio"])
    )
    assert lance.dataset(uri).checkout_version(("embeddings-2985", None)).version == (
        candidate_version
    )

    retry = promote_embedding_candidate(
        EmbeddingPromotionConfig(
            lance_uri=str(uri),
            candidate_branch="embeddings-2985",
            rollback_tag="pre-embeddings-2985",
            columns=("clap", "meanaudio_16k"),
        )
    )
    assert retry.already_complete is True
    assert lance.dataset(uri).version == 2
