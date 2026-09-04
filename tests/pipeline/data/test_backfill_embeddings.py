"""Behavioral coverage for branch-safe distributed embedding backfills."""

from __future__ import annotations

import base64
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Never, cast

import lance
import numpy as np
import pyarrow as pa
import pytest
from git import Actor, Repo

from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY
from synth_setter.pipeline.data.backfill_embeddings import (
    EmbeddingBackfillConfig,
    EmbeddingBackfillResult,
    EmbeddingPromotionConfig,
    _BackfillContext,
    _BackfillOutcome,
    _CacheIdentity,
    _commit_reports,
    _copy_candidate_data_directory,
    _DispatchResult,
    _DispatchState,
    _embedding_is_complete,
    _FragmentReport,
    _FragmentTask,
    _implementation_revision,
    _ensure_embedding_index,
    _load_reports,
    _log_dispatch_progress,
    _normalise_candidate_data_file,
    _persist_report,
    _poll_dispatch,
    _prepare_dispatch,
    _PromotionContext,
    _publish_candidate,
    _report_store,
    _ReportStore,
    _resolve_source_git_sha,
    _run_backfill,
    _RunIdentity,
    _summarize_backfill,
    _transform_fragment,
    _validate_promotion,
    _write_result,
    backfill_embedding,
    main,
    promote_embedding_candidate,
)
from synth_setter.pipeline.data.lance_shard import tensor_array
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

_EMBEDDING_ARTIFACT_KEY = b"synth_setter.embedding.artifact"
_EMBEDDING_NAME_KEY = b"synth_setter.embedding.name"
_CLAP_ARTIFACT = b"clap:test-artifact"
_MEANAUDIO_ARTIFACT = b"meanaudio:test-artifact"


def _embedding_metadata(name: str, artifact: bytes) -> dict[bytes, bytes]:
    """Build the exact field metadata written by an embedding worker.

    :param name: Registry policy name.
    :param artifact: Nonempty artifact identity.
    :returns: Arrow field metadata.
    """
    return {_EMBEDDING_NAME_KEY: name.encode(), _EMBEDDING_ARTIFACT_KEY: artifact}


def _clap_promotion_table(
    row_values: tuple[float, ...] = (1.0, 2.0),
    *,
    width: int = 512,
    value_type: pa.DataType = pa.float32(),
    metadata: dict[bytes, bytes] | None = None,
) -> pa.Table:
    """Build CLAP promotion rows with an explicit schema contract.

    :param row_values: Scalar repeated across each row vector.
    :param width: Fixed CLAP vector width.
    :param value_type: Arrow vector element type.
    :param metadata: Field metadata override.
    :returns: Promotion table carrying one CLAP field.
    """
    vector_type = pa.list_(value_type, width)
    field = pa.field("clap", vector_type).with_metadata(
        _embedding_metadata("clap", _CLAP_ARTIFACT) if metadata is None else metadata
    )
    rows = [[value] * width for value in row_values]
    return pa.Table.from_arrays([pa.array(rows, type=vector_type)], schema=pa.schema([field]))


def _meanaudio_promotion_table(
    row_values: tuple[float, ...] = (1.0, 2.0),
    *,
    frames: int = 3,
    vector_artifact: bytes = _MEANAUDIO_ARTIFACT,
) -> pa.Table:
    """Build valid MeanAudio sequence and pooled-vector promotion rows.

    :param row_values: Scalar repeated across each row's outputs.
    :param frames: Positive latent-frame width.
    :param vector_artifact: Artifact identity on the pooled-vector field.
    :returns: Promotion table carrying the complete MeanAudio policy.
    """
    values = np.asarray(row_values, dtype=np.float32)
    sequence_values = np.broadcast_to(
        values[:, None, None], (len(values), 20, frames)
    ).copy()
    vector_values = np.broadcast_to(values[:, None], (len(values), 20)).copy()
    sequence = tensor_array(sequence_values, np.dtype("float32"), (20, frames))
    vector = pa.array(vector_values.tolist(), type=pa.list_(pa.float32(), 20))
    sequence_field = pa.field("meanaudio_16k", sequence.type).with_metadata(
        _embedding_metadata("meanaudio_16k", _MEANAUDIO_ARTIFACT)
    )
    vector_field = pa.field("meanaudio_16k_vec", vector.type).with_metadata(
        _embedding_metadata("meanaudio_16k", vector_artifact)
    )
    return pa.Table.from_arrays(
        [sequence, vector], schema=pa.schema([sequence_field, vector_field])
    )


def _embedding_promotion_table(row_values: tuple[float, ...]) -> pa.Table:
    """Build complete CLAP and MeanAudio promotion outputs.

    :param row_values: Scalar repeated across each row's policy outputs.
    :returns: Promotion table carrying both complete policies.
    """
    clap = _clap_promotion_table(row_values)
    meanaudio = _meanaudio_promotion_table(row_values)
    return pa.Table.from_arrays(
        [*clap.columns, *meanaudio.columns],
        schema=pa.schema([*clap.schema, *meanaudio.schema]),
    )


def _summary_context(
    *,
    dataset: lance.LanceDataset,
    config: EmbeddingBackfillConfig,
    identity: _RunIdentity,
    checkpoint: str,
    artifact: str,
) -> _BackfillContext:
    """Build a complete typed context for summary and dispatch tests.

    :param dataset: Real local Lance snapshot used by the test.
    :param config: Backfill policy represented by the context.
    :param identity: Invocation provenance under test.
    :param checkpoint: Resolved checkpoint identity.
    :param artifact: Resolved artifact identity.
    :returns: Fully populated internal context.
    """
    add_config = AddEmbeddingsConfig(
        lance_uri=config.lance_uri,
        embeddings=(config.embedding,),
        checkpoints={config.embedding: checkpoint},
        device="cuda",
        batch_size=config.batch_size,
        build_index=config.build_index,
        num_partitions=config.num_partitions,
    )
    return _BackfillContext(
        dataset=dataset,
        storage_options=None,
        spec=EMBEDDING_REGISTRY[config.embedding],
        add_config=add_config,
        checkpoint=checkpoint,
        artifact=artifact,
        source_version=dataset.version,
        sample_rate=44_100,
        total_rows=dataset.count_rows(),
        identity=identity,
    )


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
    report = _FragmentReport.model_validate(
        _transform_fragment(
            _FragmentTask(
                uri=str(uri),
                storage_options=None,
                branch="embeddings-2985",
                source_version=candidate.version,
                fragment_id=fragment_id,
                embedding="clap",
                checkpoint="unused-checkpoint",
                sample_rate=44_100,
                batch_size=2,
                artifact="clap:test-artifact",
                run_id="1" * 32,
            )
        ),
        strict=True,
    )

    config = EmbeddingBackfillConfig(
        lance_uri=str(uri),
        branch="embeddings-2985",
        embedding="clap",
        workers=1,
        batch_size=2,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
        build_index=False,
    )
    context = _summary_context(
        dataset=candidate,
        config=config,
        identity=_RunIdentity(run_id="1" * 32, git_commit="a" * 40),
        checkpoint="unused-checkpoint",
        artifact="clap:test-artifact",
    )

    committed = _commit_reports(config, context, (report,))

    assert candidate.version == 1
    assert pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc))).names == [
        "clap"
    ]
    assert report.rows == 8
    assert committed.version == 2
    assert committed.schema.field("clap").metadata == {
        b"synth_setter.embedding.name": b"clap",
        b"synth_setter.embedding.artifact": b"clap:test-artifact",
    }
    expected = np.repeat(audio.mean(axis=(1, 2), keepdims=False)[:, None], 512, axis=1)
    actual = committed.take(range(8), columns=["clap"]).column(0).combine_chunks()
    np.testing.assert_allclose(actual.values.to_numpy().reshape(8, 512), expected)


def test_transform_fragment_writes_meanaudio_sequence_and_vector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A MeanAudio worker must publish the policy-defined sequence and pooled vector.

    :param tmp_path: Temporary directory for the real Lance dataset.
    :param monkeypatch: Fixture replacing only the heavyweight encoder loader.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill
    from synth_setter.pipeline.data.meanaudio import meanaudio_num_latent_frames

    audio = np.stack(
        (
            np.full((1, 4_410), 1.0, dtype=np.float32),
            np.full((1, 4_410), -0.5, dtype=np.float32),
        )
    )
    uri = tmp_path / "meanaudio-worker.lance"
    dataset = lance.write_dataset(
        pa.table({"audio": tensor_array(audio, np.dtype("float32"), (1, 4_410))}),
        uri,
    )
    candidate = dataset.create_branch("embeddings-2985", (None, dataset.version))

    def encode(values: np.ndarray, sample_rate: int) -> np.ndarray:
        frames = meanaudio_num_latent_frames(values.shape[-1], sample_rate)
        means = values.mean(axis=(1, 2), keepdims=True)
        return np.broadcast_to(means, (len(values), 20, frames)).copy()

    monkeypatch.setattr(backfill, "_worker_encoder", lambda *args: encode)
    report = _transform_fragment(
        _FragmentTask(
            uri=str(uri),
            storage_options=None,
            branch="embeddings-2985",
            source_version=candidate.version,
            fragment_id=candidate.get_fragments()[0].metadata.id,
            embedding="meanaudio_16k",
            checkpoint="unused-checkpoint",
            sample_rate=44_100,
            batch_size=2,
            artifact="meanaudio:test-artifact",
            run_id="2" * 32,
        )
    )
    config = EmbeddingBackfillConfig(
        lance_uri=str(uri),
        branch="embeddings-2985",
        embedding="meanaudio_16k",
        workers=1,
        batch_size=2,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
        build_index=False,
    )
    context = _summary_context(
        dataset=candidate,
        config=config,
        identity=_RunIdentity(run_id="2" * 32, git_commit="b" * 40),
        checkpoint="unused-checkpoint",
        artifact="meanaudio:test-artifact",
    )

    committed = _commit_reports(config, context, (report,))

    assert report.rows == 2
    assert pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc))).names == [
        "meanaudio_16k",
        "meanaudio_16k_vec",
    ]
    assert committed.schema.names == ["audio", "meanaudio_16k", "meanaudio_16k_vec"]
    expected_metadata = {
        b"synth_setter.embedding.name": b"meanaudio_16k",
        b"synth_setter.embedding.artifact": b"meanaudio:test-artifact",
    }
    assert committed.schema.field("meanaudio_16k").metadata == expected_metadata
    assert committed.schema.field("meanaudio_16k_vec").metadata == expected_metadata
    sequence = committed.take([0, 1], columns=["meanaudio_16k"]).column(0).combine_chunks()
    vector = committed.take([0, 1], columns=["meanaudio_16k_vec"]).column(0).combine_chunks()
    assert sequence.type == pa.fixed_shape_tensor(pa.float32(), (20, 3))
    assert vector.type == pa.list_(pa.float32(), 20)
    assert sequence.to_numpy_ndarray().shape == (2, 20, 3)
    assert vector.values.to_numpy().reshape(2, 20).shape == (2, 20)
    np.testing.assert_array_equal(
        sequence.to_numpy_ndarray(),
        np.array(
            [
                [[1.0, 1.0, 1.0]] * 20,
                [[-0.5, -0.5, -0.5]] * 20,
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        vector.values.to_numpy().reshape(2, 20),
        np.array([[1.0] * 20, [-0.5] * 20], dtype=np.float32),
    )


def test_fragment_deadline_cancels_pending_work_for_resumable_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired dispatch must cancel unfinished work instead of polling forever.

    :param tmp_path: Temporary path used for a valid strict config.
    :param monkeypatch: Fixture controlling monotonic time and Ray cancellation.
    """
    import ray

    import synth_setter.pipeline.data.backfill_embeddings as backfill

    cancelled: list[object] = []
    reference = cast("ray.ObjectRef", object())
    monkeypatch.setattr(backfill.time, "monotonic", lambda: 2.0)
    monkeypatch.setattr(ray, "cancel", lambda value, force: cancelled.append(value))
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
        timeout_seconds=1.0,
    )
    state = _DispatchState(
        pending=[reference],
        reports={},
        fragment_ids={0},
        total_rows=1,
        store=_report_store(
            config,
            _CacheIdentity(
                lance_uri=config.lance_uri,
                branch=config.branch,
                source_version=1,
                embedding="clap",
                checkpoint="checkpoint",
                sample_rate=44_100,
                batch_size=1,
                artifact="artifact",
                implementation_revision="b" * 40,
            ),
        ),
        run_id="6" * 32,
    )

    with pytest.raises(TimeoutError, match="retry resumes"):
        _poll_dispatch(state, config, started=0.0)
    assert cancelled == [reference]


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
        _embedding_promotion_table(tuple(float(value) for value in range(8)))
    )
    candidate_version = candidate.version

    result = promote_embedding_candidate(
        EmbeddingPromotionConfig(
            lance_uri=str(uri),
            candidate_branch="embeddings-2985",
            rollback_tag="pre-embeddings-2985",
            columns=("clap", "meanaudio_16k", "meanaudio_16k_vec"),
        )
    )

    main = lance.dataset(uri)
    assert result.source_version == 1
    assert result.candidate_version == candidate_version
    assert result.committed_version == 2
    assert main.version == 2
    assert main.schema.names == [
        "row_id",
        "audio",
        "clap",
        "meanaudio_16k",
        "meanaudio_16k_vec",
    ]
    assert main.schema.metadata == source.schema.metadata
    assert main.schema.field("clap") == candidate.schema.field("clap")
    assert main.schema.field("meanaudio_16k") == candidate.schema.field("meanaudio_16k")
    assert main.schema.field("meanaudio_16k_vec") == candidate.schema.field(
        "meanaudio_16k_vec"
    )
    embedding_columns = ["clap", "meanaudio_16k", "meanaudio_16k_vec"]
    assert main.take(range(8), columns=embedding_columns).equals(
        candidate.take(range(8), columns=embedding_columns)
    )
    assert main.take(range(8), columns=["row_id", "audio"]).equals(
        source.select(["row_id", "audio"])
    )
    assert lance.dataset(uri).checkout_version(("embeddings-2985", None)).version == (
        candidate_version
    )



def test_validate_promotion_rejects_clap_wrong_vector_width(tmp_path: Path) -> None:
    """A selected CLAP field must contain exactly 512 values per row.

    :param tmp_path: Holds the malformed real Lance candidate.
    """
    uri = tmp_path / "clap-width.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_clap_promotion_table(width=511))

    with pytest.raises(ValueError, match="512-element float32 vector"):
        _validate_promotion(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_validate_promotion_rejects_clap_wrong_element_type(tmp_path: Path) -> None:
    """A selected CLAP field must contain float32 values.

    :param tmp_path: Holds the malformed real Lance candidate.
    """
    uri = tmp_path / "clap-type.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_clap_promotion_table(value_type=pa.float64()))

    with pytest.raises(ValueError, match="512-element float32 vector"):
        _validate_promotion(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_validate_promotion_rejects_clap_wrong_embedding_name(tmp_path: Path) -> None:
    """A selected CLAP field must identify the CLAP registry policy.

    :param tmp_path: Holds the malformed real Lance candidate.
    """
    uri = tmp_path / "clap-name.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(
        _clap_promotion_table(
            metadata=_embedding_metadata("meanaudio_16k", _CLAP_ARTIFACT)
        )
    )

    with pytest.raises(ValueError, match="artifact identity"):
        _validate_promotion(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_validate_promotion_rejects_clap_missing_artifact_bytes(tmp_path: Path) -> None:
    """A selected CLAP field must carry a nonempty artifact identity.

    :param tmp_path: Holds the malformed real Lance candidate.
    """
    uri = tmp_path / "clap-artifact.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(
        _clap_promotion_table(metadata={_EMBEDDING_NAME_KEY: b"clap"})
    )

    with pytest.raises(ValueError, match="artifact bytes"):
        _validate_promotion(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_validate_promotion_rejects_partial_meanaudio_policy(tmp_path: Path) -> None:
    """Selecting only one MeanAudio output cannot publish a partial policy.

    :param tmp_path: Holds the incomplete real Lance candidate.
    """
    uri = tmp_path / "partial-meanaudio.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_meanaudio_promotion_table().select(["meanaudio_16k"]))

    with pytest.raises(ValueError, match="partial meanaudio_16k columns"):
        _validate_promotion(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("meanaudio_16k",),
            )
        )


def test_validate_promotion_rejects_inconsistent_meanaudio_artifacts(tmp_path: Path) -> None:
    """MeanAudio sequence and vector fields must share one artifact identity.

    :param tmp_path: Holds the inconsistent real Lance candidate.
    """
    uri = tmp_path / "meanaudio-artifact.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_meanaudio_promotion_table(vector_artifact=b"other-artifact"))

    with pytest.raises(ValueError, match="artifact identity"):
        _validate_promotion(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("meanaudio_16k", "meanaudio_16k_vec"),
            )
        )


def test_validate_promotion_rejects_unknown_selected_column(tmp_path: Path) -> None:
    """Promotion columns must belong to a registered embedding policy.

    :param tmp_path: Holds the unknown real Lance candidate.
    """
    uri = tmp_path / "unknown-column.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(pa.table({"experimental": [3, 4]}))

    with pytest.raises(ValueError, match="unknown embedding columns"):
        _validate_promotion(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("experimental",),
            )
        )


def test_promote_embedding_candidate_with_file_uri_publishes_candidate(
    tmp_path: Path,
) -> None:
    """Promotion must resolve a local file URI before copying candidate data.

    :param tmp_path: Temporary directory for the real Lance dataset.
    """
    uri = tmp_path / "file-uri.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_clap_promotion_table())

    result = promote_embedding_candidate(
        EmbeddingPromotionConfig(
            lance_uri=uri.as_uri(),
            candidate_branch="candidate",
            rollback_tag="rollback",
            columns=("clap",),
        )
    )

    assert result.committed_version == 2
    clap = lance.dataset(uri).to_table(columns=["clap"]).column("clap")
    assert clap.type == pa.list_(pa.float32(), 512)
    assert [row.as_py()[0] for row in clap] == [1.0, 2.0]


def test_promote_embedding_candidate_rejects_unselected_candidate_column(
    tmp_path: Path,
) -> None:
    """Promotion must not leak unrelated branch columns into main.

    :param tmp_path: Temporary directory for the real Lance dataset.
    """
    uri = tmp_path / "extra-column.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(
        _clap_promotion_table().append_column("experimental", pa.array([3, 4]))
    )

    with pytest.raises(ValueError, match="unselected columns"):
        promote_embedding_candidate(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_promote_embedding_candidate_rejects_modified_source_files(
    tmp_path: Path,
) -> None:
    """Promotion must reject a candidate that rewrote rollback source values.

    :param tmp_path: Temporary directory for the real Lance dataset.
    """
    uri = tmp_path / "modified-source.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.update({"row_id": "row_id + 10"})
    candidate.add_columns(_clap_promotion_table())

    with pytest.raises(ValueError, match="do not match the rollback source"):
        promote_embedding_candidate(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_promote_embedding_candidate_rejects_same_schema_independent_main_write(
    tmp_path: Path,
) -> None:
    """Schema equality must not masquerade as candidate publication provenance.

    :param tmp_path: Temporary directory for the real Lance dataset.
    """
    uri = tmp_path / "independent-main.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_clap_promotion_table())
    dataset.add_columns(_clap_promotion_table((9.0, 9.0)))

    with pytest.raises(ValueError, match="does not contain the validated candidate merge"):
        promote_embedding_candidate(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_promote_embedding_candidate_rejects_later_selected_value_mutation(
    tmp_path: Path,
) -> None:
    """Idempotent promotion must reject later rewrites of selected values.

    :param tmp_path: Holds the real promoted and subsequently mutated main branch.
    """
    uri = tmp_path / "mutated-promotion.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_clap_promotion_table())
    config = EmbeddingPromotionConfig(
        lance_uri=str(uri),
        candidate_branch="candidate",
        rollback_tag="rollback",
        columns=("clap",),
    )
    promote_embedding_candidate(config)
    lance.dataset(uri).update({"clap": str([9.0] * 512)})

    with pytest.raises(ValueError, match="validated candidate merge"):
        promote_embedding_candidate(config)


def _promotion_race_case(
    tmp_path: Path,
) -> tuple[EmbeddingPromotionConfig, _PromotionContext]:
    """Build validated real Lance state immediately before promotion.

    :param tmp_path: Holds the real main and candidate branch.
    :returns: Promotion config and validated publication context.
    """
    uri = tmp_path / "promotion-race.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_clap_promotion_table())
    config = EmbeddingPromotionConfig(
        lance_uri=str(uri),
        candidate_branch="candidate",
        rollback_tag="rollback",
        columns=("clap",),
    )
    return config, _validate_promotion(config)


def test_promote_embedding_candidate_retry_reports_idempotent_provenance(
    tmp_path: Path,
) -> None:
    """A retry reports idempotency under fresh run and stable source identities.

    :param tmp_path: Holds the real main and candidate branch.
    """
    config, _ = _promotion_race_case(tmp_path)

    first = promote_embedding_candidate(config)
    retry = promote_embedding_candidate(config)

    assert first.already_complete is False
    assert retry.already_complete is True
    assert first.run_id != retry.run_id
    assert len(first.run_id) == 32
    assert first.git_commit == retry.git_commit
    assert len(first.git_commit) == 40
    assert lance.dataset(config.lance_uri).version == 2


def test_publish_candidate_recovers_commit_visible_before_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain promotion error must recover the exact visible candidate merge.

    :param tmp_path: Holds the real main and candidate branch.
    :param monkeypatch: Injects an error after the real commit becomes visible.
    """
    config, context = _promotion_race_case(tmp_path)
    real_commit = lance.LanceDataset.commit

    def visible_then_raise(
        base_uri: str | Path | lance.LanceDataset,
        operation: lance.LanceOperation.BaseOperation,
        read_version: int | None = None,
        storage_options: dict[str, str] | None = None,
        *,
        commit_message: str | None = None,
    ) -> Never:
        real_commit(
            base_uri,
            operation,
            read_version=read_version,
            storage_options=storage_options,
            commit_message=commit_message,
        )
        raise RuntimeError("uncertain promotion")

    monkeypatch.setattr(lance.LanceDataset, "commit", visible_then_raise)

    committed, already_complete = _publish_candidate(config, context)

    assert already_complete is True
    assert committed.version == 2
    clap = committed.to_table(columns=["clap"]).column("clap")
    assert [row.as_py()[0] for row in clap] == [1.0, 2.0]


def test_publish_candidate_reraises_original_error_after_incompatible_advancement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain promotion must not accept an unrelated concurrent main write.

    :param tmp_path: Holds the real main and candidate branch.
    :param monkeypatch: Injects the original commit-boundary error.
    """
    config, context = _promotion_race_case(tmp_path)
    lance.dataset(config.lance_uri).add_columns(pa.table({"other": [3, 4]}))
    failure = RuntimeError("original promotion error")

    def fail_commit(
        base_uri: str | Path | lance.LanceDataset,
        operation: lance.LanceOperation.BaseOperation,
        read_version: int | None = None,
        storage_options: dict[str, str] | None = None,
        *,
        commit_message: str | None = None,
    ) -> Never:
        del base_uri, operation, read_version, storage_options, commit_message
        raise failure

    monkeypatch.setattr(lance.LanceDataset, "commit", fail_commit)

    with pytest.raises(RuntimeError, match="original promotion error") as caught:
        _publish_candidate(config, context)

    assert caught.value is failure


def test_publish_candidate_reraises_original_error_when_reopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovery-read failure must not replace the original commit error.

    :param tmp_path: Holds the real main and candidate branch.
    :param monkeypatch: Injects commit and recovery-read failures.
    """
    config, context = _promotion_race_case(tmp_path)
    failure = RuntimeError("original promotion error")

    def fail_reopen(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise OSError("reopen failed")

    def fail_commit(
        base_uri: str | Path | lance.LanceDataset,
        operation: lance.LanceOperation.BaseOperation,
        read_version: int | None = None,
        storage_options: dict[str, str] | None = None,
        *,
        commit_message: str | None = None,
    ) -> Never:
        del base_uri, operation, read_version, storage_options, commit_message
        monkeypatch.setattr(lance, "dataset", fail_reopen)
        raise failure

    monkeypatch.setattr(lance.LanceDataset, "commit", fail_commit)

    with pytest.raises(RuntimeError, match="original promotion error") as caught:
        _publish_candidate(config, context)

    assert caught.value is failure


def test_promote_cli_publishes_candidate_and_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real CLI must dispatch promotion and emit its machine-readable result.

    :param tmp_path: Temporary directory for the real Lance dataset.
    :param monkeypatch: Supplies command-line arguments.
    :param capsys: Captures the machine-readable result.
    """
    uri = tmp_path / "cli.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(_clap_promotion_table())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_embeddings",
            "promote",
            "--lance-uri",
            str(uri),
            "--candidate-branch",
            "candidate",
            "--rollback-tag",
            "rollback",
            "--columns",
            "clap",
        ],
    )

    main()

    assert json.loads(capsys.readouterr().out)["already_complete"] is False
    assert lance.dataset(uri).schema.names == ["row_id", "clap"]


def test_ensure_embedding_index_recovers_concurrent_matching_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent matching index must satisfy the requested final state.

    :param tmp_path: Holds the real indexed Lance dataset.
    :param monkeypatch: Injects a commit conflict after the competing index publishes.
    """
    from synth_setter.pipeline.data import add_embeddings

    uri = tmp_path / "index-race.lance"
    vectors = pa.array(
        np.ones((256, 512), dtype=np.float32).tolist(),
        type=pa.list_(pa.float32(), 512),
    )
    dataset = lance.write_dataset(pa.table({"clap": vectors}), uri)
    competitor = lance.dataset(uri)
    config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("clap",),
        build_index=True,
        num_partitions=1,
    )
    real_build_index = add_embeddings.build_index
    index = EMBEDDING_REGISTRY["clap"].index
    assert index is not None

    def publish_then_conflict(*args: object, **kwargs: object) -> bool:
        real_build_index(
            competitor,
            "clap",
            index=index,
            config=config,
        )
        raise RuntimeError("Retryable commit conflict")

    monkeypatch.setattr(add_embeddings, "build_index", publish_then_conflict)

    latest, built = _ensure_embedding_index(dataset, EMBEDDING_REGISTRY["clap"], config)

    assert built is False
    assert len(latest.list_indices()) == 1


def test_embedding_complete_wrong_vector_width_rejected(tmp_path: Path) -> None:
    """Matching metadata must not hide a malformed CLAP schema.

    :param tmp_path: Holds the malformed real Lance candidate.
    """
    field = pa.field("clap", pa.list_(pa.float32(), 1)).with_metadata(
        {
            b"synth_setter.embedding.name": b"clap",
            b"synth_setter.embedding.artifact": b"artifact",
        }
    )
    dataset = lance.write_dataset(
        pa.Table.from_arrays(
            [pa.array([[1.0]], type=pa.list_(pa.float32(), 1))],
            schema=pa.schema([field]),
        ),
        tmp_path / "malformed.lance",
    )

    with pytest.raises(ValueError, match="schema"):
        _embedding_is_complete(dataset, EMBEDDING_REGISTRY["clap"], b"artifact")


def test_implementation_revision_changes_with_python_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume identity must change when executed Python changes without a commit.

    :param tmp_path: Holds two revisions of a synthetic package.
    :param monkeypatch: Redirects implementation hashing to the synthetic package.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill

    module = tmp_path / "synth_setter/pipeline/data/backfill_embeddings.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n")
    monkeypatch.setattr(backfill, "__file__", str(module))
    first = _implementation_revision("a" * 40)

    module.write_text("VALUE = 2\n")

    assert _implementation_revision("a" * 40) != first


def test_source_git_sha_ignores_caller_git_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance must resolve from the synth-setter checkout, not caller CWD.

    :param tmp_path: Holds a distinct caller git repository.
    :param monkeypatch: Changes the caller working directory.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill

    caller = tmp_path / "caller"
    caller.mkdir()
    repo = Repo.init(caller)
    actor = Actor("Caller", "caller@example.com")
    repo.index.commit("caller", author=actor, committer=actor)
    caller_sha = repo.head.commit.hexsha
    source_root = Path(cast("str", Repo(search_parent_directories=True).working_tree_dir))
    source_sha = Repo(source_root).head.commit.hexsha
    monkeypatch.setattr(
        backfill,
        "__file__",
        str(source_root / "src/synth_setter/pipeline/data/backfill_embeddings.py"),
    )
    monkeypatch.chdir(caller)

    assert _resolve_source_git_sha() == source_sha
    assert source_sha != caller_sha


def test_promotion_unknown_git_sha_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Promotion must stop before storage access when source provenance is unavailable.

    :param monkeypatch: Makes git resolution return its unknown sentinel.
    """
    from synth_setter.utils import logging_utils

    monkeypatch.setattr(logging_utils, "resolve_git_sha", lambda repo_root=None: "unknown")

    with pytest.raises(RuntimeError, match="validated source git SHA"):
        promote_embedding_candidate(
            EmbeddingPromotionConfig(
                lance_uri="unused",
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_backfill_invalid_git_sha_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backfill must stop before Ray startup when source provenance is invalid.

    :param monkeypatch: Enables the GPU gate and supplies an uppercase SHA.
    """
    import torch

    from synth_setter.utils import logging_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(logging_utils, "resolve_git_sha", lambda repo_root=None: "A" * 40)
    config = EmbeddingBackfillConfig(
        lance_uri="unused",
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
    )

    with pytest.raises(RuntimeError, match="validated source git SHA"):
        backfill_embedding(config)


def test_candidate_data_file_base_zero_traversal_rejected() -> None:
    """Main-relative promotion files must remain inside the data directory."""
    data_file = {"base_id": 0, "path": "../foreign.lance"}

    with pytest.raises(ValueError, match="unsafe candidate data path"):
        _normalise_candidate_data_file(data_file)


def test_resume_identity_competing_operations_only_one_claims_valid_json(
    tmp_path: Path,
) -> None:
    """A shared empty resume directory must have one atomic identity winner.

    :param tmp_path: Isolated local reconciliation directory.
    """
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=2,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
        resume_dir=tmp_path / "resume",
    )
    first = _CacheIdentity(
        lance_uri=config.lance_uri,
        branch=config.branch,
        source_version=1,
        embedding="clap",
        checkpoint="checkpoint",
        sample_rate=44_100,
        batch_size=2,
        artifact="artifact",
        implementation_revision="a" * 40,
    )
    second = first.model_copy(update={"artifact": "other-artifact"})
    store = _report_store(config, first)

    def claim(identity: _CacheIdentity) -> str:
        try:
            _load_reports(store, identity, {7})
        except ValueError:
            return "rejected"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(claim, (first, second)))

    winner = _CacheIdentity.model_validate_json(
        (store.local_dir / "identity.json").read_text(), strict=True
    )
    assert sorted(outcomes) == ["claimed", "rejected"]
    assert winner == first or winner == second


def test_implementation_revision_scopes_default_and_explicit_resume_cache(
    tmp_path: Path,
) -> None:
    """Implementation changes must not reconcile reports across worker behavior.

    :param tmp_path: Isolated explicit-cache directory.
    """
    default_config = EmbeddingBackfillConfig(
        lance_uri="s3://bucket/source.lance",
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=2,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
    )
    first = _CacheIdentity(
        lance_uri=default_config.lance_uri,
        branch=default_config.branch,
        source_version=1,
        embedding="clap",
        checkpoint="checkpoint",
        sample_rate=44_100,
        batch_size=2,
        artifact="artifact",
        implementation_revision="a" * 40,
    )
    second = first.model_copy(update={"implementation_revision": "b" * 40})
    first_store = _report_store(default_config, first)
    second_store = _report_store(default_config, second)

    explicit_config = default_config.model_copy(
        update={
            "lance_uri": str(tmp_path / "source.lance"),
            "resume_dir": tmp_path / "resume",
        }
    )
    explicit_first = first.model_copy(update={"lance_uri": explicit_config.lance_uri})
    explicit_second = second.model_copy(update={"lance_uri": explicit_config.lance_uri})
    explicit_store = _report_store(explicit_config, explicit_first)
    _load_reports(explicit_store, explicit_first, set())

    assert first_store.local_dir != second_store.local_dir
    assert first_store.remote_uri != second_store.remote_uri
    with pytest.raises(ValueError, match="another identity"):
        _load_reports(explicit_store, explicit_second, set())


def test_persisted_report_filename_carries_worker_attempt_and_round_trips(
    tmp_path: Path,
) -> None:
    """Durable report names must preserve worker-generated provenance.

    :param tmp_path: Isolated local reconciliation directory.
    """
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=2,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
        resume_dir=tmp_path / "resume",
    )
    identity = _CacheIdentity(
        lance_uri=config.lance_uri,
        branch=config.branch,
        source_version=1,
        embedding="clap",
        checkpoint="checkpoint",
        sample_rate=44_100,
        batch_size=2,
        artifact="artifact",
        implementation_revision="a" * 40,
    )
    report = _FragmentReport(
        fragment_id=7,
        metadata_json="{}",
        schema_ipc="schema",
        pid=12,
        rows=8,
        elapsed_seconds=1.0,
        peak_rss_bytes=10,
        peak_gpu_allocated_bytes=20,
        peak_gpu_reserved_bytes=30,
        run_id="1" * 32,
        worker_id="2" * 32,
        attempt_uuid="3" * 32,
    )
    store = _report_store(config, identity)
    _load_reports(store, identity, {7})

    _persist_report(store, report)

    path = store.local_dir / f"fragment-7-{report.worker_id}-{report.attempt_uuid}.json"
    assert path.is_file()
    assert _load_reports(store, identity, {7}) == {7: report}


def test_remote_report_hydrates_on_fresh_host_and_commits(
    fake_r2_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh host must publish a usable fragment hydrated through real rclone.

    :param fake_r2_remote: Local-backed R2 remote carrying shared reports.
    :param tmp_path: Holds the real Lance source and independent host caches.
    :param monkeypatch: Replaces only the heavyweight CLAP encoder.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill

    audio = np.arange(16, dtype=np.float32).reshape(2, 1, 8) / 16
    uri = tmp_path / "source.lance"
    root = lance.write_dataset(
        pa.table({"audio": tensor_array(audio, np.dtype("float32"), (1, 8))}), uri
    )
    candidate = root.create_branch("candidate", (None, root.version))

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.repeat(mono.mean(axis=1, keepdims=True), 512, axis=1)

    monkeypatch.setattr(backfill, "_worker_encoder", lambda *args: encode)
    report = _transform_fragment(
        _FragmentTask(
            uri=str(uri),
            storage_options=None,
            branch="candidate",
            source_version=candidate.version,
            fragment_id=candidate.get_fragments()[0].metadata.id,
            embedding="clap",
            checkpoint="checkpoint",
            sample_rate=44_100,
            batch_size=2,
            artifact="artifact",
            run_id="1" * 32,
        )
    )
    shared = EmbeddingBackfillConfig(
        lance_uri="s3://bucket/source.lance",
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=2,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
    )
    identity = _CacheIdentity(
        lance_uri=shared.lance_uri,
        branch=shared.branch,
        source_version=1,
        embedding="clap",
        checkpoint="checkpoint",
        sample_rate=44_100,
        batch_size=2,
        artifact="artifact",
        implementation_revision="a" * 40,
    )
    first_store = _report_store(
        shared.model_copy(update={"resume_dir": tmp_path / "first-cache"}), identity
    )
    second_store = _report_store(
        shared.model_copy(update={"resume_dir": tmp_path / "second-cache"}), identity
    )
    _load_reports(first_store, identity, {report.fragment_id})
    _persist_report(first_store, report)
    shutil.rmtree(first_store.local_dir)

    hydrated = _load_reports(second_store, identity, {report.fragment_id})
    local_config = shared.model_copy(update={"lance_uri": str(uri)})
    context = _summary_context(
        dataset=candidate,
        config=local_config,
        identity=_RunIdentity(run_id="1" * 32, git_commit="a" * 40),
        checkpoint="checkpoint",
        artifact="artifact",
    )
    committed = _commit_reports(local_config, context, tuple(hydrated.values()))

    values = committed.take([0, 1], columns=["clap"]).column(0).combine_chunks()
    np.testing.assert_array_equal(
        values.values.to_numpy().reshape(2, 512),
        np.repeat(audio.mean(axis=(1, 2), keepdims=False)[:, None], 512, axis=1),
    )
    assert first_store.remote_uri == second_store.remote_uri
    assert (fake_r2_remote / "bucket/source.lance/metadata/workers").is_dir()


def test_summarize_backfill_uses_only_current_reports_for_runtime_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resumed worker reuse and peaks must not contaminate invocation telemetry.

    :param tmp_path: Holds the real result dataset.
    :param monkeypatch: Fixes elapsed time for exact throughput assertions.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill

    dataset = lance.write_dataset(pa.table({"row_id": [0, 1, 2, 3]}), tmp_path / "rows.lance")
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "rows.lance"),
        branch="candidate",
        embedding="clap",
        workers=2,
        batch_size=2,
        tasks_per_worker=2,
        gpu_per_worker=0.5,
    )
    resumed_first = _FragmentReport(
        fragment_id=0,
        metadata_json="{}",
        schema_ipc="schema",
        pid=77,
        rows=1,
        elapsed_seconds=9.0,
        peak_rss_bytes=9_000,
        peak_gpu_allocated_bytes=8_000,
        peak_gpu_reserved_bytes=7_000,
        run_id="1" * 32,
        worker_id="a" * 32,
        attempt_uuid="1" * 32,
    )
    resumed = (
        resumed_first,
        resumed_first.model_copy(update={"fragment_id": 1, "attempt_uuid": "2" * 32}),
    )
    current_first = _FragmentReport(
        fragment_id=2,
        metadata_json="{}",
        schema_ipc="schema",
        pid=77,
        rows=1,
        elapsed_seconds=1.0,
        peak_rss_bytes=300,
        peak_gpu_allocated_bytes=200,
        peak_gpu_reserved_bytes=100,
        run_id="2" * 32,
        worker_id="b" * 32,
        attempt_uuid="3" * 32,
    )
    current = (
        current_first,
        current_first.model_copy(update={"fragment_id": 3, "attempt_uuid": "4" * 32}),
    )
    context = _summary_context(
        dataset=dataset,
        config=config,
        identity=_RunIdentity(run_id="2" * 32, git_commit="c" * 40),
        checkpoint="resolved-checkpoint",
        artifact="resolved-artifact",
    )
    monkeypatch.setattr(backfill.time, "monotonic", lambda: 12.0)

    result = _summarize_backfill(
        config=config,
        context=context,
        outcome=_BackfillOutcome(
            dataset=dataset,
            dispatch=_DispatchResult(all_reports=(*resumed, *current), current_reports=current),
            data_version=2,
            index_built=False,
            already_complete=False,
        ),
        started=10.0,
    )

    assert result.current_rows == 2
    assert result.current_fragments == 2
    assert result.resumed_rows == 2
    assert result.resumed_fragments == 2
    assert result.rows_per_second == 1.0
    assert result.worker_processes == 1
    assert result.max_tasks_per_process == 2
    assert result.peak_rss_bytes == 300
    assert result.peak_gpu_allocated_bytes == 200
    assert result.peak_gpu_reserved_bytes == 100


def test_log_dispatch_progress_rates_only_current_invocation_rows(tmp_path: Path) -> None:
    """Throughput excludes resumed rows while ETA uses all completed rows.

    :param tmp_path: Holds the unused reconciliation directory.
    """
    from structlog.testing import capture_logs

    resumed = _FragmentReport(
        fragment_id=0,
        metadata_json="{}",
        schema_ipc="schema",
        pid=1,
        rows=900,
        elapsed_seconds=100.0,
        peak_rss_bytes=0,
        peak_gpu_allocated_bytes=0,
        peak_gpu_reserved_bytes=0,
        run_id="1" * 32,
        worker_id="2" * 32,
        attempt_uuid="3" * 32,
    )
    current = resumed.model_copy(
        update={
            "fragment_id": 1,
            "rows": 10,
            "run_id": "4" * 32,
            "attempt_uuid": "5" * 32,
        }
    )
    state = _DispatchState(
        pending=[],
        reports={0: resumed, 1: current},
        fragment_ids={0, 1, 2},
        total_rows=1_000,
        store=_ReportStore(local_dir=tmp_path / "reports", remote_uri=None),
        run_id="4" * 32,
    )
    config = EmbeddingBackfillConfig(
        lance_uri="unused",
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
    )

    with capture_logs() as logs:
        _log_dispatch_progress(
            state,
            config=config,
            rows_done=910,
            current_rows_done=10,
            started=100.0,
            now=110.0,
        )

    assert logs == [
        {
            "branch": "candidate",
            "elapsed_seconds": 10.0,
            "embedding": "clap",
            "eta_seconds": 90.0,
            "event": "embedding_backfill_progress",
            "fragments": 2,
            "log_level": "info",
            "rows": 910,
            "rows_per_second": 1.0,
            "total_fragments": 3,
            "total_rows": 1_000,
        }
    ]


def test_prepare_dispatch_configures_bounded_exception_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray tasks must retry exceptions only within the coordinator deadline.

    :param tmp_path: Holds source and reconciliation paths.
    :param monkeypatch: Captures the Ray remote boundary configuration.
    """
    import ray

    dataset = lance.write_dataset(pa.table({"row_id": [0]}), tmp_path / "dispatch.lance")
    captured: dict[str, object] = {}

    class RemoteFunction:
        def remote(self, task: _FragmentTask) -> object:
            captured["task"] = task
            return object()

    def remote(**options: object) -> object:
        captured.update(options)

        def decorate(function: object) -> RemoteFunction:
            captured["function"] = function
            return RemoteFunction()

        return decorate

    monkeypatch.setattr(ray, "remote", remote)
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "dispatch.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=3,
        gpu_per_worker=1.0,
        resume_dir=tmp_path / "resume",
    )
    context = _summary_context(
        dataset=dataset,
        config=config,
        identity=_RunIdentity(run_id="1" * 32, git_commit="a" * 40),
        checkpoint="checkpoint",
        artifact="artifact",
    )

    state = _prepare_dispatch(config, context)

    assert len(state.pending) == 1
    assert captured["max_retries"] == 2
    assert captured["retry_exceptions"] is True
    assert captured["function"] is _transform_fragment
    task = captured["task"]
    assert isinstance(task, _FragmentTask)
    assert task.run_id == "1" * 32


def test_nested_r2_candidate_copy_preserves_relative_path(
    fake_r2_remote: Path,
) -> None:
    """Candidate data copies must retain nested object paths under main data.

    :param fake_r2_remote: Local-backed R2 tree used for the real directory copy.
    """
    source = fake_r2_remote / "bucket/dataset/tree/candidate/data/nested/deep"
    source.mkdir(parents=True)
    (source / "fragment.lance").write_text("payload")

    _copy_candidate_data_directory("s3://bucket/dataset", "candidate")

    copied = fake_r2_remote / "bucket/dataset/data/nested/deep/fragment.lance"
    assert copied.read_text() == "payload"


def test_poll_dispatch_returns_resumed_and_current_reports_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray success must preserve publication order without relabeling resumed work.

    :param tmp_path: Holds durable current-invocation reports.
    :param monkeypatch: Supplies deterministic Ray completion values.
    """
    import ray

    reference = cast("ray.ObjectRef", object())
    resumed = _FragmentReport(
        fragment_id=0,
        metadata_json="{}",
        schema_ipc="schema",
        pid=5,
        rows=2,
        elapsed_seconds=1.0,
        peak_rss_bytes=10,
        peak_gpu_allocated_bytes=20,
        peak_gpu_reserved_bytes=30,
        run_id="1" * 32,
        worker_id="2" * 32,
        attempt_uuid="3" * 32,
    )
    current = _FragmentReport(
        fragment_id=1,
        metadata_json="{}",
        schema_ipc="schema",
        pid=5,
        rows=3,
        elapsed_seconds=1.0,
        peak_rss_bytes=11,
        peak_gpu_allocated_bytes=21,
        peak_gpu_reserved_bytes=31,
        run_id="4" * 32,
        worker_id="5" * 32,
        attempt_uuid="6" * 32,
    )
    store = _ReportStore(local_dir=tmp_path / "reports", remote_uri=None)
    store.local_dir.mkdir()
    state = _DispatchState(
        pending=[reference],
        reports={0: resumed},
        fragment_ids={0, 1},
        total_rows=5,
        store=store,
        run_id="4" * 32,
    )
    monkeypatch.setattr(ray, "wait", lambda *args, **kwargs: ([reference], []))
    monkeypatch.setattr(ray, "get", lambda value: current)
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
    )

    result = _poll_dispatch(state, config, started=time.monotonic())

    assert result.all_reports == (resumed, current)
    assert result.current_reports == (current,)


def test_poll_dispatch_mismatched_run_cancels_remaining_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A report from another invocation must cancel outstanding Ray work.

    :param tmp_path: Holds the unused report store.
    :param monkeypatch: Supplies mismatched Ray provenance and records cancellation.
    """
    import ray

    ready = cast("ray.ObjectRef", object())
    pending = cast("ray.ObjectRef", object())
    cancelled: list[object] = []
    report = _FragmentReport(
        fragment_id=0,
        metadata_json="{}",
        schema_ipc="schema",
        pid=5,
        rows=1,
        elapsed_seconds=1.0,
        peak_rss_bytes=10,
        peak_gpu_allocated_bytes=20,
        peak_gpu_reserved_bytes=30,
        run_id="7" * 32,
        worker_id="5" * 32,
        attempt_uuid="6" * 32,
    )
    store = _ReportStore(local_dir=tmp_path / "reports", remote_uri=None)
    store.local_dir.mkdir()
    state = _DispatchState(
        pending=[ready, pending],
        reports={},
        fragment_ids={0, 1},
        total_rows=2,
        store=store,
        run_id="4" * 32,
    )
    monkeypatch.setattr(ray, "wait", lambda *args, **kwargs: ([ready], [pending]))
    monkeypatch.setattr(ray, "get", lambda value: report)
    monkeypatch.setattr(ray, "cancel", lambda value, force: cancelled.append(value))
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
    )

    with pytest.raises(ValueError, match="unexpected worker report"):
        _poll_dispatch(state, config, started=time.monotonic())
    assert cancelled == [pending]


def test_poll_dispatch_duplicate_report_cancels_remaining_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate fragment report must cancel outstanding Ray work.

    :param tmp_path: Holds the unused report store.
    :param monkeypatch: Supplies duplicate Ray output and records cancellation.
    """
    import ray

    ready = cast("ray.ObjectRef", object())
    pending = cast("ray.ObjectRef", object())
    cancelled: list[object] = []
    existing = _FragmentReport(
        fragment_id=0,
        metadata_json="{}",
        schema_ipc="schema",
        pid=5,
        rows=1,
        elapsed_seconds=1.0,
        peak_rss_bytes=10,
        peak_gpu_allocated_bytes=20,
        peak_gpu_reserved_bytes=30,
        run_id="1" * 32,
        worker_id="2" * 32,
        attempt_uuid="3" * 32,
    )
    duplicate = existing.model_copy(
        update={
            "run_id": "4" * 32,
            "worker_id": "5" * 32,
            "attempt_uuid": "6" * 32,
        }
    )
    store = _ReportStore(local_dir=tmp_path / "reports", remote_uri=None)
    store.local_dir.mkdir()
    state = _DispatchState(
        pending=[ready, pending],
        reports={0: existing},
        fragment_ids={0, 1},
        total_rows=2,
        store=store,
        run_id="4" * 32,
    )
    monkeypatch.setattr(ray, "wait", lambda *args, **kwargs: ([ready], [pending]))
    monkeypatch.setattr(ray, "get", lambda value: duplicate)
    monkeypatch.setattr(ray, "cancel", lambda value, force: cancelled.append(value))
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
    )

    with pytest.raises(ValueError, match="unexpected worker report"):
        _poll_dispatch(state, config, started=time.monotonic())
    assert cancelled == [pending]


def _clap_commit_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[EmbeddingBackfillConfig, _BackfillContext, _FragmentReport]:
    """Build a real branch and uncommitted CLAP fragment for publication tests.

    :param tmp_path: Holds the real branch-local Lance dataset.
    :param monkeypatch: Replaces only the heavyweight encoder loader.
    :returns: Backfill config, immutable context, and worker report.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill

    audio = np.arange(16, dtype=np.float32).reshape(2, 1, 8) / 16
    uri = tmp_path / "commit.lance"
    root = lance.write_dataset(
        pa.table(
            {
                "row_id": [0, 1],
                "audio": tensor_array(audio, np.dtype("float32"), (1, 8)),
            }
        ),
        uri,
    )
    candidate = root.create_branch("candidate", (None, root.version))

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.repeat(mono.mean(axis=1, keepdims=True), 512, axis=1)

    monkeypatch.setattr(backfill, "_worker_encoder", lambda *args: encode)
    report = _transform_fragment(
        _FragmentTask(
            uri=str(uri),
            storage_options=None,
            branch="candidate",
            source_version=candidate.version,
            fragment_id=candidate.get_fragments()[0].metadata.id,
            embedding="clap",
            checkpoint="checkpoint",
            sample_rate=44_100,
            batch_size=1,
            artifact="artifact",
            run_id="1" * 32,
        )
    )
    config = EmbeddingBackfillConfig(
        lance_uri=str(uri),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
        build_index=False,
    )
    context = _summary_context(
        dataset=candidate,
        config=config,
        identity=_RunIdentity(
            run_id="1" * 32,
            git_commit="a" * 40,
            implementation_revision="a" * 40,
        ),
        checkpoint="checkpoint",
        artifact="artifact",
    )
    return config, context, report


def test_commit_reports_rejects_worker_schema_with_extra_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker output schemas must contain exactly the policy output columns.

    :param tmp_path: Holds the real branch-local Lance dataset.
    :param monkeypatch: Replaces only the heavyweight encoder loader.
    """
    config, context, report = _clap_commit_case(tmp_path, monkeypatch)
    schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc)))
    schema_with_extra = schema.append(pa.field("unexpected", pa.int64()))
    modified = report.model_copy(
        update={
            "schema_ipc": base64.b64encode(
                schema_with_extra.serialize().to_pybytes()
            ).decode()
        }
    )

    with pytest.raises(ValueError, match="exactly the policy output columns"):
        _commit_reports(config, context, (modified,))


def test_commit_reports_publishes_and_recovers_same_local_lance_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated commit after uncertain success must recover the published merge.

    :param tmp_path: Holds the real branch-local Lance dataset.
    :param monkeypatch: Replaces only the heavyweight encoder loader.
    """
    config, context, report = _clap_commit_case(tmp_path, monkeypatch)

    committed = _commit_reports(config, context, (report,))
    recovered = _commit_reports(config, context, (report,))

    assert committed.version == context.source_version + 1
    assert recovered.version == committed.version
    assert recovered.schema.names == ["row_id", "audio", "clap"]


def test_commit_reports_recovers_commit_that_became_visible_before_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain commit error must recover the exact visible Lance merge.

    :param tmp_path: Holds the real branch-local Lance dataset.
    :param monkeypatch: Injects an error after the real commit becomes visible.
    """
    config, context, report = _clap_commit_case(tmp_path, monkeypatch)
    real_commit = lance.LanceDataset.commit

    def visible_then_raise(
        base_uri: str | Path | lance.LanceDataset,
        operation: lance.LanceOperation.BaseOperation,
        read_version: int | None = None,
        storage_options: dict[str, str] | None = None,
        *,
        commit_message: str | None = None,
    ) -> Never:
        real_commit(
            base_uri,
            operation,
            read_version=read_version,
            storage_options=storage_options,
            commit_message=commit_message,
        )
        raise RuntimeError("uncertain commit")

    monkeypatch.setattr(lance.LanceDataset, "commit", visible_then_raise)

    recovered = _commit_reports(config, context, (report,))

    assert recovered.version == context.source_version + 1
    assert recovered.count_rows() == context.total_rows


def test_commit_reports_reraises_after_visible_merge_and_non_index_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery must reject a visible merge followed by a data mutation.

    :param tmp_path: Holds the real branch-local Lance dataset.
    :param monkeypatch: Injects mutation and failure after the real commit.
    """
    config, context, report = _clap_commit_case(tmp_path, monkeypatch)
    real_commit = lance.LanceDataset.commit
    failure = RuntimeError("original commit error")

    def publish_mutate_then_raise(
        base_uri: str | Path | lance.LanceDataset,
        operation: lance.LanceOperation.BaseOperation,
        read_version: int | None = None,
        storage_options: dict[str, str] | None = None,
        *,
        commit_message: str | None = None,
    ) -> Never:
        committed = real_commit(
            base_uri,
            operation,
            read_version=read_version,
            storage_options=storage_options,
            commit_message=commit_message,
        )
        committed.update({"row_id": "row_id + 1"})
        raise failure

    monkeypatch.setattr(lance.LanceDataset, "commit", publish_mutate_then_raise)

    with pytest.raises(RuntimeError, match="original commit error") as caught:
        _commit_reports(config, context, (report,))

    assert caught.value is failure


def test_run_backfill_recovered_merge_followed_by_index_reports_merge_data_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovered index snapshots must retain the exact embedding merge version.

    :param tmp_path: Holds the real indexed candidate branch.
    :param monkeypatch: Supplies reconciled reports to the coordinator.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill

    config, context, report = _clap_commit_case(tmp_path, monkeypatch)
    committed = _commit_reports(config, context, (report,))
    committed.create_scalar_index("row_id", "BTREE")
    monkeypatch.setattr(backfill, "_prepare_context", lambda selected, identity: context)
    monkeypatch.setattr(
        backfill,
        "_prepare_dispatch",
        lambda selected, prepared: _DispatchState(
            pending=[],
            reports={report.fragment_id: report},
            fragment_ids={report.fragment_id},
            total_rows=context.total_rows,
            store=_ReportStore(local_dir=tmp_path / "unused", remote_uri=None),
            run_id=prepared.identity.run_id,
        ),
    )

    result = _run_backfill(config, context.identity, started=time.monotonic())

    assert result.data_version == context.source_version + 1
    assert result.final_version == context.source_version + 2


def test_run_backfill_already_complete_indexed_input_reports_input_data_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already-complete results identify the indexed input snapshot as their data version.

    :param tmp_path: Holds the real indexed candidate branch.
    :param monkeypatch: Supplies the already-complete input context.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill

    config, context, report = _clap_commit_case(tmp_path, monkeypatch)
    committed = _commit_reports(config, context, (report,))
    committed.create_scalar_index("row_id", "BTREE")
    latest = lance.dataset(config.lance_uri).checkout_version((config.branch, None))
    complete_context = replace(
        context,
        dataset=latest,
        source_version=latest.version,
    )
    monkeypatch.setattr(
        backfill, "_prepare_context", lambda selected, identity: complete_context
    )

    result = _run_backfill(config, complete_context.identity, started=time.monotonic())

    assert result.already_complete is True
    assert result.data_version == latest.version
    assert result.final_version == latest.version


def test_write_backfill_result_serializes_resolved_audit_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Result JSON must preserve resolved model, worker, and source identities.

    :param tmp_path: Holds the persisted result JSON.
    :param capsys: Captures the identical stdout payload.
    """
    result = EmbeddingBackfillResult(
        run_id="1" * 32,
        git_commit="a" * 40,
        implementation_revision=f"{'a' * 40}:digest",
        branch="candidate",
        embedding="clap",
        checkpoint="resolved-checkpoint",
        artifact="resolved-artifact",
        workers=2,
        batch_size=4,
        tasks_per_worker=3,
        gpu_per_worker=0.5,
        rows=8,
        fragments=2,
        current_rows=8,
        current_fragments=2,
        resumed_rows=0,
        resumed_fragments=0,
        source_version=1,
        data_version=2,
        final_version=2,
        elapsed_seconds=4.0,
        rows_per_second=2.0,
        worker_processes=2,
        max_tasks_per_process=1,
        peak_rss_bytes=10,
        peak_gpu_allocated_bytes=20,
        peak_gpu_reserved_bytes=30,
        already_complete=False,
        index_built=False,
    )
    destination = tmp_path / "result.json"

    _write_result(result, destination)

    payload = json.loads(destination.read_text())
    assert json.loads(capsys.readouterr().out) == payload
    assert payload["checkpoint"] == "resolved-checkpoint"
    assert payload["artifact"] == "resolved-artifact"
    assert payload["workers"] == 2
    assert payload["batch_size"] == 4
    assert payload["tasks_per_worker"] == 3
    assert payload["gpu_per_worker"] == 0.5
    assert payload["git_commit"] == "a" * 40
    assert payload["implementation_revision"] == f"{'a' * 40}:digest"
    assert payload["run_id"] == "1" * 32


def test_backfill_all_resumed_flow_commits_hydrated_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coordinator must publish hydrated reports without relabeling current work.

    :param tmp_path: Holds the real Lance dataset.
    :param monkeypatch: Replaces GPU/Ray process boundaries while retaining coordinator logic.
    """
    import ray
    import torch

    import synth_setter.pipeline.data.backfill_embeddings as backfill

    audio = np.arange(16, dtype=np.float32).reshape(2, 1, 8) / 16
    uri = tmp_path / "coordinator.lance"
    root = lance.write_dataset(
        pa.table({"audio": tensor_array(audio, np.dtype("float32"), (1, 8))}), uri
    )
    candidate = root.create_branch("candidate", (None, root.version))

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.repeat(mono.mean(axis=1, keepdims=True), 512, axis=1)

    monkeypatch.setattr(backfill, "_worker_encoder", lambda *args: encode)
    report = _transform_fragment(
        _FragmentTask(
            uri=str(uri),
            storage_options=None,
            branch="candidate",
            source_version=candidate.version,
            fragment_id=candidate.get_fragments()[0].metadata.id,
            embedding="clap",
            checkpoint="resolved-checkpoint",
            sample_rate=44_100,
            batch_size=1,
            artifact="resolved-artifact",
            run_id="1" * 32,
        )
    )
    config = EmbeddingBackfillConfig(
        lance_uri=str(uri),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
        build_index=False,
    )
    add_config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("clap",),
        checkpoints={"clap": "resolved-checkpoint"},
        device="cuda",
        batch_size=1,
        build_index=False,
    )
    base_context = _BackfillContext(
        dataset=candidate,
        storage_options=None,
        spec=EMBEDDING_REGISTRY["clap"],
        add_config=add_config,
        checkpoint="resolved-checkpoint",
        artifact="resolved-artifact",
        source_version=candidate.version,
        sample_rate=44_100,
        total_rows=2,
        identity=_RunIdentity(run_id="0" * 32, git_commit="0" * 40),
    )
    shutdown: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: shutdown.append(True))
    monkeypatch.setattr(backfill, "_resolve_source_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(
        backfill,
        "_prepare_context",
        lambda selected, identity: replace(base_context, identity=identity),
    )
    monkeypatch.setattr(
        backfill,
        "_prepare_dispatch",
        lambda selected, context: _DispatchState(
            pending=[],
            reports={report.fragment_id: report},
            fragment_ids={report.fragment_id},
            total_rows=2,
            store=_ReportStore(local_dir=tmp_path / "unused", remote_uri=None),
            run_id=context.identity.run_id,
        ),
    )

    result = backfill_embedding(config)

    assert result.already_complete is False
    assert result.current_rows == 0
    assert result.resumed_rows == 2
    assert shutdown == [True]
    assert lance.dataset(uri).checkout_version(("candidate", None)).schema.names == [
        "audio",
        "clap",
    ]


def test_backfill_already_complete_flow_reports_explicit_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preexisting output must bypass dispatch while retaining resolved audit fields.

    :param tmp_path: Holds the complete branch-local Lance dataset.
    :param monkeypatch: Replaces GPU/Ray process boundaries while retaining coordinator logic.
    """
    import ray
    import torch

    import synth_setter.pipeline.data.backfill_embeddings as backfill

    uri = tmp_path / "complete.lance"
    root = lance.write_dataset(pa.table({"row_id": [0, 1]}), uri)
    candidate = root.create_branch("candidate", (None, root.version))
    field = pa.field("clap", pa.list_(pa.float32(), 512)).with_metadata(
        {
            b"synth_setter.embedding.name": b"clap",
            b"synth_setter.embedding.artifact": b"resolved-artifact",
        }
    )
    candidate.add_columns(
        pa.Table.from_arrays(
            [
                pa.array(
                    [[1.0] * 512, [2.0] * 512],
                    type=pa.list_(pa.float32(), 512),
                )
            ],
            schema=pa.schema([field]),
        )
    )
    config = EmbeddingBackfillConfig(
        lance_uri=str(uri),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
        build_index=False,
    )
    add_config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("clap",),
        checkpoints={"clap": "resolved-checkpoint"},
        device="cuda",
        batch_size=1,
        build_index=False,
    )
    base_context = _BackfillContext(
        dataset=candidate,
        storage_options=None,
        spec=EMBEDDING_REGISTRY["clap"],
        add_config=add_config,
        checkpoint="resolved-checkpoint",
        artifact="resolved-artifact",
        source_version=candidate.version,
        sample_rate=44_100,
        total_rows=2,
        identity=_RunIdentity(run_id="0" * 32, git_commit="0" * 40),
    )
    shutdown: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: shutdown.append(True))
    monkeypatch.setattr(backfill, "_resolve_source_git_sha", lambda: "b" * 40)
    monkeypatch.setattr(
        backfill,
        "_prepare_context",
        lambda selected, identity: replace(base_context, identity=identity),
    )

    result = backfill_embedding(config)

    assert result.already_complete is True
    assert result.current_fragments == 0
    assert result.resumed_fragments == 0
    assert result.checkpoint == "resolved-checkpoint"
    assert result.artifact == "resolved-artifact"
    assert result.git_commit == "b" * 40
    assert shutdown == [True]


def test_backfill_exception_always_shuts_down_ray(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinator failures after Ray startup must always release the runtime.

    :param tmp_path: Supplies a valid strict configuration path.
    :param monkeypatch: Injects the coordinator failure and records Ray shutdown.
    """
    import ray
    import torch

    import synth_setter.pipeline.data.backfill_embeddings as backfill

    def fail_run(
        config: EmbeddingBackfillConfig,
        identity: _RunIdentity,
        started: float,
    ) -> Never:
        del config, identity, started
        raise RuntimeError("boom")

    shutdown: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: shutdown.append(True))
    monkeypatch.setattr(backfill, "_resolve_source_git_sha", lambda: "c" * 40)
    monkeypatch.setattr(backfill, "_run_backfill", fail_run)
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=1,
        tasks_per_worker=1,
        gpu_per_worker=1.0,
    )

    with pytest.raises(RuntimeError, match="boom"):
        backfill_embedding(config)
    assert shutdown == [True]
