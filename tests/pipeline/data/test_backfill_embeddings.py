"""Behavioral coverage for branch-safe distributed embedding backfills."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.pipeline.data.backfill_embeddings import (
    EmbeddingBackfillConfig,
    EmbeddingPromotionConfig,
    _CacheIdentity,
    _DispatchState,
    _FragmentReport,
    _FragmentTask,
    _load_reports,
    _persist_report,
    _poll_dispatch,
    _report_store,
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
            )
        ),
        strict=True,
    )

    assert candidate.version == 1
    operation = lance.LanceOperation.Merge(
        [lance.FragmentMetadata.from_json(report.metadata_json)],
        pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc))),
    )
    committed = lance.LanceDataset.commit(candidate, operation, read_version=candidate.version)
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

    audio = np.linspace(-1, 1, 2 * 4_410, dtype=np.float32).reshape(2, 1, 4_410)
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
        )
    )
    operation = lance.LanceOperation.Merge(
        [lance.FragmentMetadata.from_json(report.metadata_json)],
        pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc))),
    )
    committed = lance.LanceDataset.commit(candidate, operation, read_version=candidate.version)

    assert report.rows == 2
    assert committed.schema.names == ["audio", "meanaudio_16k", "meanaudio_16k_vec"]
    for name in ("meanaudio_16k", "meanaudio_16k_vec"):
        assert committed.schema.field(name).metadata == {
            b"synth_setter.embedding.name": b"meanaudio_16k",
            b"synth_setter.embedding.artifact": b"meanaudio:test-artifact",
        }
    sequence = committed.take([0, 1], columns=["meanaudio_16k"]).column(0).combine_chunks()
    vector = committed.take([0, 1], columns=["meanaudio_16k_vec"]).column(0).combine_chunks()
    assert sequence.type.shape[0] == 20
    assert vector.type.list_size == 20
    np.testing.assert_allclose(
        vector.values.to_numpy().reshape(2, 20),
        np.repeat(audio.mean(axis=(1, 2))[:, None], 20, axis=1),
    )


def test_completed_fragment_report_survives_driver_restart(tmp_path: Path) -> None:
    """Durable reports must let a retry omit completed fragment work.

    :param tmp_path: Temporary directory for isolated reconciliation staging.
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
    )
    store = _report_store(config, identity)
    assert _load_reports(store, identity, {7}) == {}
    report = _FragmentReport(
        fragment_id=7,
        metadata_json="{}",
        schema_ipc="schema",
        pid=1,
        rows=8,
        elapsed_seconds=1.0,
        peak_rss_bytes=1,
        peak_gpu_allocated_bytes=2,
        peak_gpu_reserved_bytes=3,
    )
    _persist_report(store, report)

    assert _load_reports(store, identity, {7}) == {7: report}


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
            ),
        ),
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
    for name in ("clap", "meanaudio_16k"):
        assert main.schema.field(name) == candidate.schema.field(name)
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
    candidate.add_columns(pa.table({"clap": [[1.0], [2.0]], "experimental": [3, 4]}))

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
    candidate.add_columns(pa.table({"clap": [[1.0], [2.0]]}))

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
    candidate.add_columns(pa.table({"clap": [[1.0], [2.0]]}))
    dataset.add_columns(pa.table({"clap": [[9.0], [9.0]]}))

    with pytest.raises(ValueError, match="does not contain the validated candidate merge"):
        promote_embedding_candidate(
            EmbeddingPromotionConfig(
                lance_uri=str(uri),
                candidate_branch="candidate",
                rollback_tag="rollback",
                columns=("clap",),
            )
        )


def test_promote_cli_publishes_candidate_and_prints_json(tmp_path: Path) -> None:
    """The real CLI must dispatch promotion and emit its machine-readable result.

    :param tmp_path: Temporary directory for the real Lance dataset.
    """
    uri = tmp_path / "cli.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(pa.table({"clap": [[1.0], [2.0]]}))

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "synth_setter.pipeline.data.backfill_embeddings",
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
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["already_complete"] is False
    assert lance.dataset(uri).schema.names == ["row_id", "clap"]
