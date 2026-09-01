"""Behavioral coverage for branch-safe distributed embedding backfills."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError

from synth_setter.pipeline.data.backfill_embeddings import (
    EmbeddingBackfillConfig,
    EmbeddingPromotionConfig,
    _ActorConfig,
    _BackfillContext,
    _CacheIdentity,
    _FragmentReport,
    _load_reports,
    _persist_report,
    _report_store,
    _ReportStore,
    _run_actor_pool,
    _transform_fragment,
    backfill_embedding,
    promote_embedding_candidate,
)
from synth_setter.pipeline.data.lance_shard import tensor_array


class _TestEmbeddingActor:
    """Transform real fragments with a deterministic lightweight encoder."""

    def __init__(self, config_value: object) -> None:
        """Configure a deterministic actor.

        :param config_value: Serialized actor configuration.
        """
        self.config = _ActorConfig.model_validate(config_value, strict=True)
        self.store = _ReportStore(
            local_dir=Path(self.config.resume_dir),
            remote_uri=self.config.remote_report_uri,
        )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
            del sample_rate
            return np.repeat(audio.mean(axis=1, keepdims=True), 512, axis=1)

        report = _transform_fragment(self.config, encode, int(batch["fragment_id"][0]))
        _persist_report(self.store, report)
        return {"report_json": np.array([report.model_dump_json()])}


class _RecordingActor:
    """Record actor construction and fragment execution in a shared directory."""

    def __init__(self, config_value: object) -> None:
        """Configure actor-lifetime markers.

        :param config_value: Serialized actor configuration.
        """
        config = _ActorConfig.model_validate(config_value, strict=True)
        self.marker_dir = Path(config.resume_dir)
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        (self.marker_dir / f"actor-{os.getpid()}").touch()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        fragment_id = int(batch["fragment_id"][0])
        (self.marker_dir / f"fragment-{fragment_id}-pid-{os.getpid()}").touch()
        report = _FragmentReport(
            fragment_id=fragment_id,
            metadata_json="{}",
            schema_ipc="schema",
            rows=1,
            elapsed_seconds=0.01,
        )
        return {"report_json": np.array([report.model_dump_json()])}


def test_backfill_config_rejects_process_recycling_option() -> None:
    """Process recycling must not remain part of the public configuration."""
    with pytest.raises(ValidationError, match="tasks_per_worker"):
        EmbeddingBackfillConfig.model_validate(
            {
                "lance_uri": "dataset.lance",
                "branch": "candidate",
                "embedding": "clap",
                "workers": 2,
                "batch_size": 8,
                "gpu_per_worker": 0.5,
                "tasks_per_worker": 4,
            },
            strict=True,
        )


def test_actor_pool_reuses_fixed_persistent_actors(tmp_path: Path) -> None:
    import ray

    ray.init(num_cpus=2, num_gpus=1, include_dashboard=False, log_to_driver=False)
    try:
        config = _ActorConfig(
            lance_uri=str(tmp_path / "source.lance"),
            storage_options=None,
            branch="candidate",
            source_version=1,
            embedding="clap",
            checkpoint="checkpoint",
            sample_rate=44_100,
            batch_size=8,
            artifact="artifact",
            resume_dir=str(tmp_path / "markers"),
            remote_report_uri=None,
        )
        reports = _run_actor_pool(
            fragment_ids=tuple(range(8)),
            actor_config=config,
            workers=2,
            gpu_per_worker=0.5,
            actor_type=_RecordingActor,  # type: ignore[arg-type]
        )
    finally:
        ray.shutdown()

    actor_markers = list((tmp_path / "markers").glob("actor-*"))
    fragment_markers = list((tmp_path / "markers").glob("fragment-*"))
    assert len(actor_markers) == 2
    assert len(fragment_markers) == 8
    assert {report.fragment_id for report in reports} == set(range(8))


def test_transform_fragment_writes_only_uncommitted_fragment_data(tmp_path: Path) -> None:
    audio = np.arange(64, dtype=np.float32).reshape(8, 2, 4) / 64
    uri = tmp_path / "worker.lance"
    dataset = lance.write_dataset(
        pa.table({"audio": tensor_array(audio, np.dtype("float32"), (2, 4))}),
        uri,
        max_rows_per_file=8,
    )
    candidate = dataset.create_branch("candidate", (None, dataset.version))

    def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.repeat(audio.mean(axis=1, keepdims=True), 512, axis=1)

    config = _ActorConfig(
        lance_uri=str(uri),
        storage_options=None,
        branch="candidate",
        source_version=candidate.version,
        embedding="clap",
        checkpoint="unused",
        sample_rate=44_100,
        batch_size=2,
        artifact="clap:test-artifact",
        resume_dir=str(tmp_path / "reports"),
        remote_report_uri=None,
    )
    fragment_id = candidate.get_fragments()[0].metadata.id

    report = _transform_fragment(config, encode, fragment_id)

    assert candidate.version == 1
    operation = lance.LanceOperation.Merge(
        [lance.FragmentMetadata.from_json(report.metadata_json)],
        pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc))),
    )
    committed = lance.LanceDataset.commit(candidate, operation, read_version=candidate.version)
    assert report.rows == 8
    assert committed.version == 2
    expected = np.repeat(audio.mean(axis=(1, 2))[:, None], 512, axis=1)
    actual = committed.take(range(8), columns=["clap"]).column(0).combine_chunks()
    np.testing.assert_allclose(actual.values.to_numpy().reshape(8, 512), expected)


def test_backfill_embedding_publishes_all_fragments_in_one_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The driver must publish complete actor output through one candidate commit.

    :param tmp_path: Temporary directory for the real Lance dataset.
    :param monkeypatch: Fixture replacing model policy resolution, not Lance or Ray.
    """
    import synth_setter.pipeline.data.backfill_embeddings as backfill
    from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

    audio = np.arange(96, dtype=np.float32).reshape(12, 2, 4) / 96
    uri = tmp_path / "backfill.lance"
    source = lance.write_dataset(
        pa.table({"audio": tensor_array(audio, np.dtype("float32"), (2, 4))}),
        uri,
        max_rows_per_file=4,
    )
    candidate = source.create_branch("candidate", (None, source.version))
    add_config = AddEmbeddingsConfig(
        lance_uri=str(uri),
        embeddings=("clap",),
        checkpoints={"clap": "unused"},
        device="cuda",
        build_index=False,
    )
    context = _BackfillContext(
        dataset=candidate,
        storage_options=None,
        spec=EMBEDDING_REGISTRY["clap"],
        add_config=add_config,
        checkpoint="unused",
        artifact="clap:test-artifact",
        source_version=candidate.version,
        sample_rate=44_100,
        total_rows=12,
    )
    monkeypatch.setattr(backfill, "_prepare_context", lambda config: context)
    monkeypatch.setattr(backfill, "_EmbeddingActor", _TestEmbeddingActor)
    config = EmbeddingBackfillConfig(
        lance_uri=str(uri),
        branch="candidate",
        embedding="clap",
        workers=2,
        batch_size=2,
        gpu_per_worker=0.5,
        build_index=False,
        resume_dir=tmp_path / "reports",
    )

    result = backfill_embedding(config)

    committed = lance.dataset(uri).checkout_version(("candidate", None))
    assert result.computed_fragments == 3
    assert result.resumed_fragments == 0
    assert result.data_version == 2
    assert committed.version == 2
    assert committed.schema.names == ["audio", "clap"]
    assert source.version == 1


def test_completed_fragment_report_survives_driver_restart(tmp_path: Path) -> None:
    config = EmbeddingBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        branch="candidate",
        embedding="clap",
        workers=1,
        batch_size=2,
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
        implementation_digest="abc123",
    )
    store = _report_store(config, identity)
    assert _load_reports(store, identity, {7}) == {}
    report = _FragmentReport(
        fragment_id=7,
        metadata_json="{}",
        schema_ipc="schema",
        rows=8,
        elapsed_seconds=1.0,
    )

    _persist_report(store, report)

    assert _load_reports(store, identity, {7}) == {7: report}


def test_promote_embedding_candidate_preserves_source_and_is_idempotent(
    tmp_path: Path,
) -> None:
    uri = tmp_path / "embeddings.lance"
    source = pa.table(
        {
            "row_id": np.arange(8, dtype=np.int64),
            "audio": pa.array(np.arange(32, dtype=np.float32).reshape(8, 4).tolist()),
        }
    ).replace_schema_metadata({b"source-contract": b"preserve-me"})
    dataset = lance.write_dataset(source, uri, max_rows_per_file=4)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(pa.table({"clap": np.arange(16).reshape(8, 2).tolist()}))

    result = promote_embedding_candidate(
        EmbeddingPromotionConfig(
            lance_uri=str(uri),
            candidate_branch="candidate",
            rollback_tag="rollback",
            columns=("clap",),
        )
    )

    main = lance.dataset(uri)
    assert result.already_complete is False
    assert main.schema.metadata == source.schema.metadata
    assert main.take(range(8), columns=["row_id", "audio"]).equals(source)
    assert main.to_table(columns=["clap"]).equals(candidate.to_table(columns=["clap"]))

    retry = promote_embedding_candidate(
        EmbeddingPromotionConfig(
            lance_uri=str(uri),
            candidate_branch="candidate",
            rollback_tag="rollback",
            columns=("clap",),
        )
    )
    assert retry.already_complete is True
    assert lance.dataset(uri).version == result.committed_version


def test_promote_embedding_candidate_rejects_unselected_column(tmp_path: Path) -> None:
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


def test_promote_cli_publishes_candidate_and_prints_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real CLI must publish a candidate and emit machine-readable JSON.

    :param tmp_path: Temporary directory for the real Lance dataset.
    :param monkeypatch: Fixture installing the CLI arguments.
    :param capsys: Fixture capturing the emitted result.
    """
    from synth_setter.pipeline.data.backfill_embeddings import main

    uri = tmp_path / "cli.lance"
    dataset = lance.write_dataset(pa.table({"row_id": [1, 2]}), uri)
    dataset.tags.create("rollback", (None, dataset.version))
    candidate = dataset.create_branch("candidate", (None, dataset.version))
    candidate.add_columns(pa.table({"clap": [[1.0], [2.0]]}))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill-embeddings",
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


def test_promote_embedding_candidate_rejects_modified_source_values(
    tmp_path: Path,
) -> None:
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


def test_promote_embedding_candidate_rejects_independent_main_write(
    tmp_path: Path,
) -> None:
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
