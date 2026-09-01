"""Backfill embedding columns with persistent Ray Data actors.

Workers create uncommitted Lance fragment data. The driver validates complete fragment coverage and
publishes one merge commit on the candidate branch.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol, cast
from urllib.parse import unquote, urlparse

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    import lance
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import EmbeddingSpec
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

logger = structlog.get_logger(__name__)
_PROGRESS_INTERVAL_SECONDS = 120.0


class _AudioEncoder(Protocol):
    """Encode float audio shaped by the selected embedding policy."""

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return one model-specific embedding batch.

        :param audio: Batched float audio.
        :param sample_rate: Audio sample rate in hertz.
        :return: Batched embeddings.
        """
        ...


class _FragmentReport(BaseModel):
    """Validate fragment metadata crossing Ray and persistence boundaries.

    .. attribute :: model_config

        Pydantic model configuration.
    .. attribute :: fragment_id

        Source fragment identifier.
    .. attribute :: metadata_json

        Serialized output fragment metadata.
    .. attribute :: schema_ipc

        Base64-encoded Arrow schema.
    .. attribute :: rows

        Output row count.
    .. attribute :: elapsed_seconds

        Worker elapsed time in seconds.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    fragment_id: int = Field(ge=0)
    metadata_json: str
    schema_ipc: str
    rows: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)


class _CacheIdentity(BaseModel):
    """Bind durable reports to one immutable implementation and operation.

    .. attribute :: model_config

        Pydantic model configuration.
    .. attribute :: lance_uri

        Dataset URI.
    .. attribute :: branch

        Candidate branch.
    .. attribute :: source_version

        Immutable source version.
    .. attribute :: embedding

        Embedding policy name.
    .. attribute :: checkpoint

        Model checkpoint.
    .. attribute :: sample_rate

        Audio sample rate in hertz.
    .. attribute :: batch_size

        Model batch size.
    .. attribute :: artifact

        Expected embedding provenance.
    .. attribute :: implementation_digest

        Worker implementation digest.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str
    branch: str
    source_version: int
    embedding: Literal["clap", "meanaudio_16k"]
    checkpoint: str
    sample_rate: int
    batch_size: int
    artifact: str
    implementation_digest: str


class EmbeddingBackfillConfig(BaseModel):
    """Configure one branch-scoped distributed embedding write.

    .. attribute :: model_config

        Pydantic model configuration.
    .. attribute :: lance_uri

        Dataset URI.
    .. attribute :: branch

        Candidate branch.
    .. attribute :: embedding

        Embedding policy name.
    .. attribute :: workers

        Fixed actor count.
    .. attribute :: batch_size

        Model batch size.
    .. attribute :: gpu_per_worker

        Fractional GPU reservation per actor.
    .. attribute :: checkpoint

        Optional model checkpoint override.
    .. attribute :: build_index

        Whether to build the embedding index.
    .. attribute :: num_partitions

        Optional index partition count.
    .. attribute :: resume_dir

        Optional local report directory.
    .. attribute :: result

        Optional result JSON path.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(min_length=1)
    branch: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    embedding: Literal["clap", "meanaudio_16k"]
    workers: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    gpu_per_worker: float = Field(gt=0, le=1)
    checkpoint: str | None = None
    build_index: bool = True
    num_partitions: int | None = Field(default=None, ge=1)
    resume_dir: Path | None = None
    result: Path | None = None

    @field_validator("branch")
    @classmethod
    def _reject_main(cls, value: str) -> str:
        if value == "main":
            raise ValueError("distributed writes require a candidate branch")
        return value

    @model_validator(mode="after")
    def _fit_single_gpu(self) -> EmbeddingBackfillConfig:
        if self.workers * self.gpu_per_worker > 1:
            raise ValueError("workers * gpu_per_worker must not exceed one GPU")
        return self


class EmbeddingPromotionConfig(BaseModel):
    """Configure publication of validated candidate fields on main.

    .. attribute :: model_config

        Pydantic model configuration.
    .. attribute :: lance_uri

        Dataset URI.
    .. attribute :: candidate_branch

        Candidate branch.
    .. attribute :: rollback_tag

        Tag identifying the source main version.
    .. attribute :: columns

        Candidate columns to publish.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(min_length=1)
    candidate_branch: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    rollback_tag: str = Field(min_length=1)
    columns: tuple[str, ...]

    @field_validator("columns")
    @classmethod
    def _require_unique_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("columns must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("columns must not contain duplicates")
        return value


class _ActorConfig(BaseModel):
    """Carry immutable snapshot and report-store state to each Ray actor.

    .. attribute :: model_config

        Pydantic model configuration.
    .. attribute :: lance_uri

        Dataset URI.
    .. attribute :: storage_options

        Optional Lance storage options.
    .. attribute :: branch

        Candidate branch.
    .. attribute :: source_version

        Immutable source version.
    .. attribute :: embedding

        Embedding policy name.
    .. attribute :: checkpoint

        Model checkpoint.
    .. attribute :: sample_rate

        Audio sample rate in hertz.
    .. attribute :: batch_size

        Model batch size.
    .. attribute :: artifact

        Expected embedding provenance.
    .. attribute :: resume_dir

        Actor-local report directory.
    .. attribute :: remote_report_uri

        Optional shared report URI.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str
    storage_options: dict[str, str] | None
    branch: str
    source_version: int
    embedding: Literal["clap", "meanaudio_16k"]
    checkpoint: str
    sample_rate: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    artifact: str
    resume_dir: str
    remote_report_uri: str | None


@dataclass(frozen=True)
class EmbeddingBackfillResult:
    """Describe the artifact and versions produced by one backfill run.

    .. attribute :: run_id

        Unique run identifier.
    .. attribute :: git_commit

        Source revision.
    .. attribute :: branch

        Candidate branch.
    .. attribute :: embedding

        Embedding policy name.
    .. attribute :: checkpoint

        Model checkpoint.
    .. attribute :: artifact

        Embedding provenance.
    .. attribute :: workers

        Actor count.
    .. attribute :: gpu_per_worker

        GPU reservation per actor.
    .. attribute :: batch_size

        Model batch size.
    .. attribute :: rows

        Dataset row count.
    .. attribute :: fragments

        Dataset fragment count.
    .. attribute :: resumed_fragments

        Reports reused from durable state.
    .. attribute :: computed_fragments

        Fragments computed in this run.
    .. attribute :: source_version

        Immutable source version.
    .. attribute :: data_version

        Merge commit version.
    .. attribute :: final_version

        Version after optional index creation.
    .. attribute :: elapsed_seconds

        Total elapsed time in seconds.
    .. attribute :: rows_per_second

        End-to-end throughput.
    .. attribute :: already_complete

        Whether no embedding merge was required.
    .. attribute :: index_built

        Whether the requested index exists.
    """

    run_id: str
    git_commit: str
    branch: str
    embedding: str
    checkpoint: str
    artifact: str
    workers: int
    gpu_per_worker: float
    batch_size: int
    rows: int
    fragments: int
    resumed_fragments: int
    computed_fragments: int
    source_version: int
    data_version: int
    final_version: int
    elapsed_seconds: float
    rows_per_second: float
    already_complete: bool
    index_built: bool


@dataclass(frozen=True)
class EmbeddingPromotionResult:
    """Describe one candidate-to-main publication.

    .. attribute :: source_version

        Rollback source version.
    .. attribute :: candidate_version

        Validated candidate version.
    .. attribute :: committed_version

        Published main version.
    .. attribute :: already_complete

        Whether the candidate was already published.
    """

    source_version: int
    candidate_version: int
    committed_version: int
    already_complete: bool


@dataclass(frozen=True)
class _ReportStore:
    """Locate local staging and optional shared reconciliation reports.

    .. attribute :: local_dir

        Actor-local report directory.
    .. attribute :: remote_uri

        Optional shared report URI.
    """

    local_dir: Path
    remote_uri: str | None


@dataclass(frozen=True)
class _BackfillContext:
    """Hold the immutable source snapshot and resolved embedding policy.

    .. attribute :: dataset

        Checked-out candidate snapshot.
    .. attribute :: storage_options

        Optional Lance storage options.
    .. attribute :: spec

        Resolved embedding policy.
    .. attribute :: add_config

        Shared embedding configuration.
    .. attribute :: checkpoint

        Resolved model checkpoint.
    .. attribute :: artifact

        Expected embedding provenance.
    .. attribute :: source_version

        Immutable source version.
    .. attribute :: sample_rate

        Audio sample rate in hertz.
    .. attribute :: total_rows

        Source row count.
    """

    dataset: lance.LanceDataset
    storage_options: dict[str, str] | None
    spec: EmbeddingSpec
    add_config: AddEmbeddingsConfig
    checkpoint: str
    artifact: str
    source_version: int
    sample_rate: int
    total_rows: int


def _implementation_digest() -> str:
    """Hash the worker implementation that produces resumable fragment data.

    :return: Content digest for report compatibility checks.
    """
    from synth_setter.pipeline.data import add_embeddings

    digest = hashlib.sha256()
    for path in (Path(__file__), Path(add_embeddings.__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _storage_options(uri: str) -> dict[str, str] | None:
    if not uri.startswith("s3://"):
        return None
    from synth_setter.pipeline.r2_io import r2_storage_options

    return r2_storage_options()


def _validate_resume_directory(config: EmbeddingBackfillConfig, path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve(), Path.cwd().resolve()}:
        raise ValueError(f"resume directory {resolved} is not a dedicated path")
    dataset_path = None
    if config.lance_uri.startswith("file://"):
        dataset_path = Path(unquote(urlparse(config.lance_uri).path)).resolve()
    elif "://" not in config.lance_uri:
        dataset_path = Path(config.lance_uri).expanduser().resolve()
    if dataset_path is not None and (
        resolved == dataset_path
        or resolved in dataset_path.parents
        or dataset_path in resolved.parents
    ):
        raise ValueError(f"resume directory {resolved} overlaps dataset {dataset_path}")
    return resolved


def _report_store(config: EmbeddingBackfillConfig, identity: _CacheIdentity) -> _ReportStore:
    digest = hashlib.sha256(identity.model_dump_json().encode()).hexdigest()[:20]
    default = Path.home() / ".cache" / "synth-setter" / "embedding-backfill" / digest
    local_dir = _validate_resume_directory(config, config.resume_dir or default)
    remote_uri = None
    if config.lance_uri.startswith("s3://"):
        from synth_setter.pipeline.r2_io import from_s3_uri

        root = from_s3_uri(config.lance_uri.rstrip("/"))
        remote_uri = f"{root}/metadata/workers/embedding-backfill/{digest}"
    return _ReportStore(local_dir=local_dir, remote_uri=remote_uri)


def _hydrate_remote_reports(store: _ReportStore) -> None:
    if store.remote_uri is None:
        return
    from synth_setter.pipeline.r2_io import (
        download_to_path,
        list_entries,
        r2_directory_exists,
    )

    if not r2_directory_exists(store.remote_uri):
        return
    store.local_dir.mkdir(parents=True, exist_ok=True)
    for entry in list_entries(store.remote_uri):
        download_to_path(
            f"{store.remote_uri}/{entry.path}", store.local_dir / Path(entry.path).name
        )


def _establish_identity(store: _ReportStore, identity: _CacheIdentity) -> None:
    from synth_setter.pipeline.r2_io import upload_to_uri

    path = store.local_dir / "identity.json"
    payload = identity.model_dump_json(indent=2) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        cached = _CacheIdentity.model_validate_json(path.read_text(), strict=True)
        if cached != identity:
            raise ValueError(f"resume directory {store.local_dir} has another identity")
        return
    with os.fdopen(descriptor, "w") as stream:
        stream.write(payload)
    if store.remote_uri is not None:
        upload_to_uri(path, f"{store.remote_uri}/identity.json")


def _load_reports(
    store: _ReportStore,
    identity: _CacheIdentity,
    fragment_ids: set[int],
) -> dict[int, _FragmentReport]:
    store.local_dir.mkdir(parents=True, exist_ok=True)
    _hydrate_remote_reports(store)
    _establish_identity(store, identity)
    reports: dict[int, _FragmentReport] = {}
    for path in sorted(store.local_dir.glob("fragment-*.json")):
        report = _FragmentReport.model_validate_json(path.read_text(), strict=True)
        if report.fragment_id in fragment_ids:
            reports.setdefault(report.fragment_id, report)
    return reports


def _persist_report(store: _ReportStore, report: _FragmentReport) -> None:
    from synth_setter.pipeline.r2_io import upload_to_uri

    destination = store.local_dir / f"fragment-{report.fragment_id}-{uuid.uuid4().hex}.json"
    temporary = store.local_dir / f".{destination.name}.tmp"
    temporary.write_text(report.model_dump_json() + "\n")
    temporary.replace(destination)
    if store.remote_uri is not None:
        upload_to_uri(destination, f"{store.remote_uri}/{destination.name}")


def _load_encoder(config: _ActorConfig) -> _AudioEncoder:
    from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

    add_config = AddEmbeddingsConfig(
        lance_uri=config.lance_uri,
        embeddings=(config.embedding,),
        checkpoints={config.embedding: config.checkpoint},
        device="cuda",
        build_index=False,
    )
    encoder = EMBEDDING_REGISTRY[config.embedding].load_encoder(config.checkpoint, add_config)
    return cast("_AudioEncoder", encoder)


def _transform_fragment(
    config: _ActorConfig,
    encoder: _AudioEncoder,
    fragment_id: int,
) -> _FragmentReport:
    import lance
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import (
        EMBEDDING_REGISTRY,
        _decoded_sources,
        _encode_columns,
        _output_columns,
    )

    started = time.monotonic()
    dataset = lance.dataset(
        config.lance_uri, storage_options=config.storage_options
    ).checkout_version((config.branch, config.source_version))
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        raise ValueError(f"missing fragment {fragment_id}")
    spec = EMBEDDING_REGISTRY[config.embedding]

    def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
        encoded = _encode_columns(
            _decoded_sources(batch, spec.input_fields),
            config.sample_rate,
            [spec],
            [encoder],
        )
        metadata = {
            b"synth_setter.embedding.name": config.embedding.encode(),
            b"synth_setter.embedding.artifact": config.artifact.encode(),
        }
        columns = _output_columns(spec)
        schema = pa.schema(
            [encoded.schema.field(column).with_metadata(metadata) for column in columns]
        )
        return pa.RecordBatch.from_arrays(
            [encoded.column(column) for column in columns], schema=schema
        )

    metadata, schema = fragment.merge_columns(
        transform,
        list(spec.input_fields),
        batch_size=config.batch_size,
    )
    return _FragmentReport(
        fragment_id=fragment_id,
        metadata_json=json.dumps(metadata.to_json()),
        schema_ipc=base64.b64encode(schema.to_pyarrow().serialize().to_pybytes()).decode(),
        rows=fragment.count_rows(),
        elapsed_seconds=time.monotonic() - started,
    )


class _EmbeddingActor:
    """Load one encoder and transform fragment IDs for the actor lifetime."""

    def __init__(self, config_value: object) -> None:
        """Load the actor policy and encoder.

        :param config_value: Serialized actor configuration.
        """
        self.config = _ActorConfig.model_validate(config_value, strict=True)
        self.encoder = _load_encoder(self.config)
        self.store = _ReportStore(
            local_dir=Path(self.config.resume_dir),
            remote_uri=self.config.remote_report_uri,
        )
        self.store.local_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if len(batch["fragment_id"]) != 1:
            raise ValueError("embedding actors require one fragment per batch")
        report = _transform_fragment(
            self.config,
            self.encoder,
            int(batch["fragment_id"][0]),
        )
        _persist_report(self.store, report)
        return {"report_json": np.array([report.model_dump_json()])}


def _run_actor_pool(
    *,
    fragment_ids: Sequence[int],
    actor_config: _ActorConfig,
    workers: int,
    gpu_per_worker: float,
    actor_type: type[_EmbeddingActor] | None = None,
) -> list[_FragmentReport]:
    if not fragment_ids:
        return []
    import ray

    actor_type = actor_type or _EmbeddingActor
    dataset = ray.data.from_items(
        [{"fragment_id": fragment_id} for fragment_id in fragment_ids],
        override_num_blocks=len(fragment_ids),
    ).map_batches(
        actor_type,
        batch_size=1,
        batch_format="numpy",
        compute=ray.data.ActorPoolStrategy(size=workers),
        fn_constructor_args=(actor_config.model_dump(),),
        num_cpus=1,
        num_gpus=gpu_per_worker,
    )
    reports: list[_FragmentReport] = []
    last_progress = time.monotonic()
    for row in dataset.iter_rows():
        reports.append(_FragmentReport.model_validate_json(str(row["report_json"]), strict=True))
        now = time.monotonic()
        if now - last_progress >= _PROGRESS_INTERVAL_SECONDS or len(reports) == len(fragment_ids):
            logger.info(
                "embedding_backfill_progress",
                embedding=actor_config.embedding,
                branch=actor_config.branch,
                completed_fragments=len(reports),
                total_fragments=len(fragment_ids),
            )
            last_progress = now
    return reports


def _embedding_is_complete(
    dataset: lance.LanceDataset,
    spec: EmbeddingSpec,
    artifact: bytes,
) -> bool:
    from synth_setter.pipeline.data.add_embeddings import _output_columns

    columns = _output_columns(spec)
    present = set(columns) & set(dataset.schema.names)
    if present and present != set(columns):
        raise ValueError(f"dataset has partial {spec.name} columns: {sorted(present)}")
    if not present:
        return False
    expected = {
        b"synth_setter.embedding.name": spec.name.encode(),
        b"synth_setter.embedding.artifact": artifact,
    }
    if any((dataset.schema.field(column).metadata or {}) != expected for column in columns):
        raise ValueError(f"dataset {spec.name} artifact identity does not match")
    return True


def _ensure_index(
    dataset: lance.LanceDataset,
    spec: EmbeddingSpec,
    config: AddEmbeddingsConfig,
) -> tuple[lance.LanceDataset, bool]:
    from synth_setter.pipeline.data.add_embeddings import (
        _matching_index_exists,
        build_index,
    )

    if not config.build_index or spec.index is None:
        return dataset, False
    column = spec.index.vector_column or spec.column
    if _matching_index_exists(dataset, column, index=spec.index, config=config):
        return dataset, False
    built = build_index(dataset, column, index=spec.index, config=config)
    return dataset, built


def _prepare_context(config: EmbeddingBackfillConfig) -> _BackfillContext:
    import lance

    from synth_setter.pipeline.data.add_embeddings import (
        EMBEDDING_REGISTRY,
        _resolve_artifact_identity,
    )
    from synth_setter.pipeline.data.lance_shard import read_shard_metadata
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

    storage_options = _storage_options(config.lance_uri)
    dataset = lance.dataset(config.lance_uri, storage_options=storage_options).checkout_version(
        (config.branch, None)
    )
    spec = EMBEDDING_REGISTRY[config.embedding]
    checkpoint = config.checkpoint or spec.default_checkpoint
    add_config = AddEmbeddingsConfig(
        lance_uri=config.lance_uri,
        embeddings=(config.embedding,),
        checkpoints={config.embedding: checkpoint},
        device="cuda",
        batch_size=config.batch_size,
        build_index=config.build_index,
        num_partitions=config.num_partitions,
    )
    artifact = _resolve_artifact_identity(spec, add_config)
    total_rows = dataset.count_rows()
    if total_rows < 1:
        raise ValueError("cannot backfill an empty dataset")
    return _BackfillContext(
        dataset=dataset,
        storage_options=storage_options,
        spec=spec,
        add_config=add_config,
        checkpoint=checkpoint,
        artifact=artifact,
        source_version=dataset.version,
        sample_rate=int(read_shard_metadata(dataset.schema).sample_rate),
        total_rows=total_rows,
    )


def _decode_reports(
    reports: Sequence[_FragmentReport],
) -> tuple[list[lance.FragmentMetadata], pa.Schema]:
    import lance
    import pyarrow as pa

    metadata = [lance.FragmentMetadata.from_json(report.metadata_json) for report in reports]
    schemas = [
        pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc)))
        for report in reports
    ]
    if not schemas or any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("worker schemas differ")
    return metadata, schemas[0]


def _commit_reports(
    config: EmbeddingBackfillConfig,
    context: _BackfillContext,
    reports: Sequence[_FragmentReport],
) -> lance.LanceDataset:
    import lance

    fragments = context.dataset.get_fragments()
    expected_ids = {fragment.metadata.id for fragment in fragments}
    actual_ids = [report.fragment_id for report in reports]
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        raise ValueError("worker reports do not cover every source fragment exactly once")
    if sum(report.rows for report in reports) != context.total_rows:
        raise ValueError("worker report rows do not match the source row count")

    latest = lance.dataset(
        config.lance_uri, storage_options=context.storage_options
    ).checkout_version((config.branch, None))
    if latest.version != context.source_version:
        if _embedding_is_complete(latest, context.spec, context.artifact.encode()):
            return latest
        raise ValueError("candidate branch advanced incompatibly during backfill")
    metadata, schema = _decode_reports(reports)
    return lance.LanceDataset.commit(
        context.dataset,
        lance.LanceOperation.Merge(metadata, schema),
        read_version=context.source_version,
        storage_options=context.storage_options,
        commit_message=f"Add {config.embedding} embeddings",
    )


def _write_result(
    result: EmbeddingBackfillResult | EmbeddingPromotionResult,
    destination: Path | None,
) -> None:
    payload = json.dumps(asdict(result), sort_keys=True)
    sys.stdout.write(f"{payload}\n")
    if destination is not None:
        destination.write_text(f"{payload}\n")


def backfill_embedding(config: EmbeddingBackfillConfig) -> EmbeddingBackfillResult:
    """Compute missing fragments with persistent actors and publish once.

    :param config: Validated branch-scoped execution policy.
    :return: Backfill versions, counts, provenance, and throughput.
    """
    import ray

    from synth_setter.utils.logging_utils import resolve_git_sha

    started = time.monotonic()
    run_id = uuid.uuid4().hex
    git_commit = resolve_git_sha()
    context = _prepare_context(config)
    if _embedding_is_complete(context.dataset, context.spec, context.artifact.encode()):
        final, index_built = _ensure_index(context.dataset, context.spec, context.add_config)
        result = EmbeddingBackfillResult(
            run_id=run_id,
            git_commit=git_commit,
            branch=config.branch,
            embedding=config.embedding,
            checkpoint=context.checkpoint,
            artifact=context.artifact,
            workers=config.workers,
            gpu_per_worker=config.gpu_per_worker,
            batch_size=config.batch_size,
            rows=context.total_rows,
            fragments=len(final.get_fragments()),
            resumed_fragments=0,
            computed_fragments=0,
            source_version=context.source_version,
            data_version=context.source_version,
            final_version=final.version,
            elapsed_seconds=time.monotonic() - started,
            rows_per_second=0.0,
            already_complete=True,
            index_built=index_built,
        )
        _write_result(result, config.result)
        return result

    identity = _CacheIdentity(
        lance_uri=config.lance_uri,
        branch=config.branch,
        source_version=context.source_version,
        embedding=config.embedding,
        checkpoint=context.checkpoint,
        sample_rate=context.sample_rate,
        batch_size=config.batch_size,
        artifact=context.artifact,
        implementation_digest=_implementation_digest(),
    )
    store = _report_store(config, identity)
    fragment_ids = {fragment.metadata.id for fragment in context.dataset.get_fragments()}
    resumed = _load_reports(store, identity, fragment_ids)
    actor_config = _ActorConfig(
        lance_uri=config.lance_uri,
        storage_options=context.storage_options,
        branch=config.branch,
        source_version=context.source_version,
        embedding=config.embedding,
        checkpoint=context.checkpoint,
        sample_rate=context.sample_rate,
        batch_size=config.batch_size,
        artifact=context.artifact,
        resume_dir=str(store.local_dir),
        remote_report_uri=store.remote_uri,
    )

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(
            num_cpus=config.workers,
            num_gpus=1,
            include_dashboard=False,
            log_to_driver=False,
        )
    try:
        computed = _run_actor_pool(
            fragment_ids=tuple(sorted(fragment_ids - resumed.keys())),
            actor_config=actor_config,
            workers=config.workers,
            gpu_per_worker=config.gpu_per_worker,
        )
    finally:
        if started_ray:
            ray.shutdown()

    reports_by_id = resumed | {report.fragment_id: report for report in computed}
    reports = [reports_by_id[fragment_id] for fragment_id in sorted(fragment_ids)]
    committed = _commit_reports(config, context, reports)
    data_version = committed.version
    final, index_built = _ensure_index(committed, context.spec, context.add_config)
    elapsed = time.monotonic() - started
    result = EmbeddingBackfillResult(
        run_id=run_id,
        git_commit=git_commit,
        branch=config.branch,
        embedding=config.embedding,
        checkpoint=context.checkpoint,
        artifact=context.artifact,
        workers=config.workers,
        gpu_per_worker=config.gpu_per_worker,
        batch_size=config.batch_size,
        rows=context.total_rows,
        fragments=len(final.get_fragments()),
        resumed_fragments=len(resumed),
        computed_fragments=len(computed),
        source_version=context.source_version,
        data_version=data_version,
        final_version=final.version,
        elapsed_seconds=elapsed,
        rows_per_second=context.total_rows / elapsed,
        already_complete=False,
        index_built=index_built,
    )
    _write_result(result, config.result)
    return result


def _copy_candidate_data_directory(uri: str, branch: str) -> None:
    root = uri.rstrip("/")
    if uri.startswith("s3://"):
        from synth_setter.pipeline import r2_io

        source = r2_io.from_s3_uri(f"{root}/tree/{branch}/data")
        destination = r2_io.from_s3_uri(f"{root}/data")
        r2_io.copy_directory(source, destination)
        return
    local_root = Path(unquote(urlparse(uri).path)) if uri.startswith("file://") else Path(uri)
    shutil.copytree(
        local_root / "tree" / branch / "data",
        local_root / "data",
        dirs_exist_ok=True,
    )


def _candidate_merge_operation(
    candidate: lance.LanceDataset,
    source_version: int,
) -> lance.LanceOperation.Merge:
    import lance

    operation: lance.LanceOperation.Merge | None = None
    for version in range(candidate.version, source_version, -1):
        transaction = candidate.read_transaction(version)
        if transaction is not None and isinstance(
            transaction.operation, lance.LanceOperation.Merge
        ):
            operation = transaction.operation
            break
    if operation is None:
        raise ValueError("candidate has no merge transaction to promote")

    fragments = []
    for fragment in operation.fragments:
        payload = fragment.to_json()
        for data_file in payload["files"]:
            base_id = data_file["base_id"]
            if base_id == 0:
                data_file["base_id"] = None
            elif base_id is None:
                path = PurePosixPath(data_file["path"])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe candidate data path {str(path)!r}")
            else:
                raise ValueError(f"unexpected candidate data base id {base_id}")
        fragments.append(lance.FragmentMetadata.from_json(json.dumps(payload)))
    return lance.LanceOperation.Merge(fragments, candidate.schema)


def _main_contains_candidate_merge(
    main: lance.LanceDataset,
    candidate: lance.LanceDataset,
    source_version: int,
) -> bool:
    import lance

    if main.version <= source_version:
        return False
    expected = _candidate_merge_operation(candidate, source_version)
    transaction = main.read_transaction(source_version + 1)
    if transaction is None or not isinstance(transaction.operation, lance.LanceOperation.Merge):
        return False
    actual = transaction.operation
    return actual.schema == expected.schema and [
        fragment.to_json() for fragment in actual.fragments
    ] == [fragment.to_json() for fragment in expected.fragments]


def promote_embedding_candidate(
    config: EmbeddingPromotionConfig,
) -> EmbeddingPromotionResult:
    """Publish validated candidate fields through one main merge commit.

    :param config: Validated promotion policy.
    :return: Source, candidate, and committed main versions.
    :raises ValueError: Candidate provenance, schema, or source data is invalid.
    """
    import lance

    storage_options = _storage_options(config.lance_uri)
    root = lance.dataset(config.lance_uri, storage_options=storage_options)
    main = root.checkout_version((None, None))
    candidate = root.checkout_version((config.candidate_branch, None))
    rollback = root.tags.list().get(config.rollback_tag)
    if rollback is None or rollback["branch"] not in {None, "main"}:
        raise ValueError(f"rollback tag {config.rollback_tag!r} must identify main")
    source_version = rollback["version"]
    source = root.checkout_version((None, source_version))
    branch = root.branches.list().get(config.candidate_branch)
    if (
        branch is None
        or branch["parent_branch"] is not None
        or branch["parent_version"] != source_version
    ):
        raise ValueError("candidate must descend from the rollback version on main")

    missing = set(config.columns) - set(candidate.schema.names)
    if missing:
        raise ValueError(f"candidate lacks columns {sorted(missing)}")
    expected_names = set(source.schema.names) | set(config.columns)
    if set(candidate.schema.names) != expected_names:
        extras = sorted(set(candidate.schema.names) - expected_names)
        raise ValueError(f"candidate has unselected columns {extras}")
    if candidate.schema.metadata != source.schema.metadata or any(
        candidate.schema.field(name) != source.schema.field(name) for name in source.schema.names
    ):
        raise ValueError("candidate changed the rollback source schema")

    present = set(config.columns) & set(main.schema.names)
    if present:
        if present != set(config.columns):
            raise ValueError(f"main has partial candidate columns {sorted(present)}")
        if not _main_contains_candidate_merge(main, candidate, source_version):
            raise ValueError("main does not contain the validated candidate merge")
        return EmbeddingPromotionResult(
            source_version=source_version,
            candidate_version=candidate.version,
            committed_version=main.version,
            already_complete=True,
        )
    if main.version != source_version:
        raise ValueError(f"main advanced from version {source_version} to {main.version}")
    if [fragment.metadata.id for fragment in main.get_fragments()] != [
        fragment.metadata.id for fragment in candidate.get_fragments()
    ] or main.count_rows() != candidate.count_rows():
        raise ValueError("candidate fragments do not match the rollback source")

    operation = _candidate_merge_operation(candidate, source_version)
    _copy_candidate_data_directory(config.lance_uri, config.candidate_branch)
    committed = lance.LanceDataset.commit(
        main,
        operation,
        read_version=source_version,
        storage_options=storage_options,
        commit_message="Publish validated embedding candidate",
    )
    return EmbeddingPromotionResult(
        source_version=source_version,
        candidate_version=candidate.version,
        committed_version=committed.version,
        already_complete=False,
    )


def _parse_args() -> EmbeddingBackfillConfig | EmbeddingPromotionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backfill = subparsers.add_parser("backfill")
    backfill.add_argument("--lance-uri", required=True)
    backfill.add_argument("--branch", required=True)
    backfill.add_argument("--embedding", required=True, choices=("clap", "meanaudio_16k"))
    backfill.add_argument("--workers", required=True, type=int)
    backfill.add_argument("--batch-size", required=True, type=int)
    backfill.add_argument("--gpu-per-worker", required=True, type=float)
    backfill.add_argument("--checkpoint")
    backfill.add_argument("--build-index", action=argparse.BooleanOptionalAction, default=True)
    backfill.add_argument("--num-partitions", type=int)
    backfill.add_argument("--resume-dir", type=Path)
    backfill.add_argument("--result", type=Path)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--lance-uri", required=True)
    promote.add_argument("--candidate-branch", required=True)
    promote.add_argument("--rollback-tag", required=True)
    promote.add_argument("--columns", required=True, nargs="+")

    values = vars(parser.parse_args())
    command = values.pop("command")
    if command == "backfill":
        return EmbeddingBackfillConfig.model_validate(values, strict=True)
    values["columns"] = tuple(values["columns"])
    return EmbeddingPromotionConfig.model_validate(values, strict=True)


def main() -> None:
    """Run a backfill or candidate promotion."""
    config = _parse_args()
    if isinstance(config, EmbeddingBackfillConfig):
        backfill_embedding(config)
        return
    result = promote_embedding_candidate(config)
    _write_result(result, None)


if __name__ == "__main__":
    main()
