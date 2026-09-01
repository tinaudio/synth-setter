"""Branch-safe distributed embedding backfill and candidate promotion.

Typical candidate publication uses the CLI after validating a branch::

    python -m synth_setter.pipeline.data.backfill_embeddings promote \\
        --lance-uri /data/split.lance --candidate-branch embeddings \\
        --rollback-tag pre-embeddings --columns clap meanaudio_16k
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import resource
import shutil
import sys
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload
from urllib.parse import unquote, urlparse

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    import lance
    import numpy as np
    import pyarrow as pa
    import ray

    from synth_setter.pipeline.data.add_embeddings import EmbeddingSpec
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

logger = structlog.get_logger(__name__)

_PROGRESS_INTERVAL_SECONDS = 30.0


class _AudioEncoder(Protocol):
    """Encode a decoded audio batch into one model-specific array."""

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode audio.

        :param audio: Float audio batch.
        :param sample_rate: Source sample rate in Hz.
        :returns: Model-specific embeddings.
        """
        ...


_WORKER_ENCODERS: dict[tuple[str, str], _AudioEncoder] = {}


class _FragmentTask(BaseModel):
    """Validate one immutable Ray fragment request.

    .. attribute :: model_config

        Strict immutable trust-boundary policy.
    .. attribute :: uri

        Lance-openable dataset target.
    .. attribute :: storage_options

        Optional object-store credentials.
    .. attribute :: branch

        Candidate branch name.
    .. attribute :: source_version

        Immutable branch-local source version.
    .. attribute :: fragment_id

        Source fragment ID.
    .. attribute :: embedding

        Supported registry key.
    .. attribute :: checkpoint

        Model checkpoint identity.
    .. attribute :: sample_rate

        Source audio sample rate in Hz.
    .. attribute :: batch_size

        Rows per callback batch.
    .. attribute :: artifact

        Versioned output identity.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    uri: str
    storage_options: dict[str, str] | None
    branch: str
    source_version: int
    fragment_id: int
    embedding: Literal["clap", "meanaudio_16k"]
    checkpoint: str
    sample_rate: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    artifact: str


class _FragmentReport(BaseModel):
    """Validate one resumable Ray worker report.

    .. attribute :: model_config

        Strict immutable trust-boundary policy.
    .. attribute :: fragment_id

        Transformed source fragment ID.
    .. attribute :: metadata_json

        Lance fragment metadata JSON.
    .. attribute :: schema_ipc

        Base64 Arrow schema IPC.
    .. attribute :: pid

        Worker process ID.
    .. attribute :: rows

        Transformed row count.
    .. attribute :: elapsed_seconds

        Fragment wall time.
    .. attribute :: peak_rss_bytes

        Worker peak resident memory.
    .. attribute :: peak_gpu_allocated_bytes

        PyTorch peak allocated GPU bytes.
    .. attribute :: peak_gpu_reserved_bytes

        PyTorch peak reserved GPU bytes.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    fragment_id: int
    metadata_json: str
    schema_ipc: str
    pid: int
    rows: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    peak_gpu_allocated_bytes: int = Field(ge=0)
    peak_gpu_reserved_bytes: int = Field(ge=0)


class _CacheIdentity(BaseModel):
    """Bind reconciliation reports to one immutable embedding operation.

    .. attribute :: model_config

        Strict immutable trust-boundary policy.
    .. attribute :: lance_uri

        Public dataset identity.
    .. attribute :: branch

        Candidate branch name.
    .. attribute :: source_version

        Immutable branch-local source version.
    .. attribute :: embedding

        Supported registry key.
    .. attribute :: checkpoint

        Model checkpoint identity.
    .. attribute :: sample_rate

        Source audio sample rate in Hz.
    .. attribute :: batch_size

        Rows per callback batch.
    .. attribute :: artifact

        Versioned output identity.
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


class EmbeddingBackfillConfig(BaseModel):
    """Validate one branch-scoped distributed embedding write.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: lance_uri

        Existing Lance dataset URI or local path.

    .. attribute :: branch

        Existing non-main candidate branch to mutate.

    .. attribute :: embedding

        One supported embedding registry key.

    .. attribute :: workers

        Maximum Ray worker processes.

    .. attribute :: batch_size

        Source rows decoded per fragment callback.

    .. attribute :: tasks_per_worker

        Fragment tasks served before Ray recycles a process and model.

    .. attribute :: gpu_per_worker

        Fractional Ray GPU reservation measured for this model.

    .. attribute :: checkpoint

        Optional checkpoint override.

    .. attribute :: build_index

        Whether to build the registry's canonical index.

    .. attribute :: num_partitions

        IVF partition override, or ``None`` for a row-derived count.

    .. attribute :: resume_dir

        Optional host-local reconciliation cache directory.

    .. attribute :: timeout_seconds

        Overall fragment-dispatch deadline before cancellation.

    .. attribute :: result

        Optional JSON result path.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(min_length=1)
    branch: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    embedding: Literal["clap", "meanaudio_16k"]
    workers: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    tasks_per_worker: int = Field(ge=1)
    gpu_per_worker: float = Field(gt=0, le=1)
    checkpoint: str | None = None
    build_index: bool = True
    num_partitions: int | None = Field(default=None, ge=1)
    resume_dir: Path | None = None
    timeout_seconds: float = Field(default=21_600.0, gt=0)
    result: Path | None = None

    @field_validator("branch")
    @classmethod
    def _branch_is_not_main(cls, value: str) -> str:
        """Reserve main publication for the candidate promotion path.

        :param value: Candidate branch name.
        :returns: Branch unchanged.
        :raises ValueError: The default branch was selected.
        """
        if value == "main":
            raise ValueError("distributed embedding writes require a candidate branch")
        return value


class EmbeddingPromotionConfig(BaseModel):
    """Validate promotion of one precomputed embedding candidate.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: lance_uri

        Existing Lance dataset URI or local path.

    .. attribute :: candidate_branch

        Branch carrying validated embedding fragments.

    .. attribute :: rollback_tag

        Immutable main snapshot from which the candidate branch was created.

    .. attribute :: columns

        Candidate columns published together on main.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(min_length=1)
    candidate_branch: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    rollback_tag: str = Field(min_length=1)
    columns: tuple[str, ...]

    @field_validator("columns")
    @classmethod
    def _columns_are_unique_and_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a nonempty ordered set of output columns.

        :param value: Candidate output columns.
        :returns: Columns unchanged.
        :raises ValueError: No column is selected or a name is repeated.
        """
        if not value:
            raise ValueError("columns must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("columns must not contain duplicates")
        return value


@dataclass(frozen=True)
class EmbeddingPromotionResult:
    """Summarize one candidate-to-main publication.

    .. attribute :: source_version

        Main rollback version.

    .. attribute :: candidate_version

        Validated candidate version.

    .. attribute :: committed_version

        Main version after promotion.

    .. attribute :: already_complete

        Whether main already carried every candidate field.
    """

    source_version: int
    candidate_version: int
    committed_version: int
    already_complete: bool


@dataclass(frozen=True)
class EmbeddingBackfillResult:
    """Summarize one branch-scoped distributed embedding write.

    .. attribute :: run_id

        Unique invocation identity.

    .. attribute :: git_commit

        Git revision that ran the invocation.

    .. attribute :: branch

        Candidate branch name.

    .. attribute :: embedding

        Embedding registry key.

    .. attribute :: rows

        Rows transformed.

    .. attribute :: fragments

        Fragments transformed.

    .. attribute :: source_version

        Candidate version read by workers.

    .. attribute :: data_version

        Candidate merge-commit version.

    .. attribute :: final_version

        Candidate version after optional indexing.

    .. attribute :: elapsed_seconds

        Total backfill wall time.

    .. attribute :: rows_per_second

        Aggregate backfill throughput.

    .. attribute :: worker_processes

        Distinct worker processes used.

    .. attribute :: max_tasks_per_process

        Largest observed task count in one process.

    .. attribute :: peak_rss_bytes

        Largest worker peak resident memory.

    .. attribute :: peak_gpu_allocated_bytes

        Largest worker PyTorch peak allocated GPU memory.

    .. attribute :: peak_gpu_reserved_bytes

        Largest worker PyTorch peak reserved GPU memory.

    .. attribute :: already_complete

        Whether the candidate already carried the requested embedding.

    .. attribute :: index_built

        Whether this invocation built the canonical index.
    """

    run_id: str
    git_commit: str
    branch: str
    embedding: str
    rows: int
    fragments: int
    source_version: int
    data_version: int
    final_version: int
    elapsed_seconds: float
    rows_per_second: float
    worker_processes: int
    max_tasks_per_process: int
    peak_rss_bytes: int
    peak_gpu_allocated_bytes: int
    peak_gpu_reserved_bytes: int
    already_complete: bool
    index_built: bool


@dataclass(frozen=True)
class _RunIdentity:
    """Carry audit identities shared by every result path.

    .. attribute :: run_id

        Unique invocation identity.
    .. attribute :: git_commit

        Implementation Git revision.
    """

    run_id: str
    git_commit: str


@dataclass(frozen=True)
class _BackfillContext:
    """Bind one candidate snapshot to its model and worker inputs.

    .. attribute :: dataset

        Immutable candidate snapshot.
    .. attribute :: storage_options

        Optional object-store credentials.
    .. attribute :: spec

        Registry embedding policy.
    .. attribute :: add_config

        Registry index and artifact policy.
    .. attribute :: checkpoint

        Model checkpoint identity.
    .. attribute :: artifact

        Versioned output identity.
    .. attribute :: source_version

        Candidate version read by workers.
    .. attribute :: sample_rate

        Source audio sample rate in Hz.
    .. attribute :: total_rows

        Expected transformed row count.
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


@dataclass(frozen=True)
class _BackfillOutcome:
    """Bundle publication state consumed by result rendering.

    .. attribute :: dataset

        Final candidate snapshot.
    .. attribute :: reports

        Worker reports, empty for an idempotent invocation.
    .. attribute :: data_version

        Version carrying embedding data.
    .. attribute :: index_built

        Whether this invocation built the index.
    """

    dataset: lance.LanceDataset
    reports: Sequence[_FragmentReport]
    data_version: int
    index_built: bool


@dataclass(frozen=True)
class _ReportStore:
    """Locate local staging and shared reconciliation reports.

    .. attribute :: local_dir

        Host-local atomic-write directory.
    .. attribute :: remote_uri

        Shared R2 prefix when applicable.
    """

    local_dir: Path
    remote_uri: str | None


@dataclass(frozen=True)
class _DispatchState:
    """Hold mutable report collection state for one Ray dispatch.

    .. attribute :: pending

        Outstanding Ray references.
    .. attribute :: reports

        Durable reports keyed by fragment ID.
    .. attribute :: fragment_ids

        Complete expected fragment ID set.
    .. attribute :: total_rows

        Expected source row count.
    .. attribute :: store

        Local and shared report locations.
    """

    pending: list[ray.ObjectRef]
    reports: dict[int, _FragmentReport]
    fragment_ids: set[int]
    total_rows: int
    store: _ReportStore


def _storage_options(uri: str) -> dict[str, str] | None:
    """Resolve credentials only for object-store datasets.

    :param uri: Lance dataset URI or local path.
    :returns: R2 Lance options for ``s3://`` URIs, otherwise ``None``.
    """
    if not uri.startswith("s3://"):
        return None
    from synth_setter.pipeline.r2_io import r2_storage_options

    return r2_storage_options()


def _worker_encoder(embedding: str, checkpoint: str) -> _AudioEncoder:
    """Load at most one encoder per recycled Ray worker process.

    :param embedding: Supported embedding registry key.
    :param checkpoint: Registry checkpoint source.
    :returns: Cached encoder callable.
    """
    key = embedding, checkpoint
    encoder = _WORKER_ENCODERS.get(key)
    if encoder is not None:
        return encoder
    from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

    _WORKER_ENCODERS.clear()
    config = AddEmbeddingsConfig(
        lance_uri="worker",
        embeddings=(embedding,),
        checkpoints={embedding: checkpoint},
        device="cuda",
        build_index=False,
    )
    loaded = cast("_AudioEncoder", EMBEDDING_REGISTRY[embedding].load_encoder(checkpoint, config))
    _WORKER_ENCODERS[key] = loaded
    return loaded


def _transform_fragment(task_value: object) -> _FragmentReport:
    """Write one fragment's embedding data without committing a manifest.

    :param task_value: Strict fragment request crossing the Ray process boundary.
    :returns: Strict serializable fragment metadata, schema, and resource report.
    :raises ValueError: The source fragment cannot be found.
    """
    import lance
    import pyarrow as pa
    import torch

    from synth_setter.pipeline.data.add_embeddings import (
        EMBEDDING_REGISTRY,
        _decoded_sources,
        _encode_columns,
        _output_columns,
    )

    task = _FragmentTask.model_validate(task_value, strict=True)
    started = time.monotonic()
    dataset = lance.dataset(task.uri, storage_options=task.storage_options).checkout_version(
        (task.branch, task.source_version)
    )
    fragment = dataset.get_fragment(task.fragment_id)
    if fragment is None:
        raise ValueError(f"missing fragment {task.fragment_id}")
    spec = EMBEDDING_REGISTRY[task.embedding]
    encoder = _worker_encoder(task.embedding, task.checkpoint)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
        encoded = _encode_columns(
            _decoded_sources(batch, spec.input_fields),
            task.sample_rate,
            [spec],
            [encoder],
        )
        field_metadata = {
            b"synth_setter.embedding.name": task.embedding.encode(),
            b"synth_setter.embedding.artifact": task.artifact.encode(),
        }
        columns = _output_columns(spec)
        schema = pa.schema(
            [encoded.schema.field(column).with_metadata(field_metadata) for column in columns]
        )
        return pa.RecordBatch.from_arrays(
            [encoded.column(column) for column in columns], schema=schema
        )

    metadata, schema = fragment.merge_columns(
        transform,
        list(spec.input_fields),
        batch_size=task.batch_size,
    )
    rows = fragment.count_rows()
    return _FragmentReport(
        fragment_id=task.fragment_id,
        metadata_json=json.dumps(metadata.to_json()),
        schema_ipc=base64.b64encode(schema.to_pyarrow().serialize().to_pybytes()).decode(),
        pid=os.getpid(),
        rows=rows,
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_gpu_allocated_bytes=(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        peak_gpu_reserved_bytes=(
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        ),
    )


@overload
def _write_result(
    result: EmbeddingBackfillResult, destination: Path | None
) -> EmbeddingBackfillResult: ...


@overload
def _write_result(
    result: EmbeddingPromotionResult, destination: Path | None
) -> EmbeddingPromotionResult: ...


def _write_result(
    result: EmbeddingBackfillResult | EmbeddingPromotionResult,
    destination: Path | None,
) -> EmbeddingBackfillResult | EmbeddingPromotionResult:
    """Print and optionally persist a dataclass result.

    :param result: Dataclass result to serialize.
    :param destination: Optional JSON output path.
    :returns: Result unchanged.
    """
    payload = json.dumps(asdict(result), sort_keys=True)
    sys.stdout.write(f"{payload}\n")
    sys.stdout.flush()
    if destination is not None:
        destination.write_text(f"{payload}\n")
    return result


def _embedding_is_complete(
    dataset: lance.LanceDataset, spec: EmbeddingSpec, artifact: bytes
) -> bool:
    """Validate the selected policy's complete output identity.

    :param dataset: Candidate branch dataset.
    :param spec: Embedding registry specification.
    :param artifact: Expected artifact identity.
    :returns: Whether every output field exists with matching metadata.
    :raises ValueError: Only part of the policy output exists or identity differs.
    """
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


def _ensure_embedding_index(
    dataset: lance.LanceDataset,
    spec: EmbeddingSpec,
    config: AddEmbeddingsConfig,
) -> tuple[lance.LanceDataset, bool]:
    """Build the selected registry index when requested and absent.

    :param dataset: Candidate branch dataset carrying the embedding.
    :param spec: Embedding registry specification.
    :param config: Add-embeddings index configuration.
    :returns: Latest dataset and whether this call built an index.
    """
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


def _validate_resume_directory(config: EmbeddingBackfillConfig, path: Path) -> Path:
    """Require reconciliation staging to be disjoint from protected paths.

    :param config: Dataset and output paths that staging must not overlap.
    :param path: Candidate staging directory.
    :returns: Resolved dedicated staging directory.
    :raises ValueError: The path is protected or overlaps local source/output data.
    """
    resolved = path.expanduser().resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve(), Path.cwd().resolve()}:
        raise ValueError(f"resume directory {resolved} is not a dedicated cleanup path")
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
        raise ValueError(f"resume directory {resolved} overlaps local dataset {dataset_path}")
    if config.result is not None:
        result_path = config.result.expanduser().resolve()
        if (
            result_path == resolved
            or resolved in result_path.parents
            or result_path in resolved.parents
        ):
            raise ValueError(f"result path {result_path} is inside resume directory {resolved}")
    return resolved


def _report_store(config: EmbeddingBackfillConfig, identity: _CacheIdentity) -> _ReportStore:
    """Resolve local staging and shared R2 reconciliation storage.

    :param config: Dataset and optional local report path.
    :param identity: Immutable source operation identity.
    :returns: Stable local directory and optional R2 metadata prefix.
    """
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
    """Hydrate shared reports before local trust-boundary validation.

    :param store: Local staging and optional R2 reconciliation prefix.
    """
    if store.remote_uri is None:
        return
    from synth_setter.pipeline.r2_io import (
        download_to_path,
        list_entries,
        r2_directory_exists,
    )

    if not r2_directory_exists(store.remote_uri):
        return
    for entry in list_entries(store.remote_uri):
        download_to_path(
            f"{store.remote_uri}/{entry.path}", store.local_dir / Path(entry.path).name
        )


def _load_reports(
    store: _ReportStore,
    identity: _CacheIdentity,
    fragment_ids: set[int],
) -> dict[int, _FragmentReport]:
    """Load only reports bound to the current immutable operation.

    :param store: Local and shared reconciliation locations.
    :param identity: Expected operation identity.
    :param fragment_ids: Valid source fragment IDs.
    :returns: Validated reports keyed by fragment ID.
    :raises ValueError: Existing staging belongs to another operation.
    """
    from synth_setter.pipeline.r2_io import upload_to_uri

    store.local_dir.mkdir(parents=True, exist_ok=True)
    identity_path = store.local_dir / "identity.json"
    if identity_path.exists():
        cached = _CacheIdentity.model_validate_json(identity_path.read_text(), strict=True)
        if cached != identity:
            raise ValueError(f"resume directory {store.local_dir} has another identity")
    _hydrate_remote_reports(store)
    if identity_path.exists():
        cached = _CacheIdentity.model_validate_json(identity_path.read_text(), strict=True)
        if cached != identity:
            raise ValueError(f"resume directory {store.local_dir} has another identity")
    else:
        identity_path.write_text(identity.model_dump_json(indent=2) + "\n")
        if store.remote_uri is not None:
            upload_to_uri(identity_path, f"{store.remote_uri}/identity.json")
    reports: dict[int, _FragmentReport] = {}
    for path in sorted(store.local_dir.glob("fragment-*.json")):
        report = _FragmentReport.model_validate_json(path.read_text(), strict=True)
        if report.fragment_id in fragment_ids:
            reports.setdefault(report.fragment_id, report)
    return reports


def _persist_report(store: _ReportStore, report: _FragmentReport) -> None:
    """Atomically persist one successful worker attempt before merge publication.

    :param store: Local and shared reconciliation locations.
    :param report: Strict worker report.
    """
    from synth_setter.pipeline.r2_io import upload_to_uri

    attempt = uuid.uuid4().hex
    destination = store.local_dir / f"fragment-{report.fragment_id}-{attempt}.json"
    temporary = store.local_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(report.model_dump_json() + "\n")
    temporary.replace(destination)
    if store.remote_uri is not None:
        upload_to_uri(destination, f"{store.remote_uri}/{destination.name}")


def _prepare_context(config: EmbeddingBackfillConfig) -> _BackfillContext:
    """Resolve and validate one candidate source snapshot.

    :param config: Branch and model policy.
    :returns: Immutable source, artifact, and model context.
    :raises ValueError: The candidate is empty or carries incompatible output fields.
    """
    import lance

    from synth_setter.pipeline.data.add_embeddings import (
        EMBEDDING_REGISTRY,
        _output_columns,
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
    if not _embedding_is_complete(dataset, spec, artifact.encode()) and any(
        column in dataset.schema.names for column in _output_columns(spec)
    ):
        raise ValueError(f"candidate has incompatible {config.embedding} output columns")
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


def _prepare_dispatch(
    config: EmbeddingBackfillConfig, context: _BackfillContext
) -> _DispatchState:
    """Load durable reports and schedule only missing source fragments.

    :param config: Ray and reconciliation policy.
    :param context: Immutable source and model context.
    :returns: Initialized dispatch state.
    """
    import ray

    fragments = context.dataset.get_fragments()
    fragment_ids = {fragment.metadata.id for fragment in fragments}
    identity = _CacheIdentity(
        lance_uri=config.lance_uri,
        branch=config.branch,
        source_version=context.source_version,
        embedding=config.embedding,
        checkpoint=context.checkpoint,
        sample_rate=context.sample_rate,
        batch_size=config.batch_size,
        artifact=context.artifact,
    )
    store = _report_store(config, identity)
    reports = _load_reports(store, identity, fragment_ids)
    transform = ray.remote(
        num_cpus=1,
        num_gpus=config.gpu_per_worker,
        max_calls=config.tasks_per_worker,
    )(_transform_fragment)
    pending = [
        transform.remote(
            _FragmentTask(
                uri=config.lance_uri,
                storage_options=context.storage_options,
                branch=config.branch,
                source_version=context.source_version,
                fragment_id=fragment_id,
                embedding=config.embedding,
                checkpoint=context.checkpoint,
                sample_rate=context.sample_rate,
                batch_size=config.batch_size,
                artifact=context.artifact,
            )
        )
        for fragment_id in sorted(fragment_ids - reports.keys())
    ]
    return _DispatchState(
        pending=pending,
        reports=reports,
        fragment_ids=fragment_ids,
        total_rows=context.total_rows,
        store=store,
    )


def _cancel_dispatch(pending: Sequence[ray.ObjectRef]) -> None:
    """Force-cancel unfinished Ray work before an invocation-level retry.

    :param pending: Outstanding task references.
    """
    import ray

    for reference in pending:
        ray.cancel(reference, force=True)


def _poll_dispatch(
    state: _DispatchState,
    config: EmbeddingBackfillConfig,
    started: float,
) -> list[_FragmentReport]:
    """Collect strict reports until completion or the overall deadline.

    Successful reports are durable, so a timed-out invocation can be retried without recomputing
    completed fragments.

    :param state: Prepared task and reconciliation state.
    :param config: Deadline and progress policy.
    :param started: Invocation monotonic start.
    :returns: One report per source fragment in source order.
    :raises TimeoutError: The deadline expires after pending tasks are cancelled.
    :raises ValueError: A worker returns an unknown or duplicate fragment report.
    """
    import ray

    pending = state.pending
    rows_done = sum(report.rows for report in state.reports.values())
    last_log = started
    while pending:
        remaining = config.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            _cancel_dispatch(pending)
            raise TimeoutError(
                f"fragment tasks exceeded {config.timeout_seconds} seconds; retry resumes "
                f"{len(state.reports)} completed fragments"
            )
        ready, pending = ray.wait(pending, num_returns=1, timeout=min(10.0, remaining))
        if ready:
            report = _FragmentReport.model_validate(ray.get(ready[0]), strict=True)
            if report.fragment_id not in state.fragment_ids or report.fragment_id in state.reports:
                _cancel_dispatch(pending)
                raise ValueError(f"unexpected worker report for fragment {report.fragment_id}")
            _persist_report(state.store, report)
            state.reports[report.fragment_id] = report
            rows_done += report.rows
        now = time.monotonic()
        if now - last_log >= _PROGRESS_INTERVAL_SECONDS or not pending:
            elapsed = now - started
            rate = rows_done / elapsed
            logger.info(
                "embedding_backfill_progress",
                embedding=config.embedding,
                branch=config.branch,
                rows=rows_done,
                total_rows=state.total_rows,
                fragments=len(state.reports),
                total_fragments=len(state.fragment_ids),
                rows_per_second=rate,
                elapsed_seconds=elapsed,
                eta_seconds=(state.total_rows - rows_done) / rate if rate > 0 else None,
            )
            last_log = now
    if state.reports.keys() != state.fragment_ids:
        raise ValueError("worker reports do not cover every source fragment")
    return [state.reports[fragment_id] for fragment_id in sorted(state.fragment_ids)]


def _decode_reports(
    reports: Sequence[_FragmentReport],
) -> tuple[list[lance.FragmentMetadata], pa.Schema]:
    """Decode trusted report payloads and enforce one worker schema.

    :param reports: Strict reports from durable staging.
    :returns: Lance metadata and common Arrow output schema.
    :raises ValueError: No schema exists or worker schemas differ.
    """
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
    """Publish one merge or recover an identically completed uncertain commit.

    :param config: Candidate branch identity.
    :param context: Immutable source and artifact context.
    :param reports: Complete strict fragment reports.
    :returns: Committed or recovered candidate snapshot.
    :raises ValueError: Another writer advanced the candidate incompatibly.
    """
    import lance

    latest = lance.dataset(
        config.lance_uri, storage_options=context.storage_options
    ).checkout_version((config.branch, None))
    if latest.version != context.source_version:
        if _embedding_is_complete(latest, context.spec, context.artifact.encode()):
            return latest
        raise ValueError(
            f"branch advanced from source version {context.source_version} to "
            f"incompatible version {latest.version}"
        )
    metadata, schema = _decode_reports(reports)
    return lance.LanceDataset.commit(
        context.dataset,
        lance.LanceOperation.Merge(metadata, schema),
        read_version=context.source_version,
        storage_options=context.storage_options,
        commit_message=f"Add {config.embedding} embeddings",
    )


def _summarize_backfill(
    config: EmbeddingBackfillConfig,
    context: _BackfillContext,
    outcome: _BackfillOutcome,
    identity: _RunIdentity,
    started: float,
) -> EmbeddingBackfillResult:
    """Render measured task and publication state into the public result.

    :param config: Branch and worker policy.
    :param context: Immutable source and row count.
    :param outcome: Final publication and worker state.
    :param identity: Run and Git audit identity.
    :param started: Invocation monotonic start.
    :returns: Serializable backfill result.
    :raises RuntimeError: Observed worker reuse exceeds the configured bound.
    """
    elapsed = time.monotonic() - started
    tasks_by_pid = Counter(report.pid for report in outcome.reports)
    max_tasks = max(tasks_by_pid.values(), default=0)
    if max_tasks > config.tasks_per_worker:
        raise RuntimeError(
            f"Ray worker served {max_tasks} tasks, exceeding bound {config.tasks_per_worker}"
        )
    return EmbeddingBackfillResult(
        run_id=identity.run_id,
        git_commit=identity.git_commit,
        branch=config.branch,
        embedding=config.embedding,
        rows=outcome.dataset.count_rows(),
        fragments=len(outcome.dataset.get_fragments()),
        source_version=context.source_version,
        data_version=outcome.data_version,
        final_version=outcome.dataset.version,
        elapsed_seconds=elapsed,
        rows_per_second=context.total_rows / elapsed if outcome.reports else 0.0,
        worker_processes=len(tasks_by_pid),
        max_tasks_per_process=max_tasks,
        peak_rss_bytes=max((report.peak_rss_bytes for report in outcome.reports), default=0),
        peak_gpu_allocated_bytes=max(
            (report.peak_gpu_allocated_bytes for report in outcome.reports), default=0
        ),
        peak_gpu_reserved_bytes=max(
            (report.peak_gpu_reserved_bytes for report in outcome.reports), default=0
        ),
        already_complete=not outcome.reports,
        index_built=outcome.index_built,
    )


def backfill_embedding(config: EmbeddingBackfillConfig) -> EmbeddingBackfillResult:
    """Write one embedding through recycled Ray workers and one branch merge commit.

    :param config: Strict branch, model, worker, deadline, and index configuration.
    :returns: Distributed write result with measured resource peaks and audit identities.
    :raises RuntimeError: CUDA is unavailable or Ray exceeds the process reuse bound.
    """
    import ray
    import torch

    from synth_setter.utils.logging_utils import resolve_git_sha

    if not torch.cuda.is_available():
        raise RuntimeError("distributed embedding backfill requires CUDA")
    started = time.monotonic()
    identity = _RunIdentity(run_id=uuid.uuid4().hex, git_commit=resolve_git_sha())
    ray.init(
        num_cpus=config.workers,
        num_gpus=1,
        include_dashboard=False,
        log_to_driver=False,
    )
    try:
        context = _prepare_context(config)
        if _embedding_is_complete(context.dataset, context.spec, context.artifact.encode()):
            indexed, index_built = _ensure_embedding_index(
                context.dataset, context.spec, context.add_config
            )
            result = _summarize_backfill(
                config,
                context,
                _BackfillOutcome(
                    dataset=indexed,
                    reports=(),
                    data_version=context.source_version,
                    index_built=index_built,
                ),
                identity,
                started,
            )
            return _write_result(result, config.result)
        state = _prepare_dispatch(config, context)
        reports = _poll_dispatch(state, config, started)
        committed = _commit_reports(config, context, reports)
        data_version = committed.version
        indexed, index_built = _ensure_embedding_index(committed, context.spec, context.add_config)
        result = _summarize_backfill(
            config,
            context,
            _BackfillOutcome(
                dataset=indexed,
                reports=reports,
                data_version=data_version,
                index_built=index_built,
            ),
            identity,
            started,
        )
        return _write_result(result, config.result)
    finally:
        ray.shutdown()


def _copy_candidate_data_directory(uri: str, branch: str) -> None:
    """Copy branch-owned uncommitted files into main's data directory.

    :param uri: Lance dataset URI or local path.
    :param branch: Candidate branch whose primary data directory owns the files.
    """
    root = uri.rstrip("/")
    if uri.startswith("s3://"):
        from synth_setter.pipeline import r2_io

        source = r2_io.from_s3_uri(f"{root}/tree/{branch}/data")
        destination = r2_io.from_s3_uri(f"{root}/data")
        r2_io.copy_directory(source, destination)
        return
    local_root = (
        Path(unquote(urlparse(uri).path)) if uri.startswith("file://") else Path(uri)
    )
    source_path = local_root / "tree" / branch / "data"
    destination_path = local_root / "data"
    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)


def _candidate_merge_operation(
    candidate: lance.LanceDataset, source_version: int
) -> lance.LanceOperation.Merge:
    """Rebuild the candidate's latest merge with main-relative data references.

    :param candidate: Validated candidate branch dataset.
    :param source_version: Candidate parent version on main.
    :returns: Normalized merge operation without copying files.
    :raises ValueError: No merge exists or a file uses an unexpected base.
    """
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
                raise ValueError(f"candidate data file uses unexpected base id {base_id}")
        fragments.append(lance.FragmentMetadata.from_json(json.dumps(payload)))
    return lance.LanceOperation.Merge(fragments, candidate.schema)


def _main_merge_operation(
    candidate: lance.LanceDataset, branch: str, source_version: int
) -> lance.LanceOperation.Merge:
    """Materialize candidate files and return their normalized merge operation.

    :param candidate: Validated candidate branch dataset.
    :param branch: Candidate branch name.
    :param source_version: Candidate parent version on main.
    :returns: Lance merge operation whose data paths are main-relative.
    """
    operation = _candidate_merge_operation(candidate, source_version)
    _copy_candidate_data_directory(candidate.uri, branch)
    return operation


def _main_contains_candidate_merge(
    main: lance.LanceDataset,
    candidate: lance.LanceDataset,
    source_version: int,
) -> bool:
    """Prove that main's first post-source commit published this candidate.

    :param main: Current main snapshot, potentially followed by index commits.
    :param candidate: Validated branch snapshot.
    :param source_version: Common parent version.
    :returns: Whether fragment metadata and schema match the promoted candidate.
    """
    import lance

    if main.version <= source_version:
        return False
    expected = _candidate_merge_operation(candidate, source_version)
    transaction = main.read_transaction(source_version + 1)
    if transaction is None:
        return False
    actual = transaction.operation
    if not isinstance(actual, lance.LanceOperation.Merge) or actual.schema != expected.schema:
        return False
    actual_fragments = [fragment.to_json() for fragment in actual.fragments]
    expected_fragments = [fragment.to_json() for fragment in expected.fragments]
    return actual_fragments == expected_fragments


def promote_embedding_candidate(
    config: EmbeddingPromotionConfig,
) -> EmbeddingPromotionResult:
    """Publish validated branch fragment metadata through one main merge commit.

    :param config: Candidate branch, rollback identity, and output columns.
    :returns: Promotion version identities and idempotency state.
    :raises ValueError: Branch ancestry, rollback identity, or schemas are incompatible.
    """
    import lance

    storage_options = _storage_options(config.lance_uri)
    root = lance.dataset(config.lance_uri, storage_options=storage_options)
    main = root.checkout_version((None, None))
    candidate = root.checkout_version((config.candidate_branch, None))
    tags = root.tags.list()
    rollback = tags.get(config.rollback_tag)
    if rollback is None or rollback["branch"] not in {None, "main"}:
        raise ValueError(f"rollback tag {config.rollback_tag!r} must identify main")
    source_version = rollback["version"]
    source = root.checkout_version((None, source_version))
    branches = root.branches.list()
    branch = branches.get(config.candidate_branch)
    if (
        branch is None
        or branch["parent_branch"] is not None
        or branch["parent_version"] != source_version
    ):
        raise ValueError(
            f"candidate branch {config.candidate_branch!r} must descend from main "
            f"rollback version {source_version}"
        )
    missing_candidate = set(config.columns) - set(candidate.schema.names)
    if missing_candidate:
        raise ValueError(f"candidate lacks columns {sorted(missing_candidate)}")
    expected_names = set(source.schema.names) | set(config.columns)
    candidate_names = set(candidate.schema.names)
    if candidate_names != expected_names:
        extras = sorted(candidate_names - expected_names)
        raise ValueError(f"candidate has unselected columns {extras}")
    if candidate.schema.metadata != source.schema.metadata or any(
        candidate.schema.field(name) != source.schema.field(name) for name in source.schema.names
    ):
        raise ValueError("candidate changed the rollback source schema")
    present_main = set(config.columns) & set(main.schema.names)
    if present_main:
        if present_main != set(config.columns):
            raise ValueError(f"main has partial candidate columns {sorted(present_main)}")
        if any(main.schema.field(name) != candidate.schema.field(name) for name in config.columns):
            raise ValueError("main candidate column schema differs from the validated branch")
        if not _main_contains_candidate_merge(main, candidate, source_version):
            raise ValueError("main does not contain the validated candidate merge")
        return EmbeddingPromotionResult(
            source_version=source_version,
            candidate_version=candidate.version,
            committed_version=main.version,
            already_complete=True,
        )
    if main.version != source_version:
        raise ValueError(f"main advanced from rollback version {source_version} to {main.version}")
    main_fragments = main.get_fragments()
    candidate_fragments = candidate.get_fragments()
    main_ids = [fragment.metadata.id for fragment in main_fragments]
    candidate_ids = [fragment.metadata.id for fragment in candidate_fragments]
    if main_ids != candidate_ids or main.count_rows() != candidate.count_rows():
        raise ValueError("candidate fragments do not match the rollback source")

    operation = _main_merge_operation(candidate, config.candidate_branch, source_version)
    committed = lance.LanceDataset.commit(
        main,
        operation,
        read_version=source_version,
        storage_options=storage_options,
        commit_message="Publish validated CLAP and MeanAudio embeddings",
    )
    return EmbeddingPromotionResult(
        source_version=source_version,
        candidate_version=candidate.version,
        committed_version=committed.version,
        already_complete=False,
    )


def _parse_args() -> EmbeddingBackfillConfig | EmbeddingPromotionConfig:
    """Parse the distributed backfill and promotion CLI.

    :returns: Strict command configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backfill = subparsers.add_parser("backfill")
    backfill.add_argument("--lance-uri", required=True)
    backfill.add_argument("--branch", required=True)
    backfill.add_argument("--embedding", required=True, choices=("clap", "meanaudio_16k"))
    backfill.add_argument("--workers", required=True, type=int)
    backfill.add_argument("--batch-size", required=True, type=int)
    backfill.add_argument("--tasks-per-worker", required=True, type=int)
    backfill.add_argument("--gpu-per-worker", required=True, type=float)
    backfill.add_argument("--checkpoint")
    backfill.add_argument("--build-index", action=argparse.BooleanOptionalAction, default=True)
    backfill.add_argument("--num-partitions", type=int)
    backfill.add_argument("--resume-dir", type=Path)
    backfill.add_argument("--timeout-seconds", type=float, default=21_600.0)
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
    """Run one distributed candidate write or candidate promotion."""
    config = _parse_args()
    if isinstance(config, EmbeddingBackfillConfig):
        backfill_embedding(config)
        return
    _write_result(promote_embedding_candidate(config), None)


if __name__ == "__main__":
    main()
