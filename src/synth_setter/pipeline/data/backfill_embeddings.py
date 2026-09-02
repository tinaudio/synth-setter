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

_EMBEDDING_ARTIFACT_METADATA_KEY = b"synth_setter.embedding.artifact"
_EMBEDDING_NAME_METADATA_KEY = b"synth_setter.embedding.name"
_PROGRESS_INTERVAL_SECONDS = 30.0
_RAY_EXCEPTION_RETRIES = 2


class _AudioEncoder(Protocol):
    """Encode policy-specific float32 audio batches into float32 embeddings."""

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode one policy-specific audio batch.

        :param audio: Normalized float32 audio: CLAP receives contiguous mono ``(B, T)``;
            MeanAudio receives contiguous ``(B, C, T)`` with one or two channels.
        :param sample_rate: Source sample rate in Hz.
        :returns: Float32 ``(B, 512)`` CLAP vectors or ``(B, 20, F)`` MeanAudio sequences.
        """
        ...


_WORKER_ENCODERS: dict[tuple[str, str], _AudioEncoder] = {}
_WORKER_PROCESS_ID: tuple[int, str] | None = None


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
    .. attribute :: run_id

        Invocation that scheduled the attempt.
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
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")


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
    .. attribute :: run_id

        Invocation that scheduled the attempt.
    .. attribute :: worker_id

        Stable worker-process identity.
    .. attribute :: attempt_uuid

        Worker-generated attempt identity.
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
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    worker_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    attempt_uuid: str = Field(pattern=r"^[0-9a-f]{32}$")


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
    .. attribute :: implementation_revision

        Git revision defining worker behavior.
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
    implementation_revision: str = Field(min_length=1)


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

    .. attribute :: run_id

        Unique invocation identity.
    .. attribute :: git_commit

        Git revision that ran the invocation.
    .. attribute :: implementation_revision

        Executed Python source identity.
    .. attribute :: source_version

        Main rollback version.
    .. attribute :: candidate_version

        Validated candidate version.
    .. attribute :: committed_version

        Main version after promotion.
    .. attribute :: already_complete

        Whether main already carried every candidate field.
    """

    run_id: str
    git_commit: str
    implementation_revision: str
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
    .. attribute :: implementation_revision

        Executed Python source identity.
    .. attribute :: branch

        Candidate branch name.
    .. attribute :: embedding

        Embedding registry key.
    .. attribute :: checkpoint

        Resolved model checkpoint.
    .. attribute :: artifact

        Resolved artifact identity.
    .. attribute :: workers

        Configured Ray worker count.
    .. attribute :: batch_size

        Configured transform batch size.
    .. attribute :: tasks_per_worker

        Configured process-recycling bound.
    .. attribute :: gpu_per_worker

        Configured GPU reservation per worker.
    .. attribute :: rows

        Rows in the published dataset.
    .. attribute :: fragments

        Fragments in the published dataset.
    .. attribute :: current_rows

        Rows computed by this invocation.
    .. attribute :: current_fragments

        Fragments computed by this invocation.
    .. attribute :: resumed_rows

        Rows reused from reconciliation reports.
    .. attribute :: resumed_fragments

        Fragments reused from reconciliation reports.
    .. attribute :: source_version

        Candidate version read by workers.
    .. attribute :: data_version

        Input version if already complete, otherwise the embedding merge version.
    .. attribute :: final_version

        Candidate version after optional indexing.
    .. attribute :: elapsed_seconds

        Total invocation wall time.
    .. attribute :: rows_per_second

        Current-invocation row throughput.
    .. attribute :: worker_processes

        Distinct current-invocation worker identities.
    .. attribute :: max_tasks_per_process

        Largest current-invocation task count on one worker.
    .. attribute :: peak_rss_bytes

        Largest current-invocation worker resident-memory peak.
    .. attribute :: peak_gpu_allocated_bytes

        Largest current-invocation PyTorch allocated-memory peak.
    .. attribute :: peak_gpu_reserved_bytes

        Largest current-invocation PyTorch reserved-memory peak.
    .. attribute :: already_complete

        Whether the requested embedding existed before this invocation.
    .. attribute :: index_built

        Whether this invocation built the canonical index.
    """

    run_id: str
    git_commit: str
    implementation_revision: str
    branch: str
    embedding: str
    checkpoint: str
    artifact: str
    workers: int
    batch_size: int
    tasks_per_worker: int
    gpu_per_worker: float
    rows: int
    fragments: int
    current_rows: int
    current_fragments: int
    resumed_rows: int
    resumed_fragments: int
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
    .. attribute :: implementation_revision

        Executed Python source identity.
    """

    run_id: str
    git_commit: str
    implementation_revision: str = ""


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
    .. attribute :: identity

        Invocation and implementation provenance.
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
    identity: _RunIdentity


@dataclass(frozen=True)
class _DispatchResult:
    """Separate reconciled reports from work completed by this invocation.

    .. attribute :: all_reports

        Complete source-ordered reports used for publication.
    .. attribute :: current_reports

        Source-ordered reports produced by the current run.
    """

    all_reports: tuple[_FragmentReport, ...]
    current_reports: tuple[_FragmentReport, ...]


@dataclass(frozen=True)
class _BackfillOutcome:
    """Bundle publication state consumed by result rendering.

    .. attribute :: dataset

        Final candidate snapshot.
    .. attribute :: dispatch

        Reconciled and current worker reports.
    .. attribute :: data_version

        Input version if already complete, otherwise the embedding merge version.
    .. attribute :: index_built

        Whether this invocation built the index.
    .. attribute :: already_complete

        Whether data publication was unnecessary on entry.
    """

    dataset: lance.LanceDataset
    dispatch: _DispatchResult
    data_version: int
    index_built: bool
    already_complete: bool


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
    .. attribute :: run_id

        Invocation expected on newly returned reports.
    """

    pending: list[ray.ObjectRef]
    reports: dict[int, _FragmentReport]
    fragment_ids: set[int]
    total_rows: int
    store: _ReportStore
    run_id: str


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


def _transform_batch(
    batch: pa.RecordBatch,
    *,
    task: _FragmentTask,
    spec: EmbeddingSpec,
    encoder: _AudioEncoder,
) -> pa.RecordBatch:
    """Encode one Lance callback batch and attach artifact metadata.

    :param batch: Source rows supplied by Lance.
    :param task: Strict fragment and artifact request.
    :param spec: Registry policy defining input and output columns.
    :param encoder: Worker-cached policy encoder.
    :returns: Policy output columns carrying artifact metadata.
    """
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import (
        _decoded_sources,
        _encode_columns,
        _output_columns,
    )

    encoded = _encode_columns(
        _decoded_sources(batch, spec.input_fields),
        task.sample_rate,
        [spec],
        [encoder],
    )
    field_metadata = {
        _EMBEDDING_NAME_METADATA_KEY: task.embedding.encode(),
        _EMBEDDING_ARTIFACT_METADATA_KEY: task.artifact.encode(),
    }
    columns = _output_columns(spec)
    schema = pa.schema(
        [encoded.schema.field(column).with_metadata(field_metadata) for column in columns]
    )
    return pa.RecordBatch.from_arrays(
        [encoded.column(column) for column in columns], schema=schema
    )


def _worker_id() -> str:
    """Return one UUID stable for the lifetime of the current worker process.

    The identity changes only when the process ID changes.

    :returns: Process-stable worker UUID.
    """
    global _WORKER_PROCESS_ID

    pid = os.getpid()
    if _WORKER_PROCESS_ID is None or _WORKER_PROCESS_ID[0] != pid:
        _WORKER_PROCESS_ID = pid, uuid.uuid4().hex
    return _WORKER_PROCESS_ID[1]


def _fragment_report(
    task: _FragmentTask,
    *,
    metadata: lance.FragmentMetadata,
    schema: pa.Schema,
    rows: int,
    started: float,
) -> _FragmentReport:
    """Serialize an uncommitted fragment and its worker telemetry.

    :param task: Strict request carrying invocation provenance.
    :param metadata: Uncommitted Lance fragment metadata.
    :param schema: Uncommitted Lance output schema.
    :param rows: Source fragment row count.
    :param started: Worker monotonic start time.
    :returns: Strict report ready to cross the Ray boundary.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    return _FragmentReport(
        fragment_id=task.fragment_id,
        metadata_json=json.dumps(metadata.to_json()),
        schema_ipc=base64.b64encode(schema.serialize().to_pybytes()).decode(),
        pid=os.getpid(),
        rows=rows,
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if cuda_available else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_available else 0,
        run_id=task.run_id,
        worker_id=_worker_id(),
        attempt_uuid=uuid.uuid4().hex,
    )


def _transform_fragment(task_value: object) -> _FragmentReport:
    """Write one fragment's embedding data without committing a manifest.

    :param task_value: Strict fragment request crossing the Ray process boundary.
    :returns: Strict serializable fragment metadata, schema, and resource report.
    :raises ValueError: The source fragment cannot be found.
    """
    import lance
    import pyarrow as pa
    import torch

    from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, _output_columns

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
    metadata, schema = fragment.merge_columns(
        lambda batch: _transform_batch(batch, task=task, spec=spec, encoder=encoder),
        list(spec.input_fields),
        batch_size=task.batch_size,
    )
    merged_schema = schema.to_pyarrow()
    output_schema = pa.schema(
        [merged_schema.field(column) for column in _output_columns(spec)]
    )
    return _fragment_report(
        task,
        metadata=metadata,
        schema=output_schema,
        rows=fragment.count_rows(),
        started=started,
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

    :param result: Serializable backfill or promotion result.
    :param destination: Optional JSON output path.
    :returns: The unchanged result.
    """
    payload = json.dumps(asdict(result), sort_keys=True)
    sys.stdout.write(f"{payload}\n")
    sys.stdout.flush()
    if destination is not None:
        destination.write_text(f"{payload}\n")
    return result


def _validate_embedding_output_schema(
    schema: pa.Schema,
    spec: EmbeddingSpec,
    artifact: bytes,
) -> None:
    """Reject embedding fields that violate policy type or identity contracts.

    :param schema: Candidate output schema.
    :param spec: Selected embedding policy.
    :param artifact: Expected artifact identity bytes.
    :raises ValueError: Output type, shape, or metadata violates the policy.
    """
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import _output_columns

    if not artifact:
        raise ValueError(f"dataset {spec.name} artifact bytes must be present")
    columns = _output_columns(spec)
    expected_metadata = {
        _EMBEDDING_NAME_METADATA_KEY: spec.name.encode(),
        _EMBEDDING_ARTIFACT_METADATA_KEY: artifact,
    }
    if any((schema.field(column).metadata or {}) != expected_metadata for column in columns):
        raise ValueError(f"dataset {spec.name} artifact identity does not match")
    if spec.name == "clap":
        if schema.field(spec.column).type != pa.list_(pa.float32(), 512):
            raise ValueError("clap embedding schema must be a 512-element float32 vector")
        return
    if spec.name != "meanaudio_16k":
        raise ValueError(f"unsupported promotion embedding policy {spec.name!r}")
    sequence_type = schema.field(spec.column).type
    vector_column = f"{spec.column}_vec"
    valid_sequence = (
        isinstance(sequence_type, pa.FixedShapeTensorType)
        and sequence_type.value_type == pa.float32()
        and len(sequence_type.shape) == 2
        and sequence_type.shape[0] == 20
        and sequence_type.shape[1] > 0
    )
    if not valid_sequence or schema.field(vector_column).type != pa.list_(pa.float32(), 20):
        raise ValueError("meanaudio_16k schema must contain float32 (20, F) and (20,) fields")


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
    _validate_embedding_output_schema(dataset.schema, spec, artifact)
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
    :raises RuntimeError: Index creation fails without a concurrent matching index.
    :raises ValueError: Index validation fails without a concurrent matching index.
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
    try:
        built = build_index(dataset, column, index=spec.index, config=config)
    except (RuntimeError, ValueError):
        dataset.checkout_latest()
        if _matching_index_exists(dataset, column, index=spec.index, config=config):
            return dataset, False
        raise
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
    """Hydrate only flat identity and report files into dedicated staging.

    :param store: Local and remote reconciliation locations.
    :raises ValueError: A remote path or local symlink escapes dedicated staging.
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
    if store.local_dir.is_symlink():
        raise ValueError(f"reconciliation directory {store.local_dir} must not be a symlink")
    for entry in list_entries(store.remote_uri, recursive=True):
        is_report = entry.path.startswith("fragment-") and entry.path.endswith(".json")
        if "/" in entry.path or entry.path != "identity.json" and not is_report:
            raise ValueError(f"unsafe reconciliation report path {entry.path!r}")
        destination = store.local_dir / entry.path
        if destination.is_symlink():
            raise ValueError(f"reconciliation report {destination} must not be a symlink")
        if destination.exists():
            continue
        download_to_path(f"{store.remote_uri}/{entry.path}", destination)


def _claim_cache_identity(path: Path, identity: _CacheIdentity) -> bool:
    """Atomically claim a local reconciliation directory for one operation.

    :param path: Final ``identity.json`` path.
    :param identity: Immutable operation identity.
    :returns: Whether this invocation created the claim.
    :raises ValueError: The winning claim belongs to another operation.
    """
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    payload = identity.model_dump_json(indent=2) + "\n"
    with temporary.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    created = False
    try:
        os.link(temporary, path)
        created = True
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)
    cached = _CacheIdentity.model_validate_json(path.read_text(), strict=True)
    if cached != identity:
        raise ValueError(f"resume directory {path.parent} has another identity")
    return created


def _validate_report_filename(path: Path, report: _FragmentReport) -> None:
    """Require durable filenames to carry the report's worker provenance.

    :param path: Durable report path.
    :param report: Strict report parsed from that path.
    :raises ValueError: Filename and report provenance differ.
    """
    expected = f"fragment-{report.fragment_id}-{report.worker_id}-{report.attempt_uuid}.json"
    if path.name != expected:
        raise ValueError(f"worker report filename {path.name!r} does not match its provenance")


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
    :raises ValueError: Identity, provenance, or staging-path validation fails.
    """
    from synth_setter.pipeline.r2_io import upload_to_uri

    store.local_dir.mkdir(parents=True, exist_ok=True)
    if store.local_dir.is_symlink():
        raise ValueError(f"resume directory {store.local_dir} must not be a symlink")
    identity_path = store.local_dir / "identity.json"
    created = _claim_cache_identity(identity_path, identity)
    if created and store.remote_uri is not None:
        upload_to_uri(identity_path, f"{store.remote_uri}/identity.json")
    _hydrate_remote_reports(store)
    reports: dict[int, _FragmentReport] = {}
    for path in sorted(store.local_dir.glob("fragment-*.json")):
        report = _FragmentReport.model_validate_json(path.read_text(), strict=True)
        _validate_report_filename(path, report)
        if report.fragment_id in fragment_ids:
            reports.setdefault(report.fragment_id, report)
    return reports


def _persist_report(store: _ReportStore, report: _FragmentReport) -> None:
    """Atomically persist one successful worker attempt before merge publication.

    :param store: Local and shared reconciliation locations.
    :param report: Strict worker report.
    """
    from synth_setter.pipeline.r2_io import upload_to_uri

    destination = store.local_dir / (
        f"fragment-{report.fragment_id}-{report.worker_id}-{report.attempt_uuid}.json"
    )
    temporary = store.local_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(report.model_dump_json() + "\n")
    temporary.replace(destination)
    if store.remote_uri is not None:
        upload_to_uri(destination, f"{store.remote_uri}/{destination.name}")


def _prepare_context(config: EmbeddingBackfillConfig, identity: _RunIdentity) -> _BackfillContext:
    """Resolve and validate one candidate source snapshot.

    :param config: Branch and model policy.
    :param identity: Resolved invocation and implementation provenance.
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
        identity=identity,
    )


def _prepare_dispatch(
    config: EmbeddingBackfillConfig, context: _BackfillContext
) -> _DispatchState:
    """Load durable reports and schedule only missing source fragments.

    :param config: Worker scheduling and resume policy.
    :param context: Validated immutable source operation.
    :returns: Pending tasks and durable report state.
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
        implementation_revision=(
            context.identity.implementation_revision
            or _implementation_revision(context.identity.git_commit)
        ),
    )
    store = _report_store(config, identity)
    reports = _load_reports(store, identity, fragment_ids)
    transform = ray.remote(
        num_cpus=1,
        num_gpus=config.gpu_per_worker,
        max_calls=config.tasks_per_worker,
        max_retries=_RAY_EXCEPTION_RETRIES,
        retry_exceptions=True,
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
                run_id=context.identity.run_id,
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
        run_id=context.identity.run_id,
    )


def _cancel_dispatch(pending: Sequence[ray.ObjectRef]) -> None:
    """Force-cancel unfinished Ray work before an invocation-level retry.

    :param pending: Outstanding task references.
    """
    import ray

    for reference in pending:
        ray.cancel(reference, force=True)


def _accept_current_report(
    state: _DispatchState,
    value: object,
    pending: Sequence[ray.ObjectRef],
) -> _FragmentReport:
    """Validate and durably record one current-invocation worker report.

    :param state: Expected fragments, provenance, and reconciliation store.
    :param value: Untrusted value returned across the Ray boundary.
    :param pending: Work cancelled if the report is inconsistent.
    :returns: Strict accepted report.
    :raises ValueError: Provenance, fragment identity, or uniqueness is invalid.
    """
    report = _FragmentReport.model_validate(value, strict=True)
    is_unexpected = (
        report.run_id != state.run_id
        or report.fragment_id not in state.fragment_ids
        or report.fragment_id in state.reports
    )
    if is_unexpected:
        _cancel_dispatch(pending)
        raise ValueError(f"unexpected worker report for fragment {report.fragment_id}")
    _persist_report(state.store, report)
    state.reports[report.fragment_id] = report
    return report


def _log_dispatch_progress(
    state: _DispatchState,
    *,
    config: EmbeddingBackfillConfig,
    rows_done: int,
    current_rows_done: int,
    started: float,
    now: float,
) -> None:
    """Log invocation throughput against aggregate reconciled progress.

    :param state: Current complete and expected fragment state.
    :param config: Embedding and branch labels.
    :param rows_done: Reconciled plus current completed rows.
    :param current_rows_done: Rows completed by this invocation.
    :param started: Invocation monotonic start.
    :param now: Current monotonic time.
    """
    elapsed = now - started
    rate = current_rows_done / elapsed
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


def _poll_dispatch(
    state: _DispatchState,
    config: EmbeddingBackfillConfig,
    started: float,
) -> _DispatchResult:
    """Collect strict reports until completion or the overall deadline.

    Successful reports are durable, so a timed-out invocation can be retried without recomputing
    completed fragments.

    :param state: Prepared task and reconciliation state.
    :param config: Deadline and progress policy.
    :param started: Invocation monotonic start.
    :returns: Complete publication reports and current-invocation reports.
    :raises TimeoutError: The deadline expires after pending tasks are cancelled.
    :raises ValueError: A worker returns an unknown or duplicate fragment report.
    """
    import ray

    pending = state.pending
    current_reports: list[_FragmentReport] = []
    rows_done = sum(report.rows for report in state.reports.values())
    current_rows_done = 0
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
            report = _accept_current_report(state, ray.get(ready[0]), pending)
            current_reports.append(report)
            rows_done += report.rows
            current_rows_done += report.rows
        now = time.monotonic()
        if now - last_log >= _PROGRESS_INTERVAL_SECONDS or not pending:
            _log_dispatch_progress(
                state,
                config=config,
                rows_done=rows_done,
                current_rows_done=current_rows_done,
                started=started,
                now=now,
            )
            last_log = now
    if state.reports.keys() != state.fragment_ids:
        raise ValueError("worker reports do not cover every source fragment")
    return _DispatchResult(
        all_reports=tuple(
            state.reports[fragment_id] for fragment_id in sorted(state.fragment_ids)
        ),
        current_reports=tuple(current_reports),
    )


def _decode_reports(
    reports: Sequence[_FragmentReport],
    context: _BackfillContext,
) -> tuple[list[lance.FragmentMetadata], pa.Schema]:
    """Decode reports and validate their fragment, schema, and path contracts.

    :param reports: Strict worker reports.
    :param context: Expected schema and artifact identity.
    :returns: Decoded fragment metadata and common output schema.
    :raises ValueError: Payload, schema, row count, or data path is invalid.
    """
    import lance
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import _output_columns

    metadata = [lance.FragmentMetadata.from_json(report.metadata_json) for report in reports]
    schemas = [
        pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc)))
        for report in reports
    ]
    if not schemas or any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("worker schemas differ")
    output_schema = schemas[0]
    if output_schema.names != list(_output_columns(context.spec)):
        raise ValueError("worker schema must contain exactly the policy output columns")
    _validate_embedding_output_schema(output_schema, context.spec, context.artifact.encode())
    for report, fragment in zip(reports, metadata, strict=True):
        if fragment.id != report.fragment_id or fragment.physical_rows != report.rows:
            raise ValueError(f"worker report payload differs for fragment {report.fragment_id}")
        payload = fragment.to_json()
        if not payload["files"]:
            raise ValueError(f"worker report fragment {report.fragment_id} has no data files")
        for data_file in payload["files"]:
            path = PurePosixPath(data_file["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe worker data path {str(path)!r}")
    source_schema = context.dataset.schema
    merged_schema = pa.schema(
        [*source_schema, *output_schema], metadata=source_schema.metadata
    )
    return metadata, merged_schema


def _backfill_publication_is_exact(
    latest: lance.LanceDataset,
    context: _BackfillContext,
    expected: lance.LanceOperation.Merge,
) -> bool:
    """Accept only the expected merge followed by index-only transactions.

    :param latest: Current candidate branch snapshot.
    :param context: Validated source rows and embedding identity.
    :param expected: Exact merge derived from worker reports.
    :returns: Whether the branch is an exact recoverable publication.
    """
    import lance

    if latest.version <= context.source_version or latest.count_rows() != context.total_rows:
        return False
    transaction = latest.read_transaction(context.source_version + 1)
    actual = None if transaction is None else transaction.operation
    if (
        not isinstance(actual, lance.LanceOperation.Merge)
        or actual.schema.to_pyarrow() != expected.schema.to_pyarrow()
    ):
        return False
    if [fragment.to_json() for fragment in actual.fragments] != [
        fragment.to_json() for fragment in expected.fragments
    ]:
        return False
    try:
        complete = _embedding_is_complete(latest, context.spec, context.artifact.encode())
    except ValueError:
        return False
    if not complete:
        return False
    for version in range(context.source_version + 2, latest.version + 1):
        later = latest.read_transaction(version)
        if later is None or not isinstance(later.operation, lance.LanceOperation.CreateIndex):
            return False
    return True


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
    :raises OSError: The commit failed without publishing the expected merge.
    :raises RuntimeError: The commit failed without publishing the expected merge.
    :raises ValueError: Another writer advanced the candidate incompatibly.
    """
    import lance

    metadata, schema = _decode_reports(reports, context)
    expected = lance.LanceOperation.Merge(metadata, schema)

    def latest_candidate() -> lance.LanceDataset:
        return lance.dataset(
            config.lance_uri, storage_options=context.storage_options
        ).checkout_version((config.branch, None))

    latest = latest_candidate()
    if latest.version != context.source_version:
        if _backfill_publication_is_exact(latest, context, expected):
            return latest
        raise ValueError(
            f"branch advanced from source version {context.source_version} to "
            f"incompatible version {latest.version}"
        )
    try:
        return lance.LanceDataset.commit(
            context.dataset,
            expected,
            read_version=context.source_version,
            storage_options=context.storage_options,
            commit_message=f"Add {config.embedding} embeddings",
        )
    except (OSError, RuntimeError, ValueError):
        try:
            recovered = latest_candidate()
            is_exact = _backfill_publication_is_exact(recovered, context, expected)
        except (OSError, RuntimeError, ValueError):
            is_exact = False
        if is_exact:
            return recovered
        raise


def _summarize_backfill(
    *,
    config: EmbeddingBackfillConfig,
    context: _BackfillContext,
    outcome: _BackfillOutcome,
    started: float,
) -> EmbeddingBackfillResult:
    """Render measured task and publication state into the public result.

    :param config: Branch and worker policy.
    :param context: Immutable source and row count.
    :param outcome: Final publication and worker state.
    :param started: Invocation monotonic start.
    :returns: Serializable backfill result.
    :raises RuntimeError: Observed worker reuse exceeds the configured bound.
    """
    elapsed = time.monotonic() - started
    current = outcome.dispatch.current_reports
    resumed_fragments = len(outcome.dispatch.all_reports) - len(current)
    current_rows = sum(report.rows for report in current)
    resumed_rows = sum(report.rows for report in outcome.dispatch.all_reports) - current_rows
    tasks_by_worker = Counter(report.worker_id for report in current)
    max_tasks = max(tasks_by_worker.values(), default=0)
    if max_tasks > config.tasks_per_worker:
        raise RuntimeError(
            f"Ray worker served {max_tasks} tasks, exceeding bound {config.tasks_per_worker}"
        )
    return EmbeddingBackfillResult(
        run_id=context.identity.run_id,
        git_commit=context.identity.git_commit,
        implementation_revision=(
            context.identity.implementation_revision
            or _implementation_revision(context.identity.git_commit)
        ),
        branch=config.branch,
        embedding=config.embedding,
        checkpoint=context.checkpoint,
        artifact=context.artifact,
        workers=config.workers,
        batch_size=config.batch_size,
        tasks_per_worker=config.tasks_per_worker,
        gpu_per_worker=config.gpu_per_worker,
        rows=outcome.dataset.count_rows(),
        fragments=len(outcome.dataset.get_fragments()),
        current_rows=current_rows,
        current_fragments=len(current),
        resumed_rows=resumed_rows,
        resumed_fragments=resumed_fragments,
        source_version=context.source_version,
        data_version=outcome.data_version,
        final_version=outcome.dataset.version,
        elapsed_seconds=elapsed,
        rows_per_second=current_rows / elapsed if current else 0.0,
        worker_processes=len(tasks_by_worker),
        max_tasks_per_process=max_tasks,
        peak_rss_bytes=max((report.peak_rss_bytes for report in current), default=0),
        peak_gpu_allocated_bytes=max(
            (report.peak_gpu_allocated_bytes for report in current), default=0
        ),
        peak_gpu_reserved_bytes=max(
            (report.peak_gpu_reserved_bytes for report in current), default=0
        ),
        already_complete=outcome.already_complete,
        index_built=outcome.index_built,
    )


def _run_backfill(
    config: EmbeddingBackfillConfig,
    identity: _RunIdentity,
    started: float,
) -> EmbeddingBackfillResult:
    """Prepare, reconcile, publish, and summarize one backfill.

    :param config: Strict backfill policy.
    :param identity: Validated invocation provenance.
    :param started: Invocation monotonic start.
    :returns: Measured publication result.
    """
    context = _prepare_context(config, identity)
    if _embedding_is_complete(context.dataset, context.spec, context.artifact.encode()):
        indexed, index_built = _ensure_embedding_index(
            context.dataset, context.spec, context.add_config
        )
        outcome = _BackfillOutcome(
            dataset=indexed,
            dispatch=_DispatchResult(all_reports=(), current_reports=()),
            data_version=context.source_version,
            index_built=index_built,
            already_complete=True,
        )
    else:
        dispatch = _poll_dispatch(_prepare_dispatch(config, context), config, started)
        committed = _commit_reports(config, context, dispatch.all_reports)
        data_version = context.source_version + 1
        indexed, index_built = _ensure_embedding_index(committed, context.spec, context.add_config)
        outcome = _BackfillOutcome(
            dataset=indexed,
            dispatch=dispatch,
            data_version=data_version,
            index_built=index_built,
            already_complete=False,
        )
    return _summarize_backfill(
        config=config,
        context=context,
        outcome=outcome,
        started=started,
    )


def _implementation_revision(git_commit: str) -> str:
    """Hash executed Python sources so dirty and installed code cannot share reports.

    :param git_commit: Validated source checkout commit.
    :returns: Commit-prefixed digest of executed package sources.
    """
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return f"{git_commit}:{digest.hexdigest()}"


def _resolve_source_git_sha() -> str:
    """Return the validated commit for the synth-setter source checkout.

    :returns: Lowercase 40-character source ``HEAD`` SHA.
    :raises RuntimeError: Git is unavailable or the source checkout has no valid SHA.
    """
    from synth_setter.utils.logging_utils import resolve_git_sha

    source_root = Path(__file__).resolve().parents[4]
    git_commit = resolve_git_sha(source_root)
    is_valid = len(git_commit) == 40 and all(
        character in "0123456789abcdef" for character in git_commit
    )
    if not is_valid:
        raise RuntimeError("embedding operation requires a validated source git SHA")
    return git_commit


def _new_run_identity() -> _RunIdentity:
    """Resolve one invocation and executed-source identity.

    :returns: Fresh run identity with validated source provenance.
    """
    git_commit = _resolve_source_git_sha()
    return _RunIdentity(
        run_id=uuid.uuid4().hex,
        git_commit=git_commit,
        implementation_revision=_implementation_revision(git_commit),
    )


def backfill_embedding(config: EmbeddingBackfillConfig) -> EmbeddingBackfillResult:
    """Write one embedding through recycled Ray workers and one branch merge commit.

    :param config: Strict branch, model, worker, deadline, and index configuration.
    :returns: Distributed write result with measured resource peaks and audit identities.
    :raises RuntimeError: CUDA is unavailable or Ray exceeds the process reuse bound.
    """
    import ray
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("distributed embedding backfill requires CUDA")
    started = time.monotonic()
    identity = _new_run_identity()
    ray.init(
        num_cpus=config.workers,
        num_gpus=1,
        include_dashboard=False,
        log_to_driver=False,
    )
    try:
        result = _run_backfill(config, identity, started)
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
    local_root = Path(unquote(urlparse(uri).path)) if uri.startswith("file://") else Path(uri)
    source_path = local_root / "tree" / branch / "data"
    destination_path = local_root / "data"
    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)


def _normalise_candidate_data_file(data_file: dict[str, object]) -> None:
    """Make a candidate file main-relative without permitting path traversal.

    :param data_file: Mutable Lance data-file JSON payload.
    :raises ValueError: Base identity or relative path is unsafe.
    """
    base_id = data_file.get("base_id")
    if base_id not in {None, 0}:
        raise ValueError(f"candidate data file uses unexpected base id {base_id}")
    path_value = data_file.get("path")
    if not isinstance(path_value, str):
        raise ValueError("candidate data file path must be a string")
    path = PurePosixPath(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe candidate data path {str(path)!r}")
    data_file["base_id"] = None


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
            _normalise_candidate_data_file(data_file)
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
    if (
        not isinstance(actual, lance.LanceOperation.Merge)
        or actual.schema.to_pyarrow() != expected.schema.to_pyarrow()
    ):
        return False
    expected_fragments = [fragment.to_json() for fragment in expected.fragments]
    if [fragment.to_json() for fragment in actual.fragments] != expected_fragments:
        return False
    if main.count_rows() != candidate.count_rows():
        return False
    current_fragments = [fragment.metadata.to_json() for fragment in main.get_fragments()]
    if current_fragments != expected_fragments:
        return False
    for version in range(source_version + 2, main.version + 1):
        later = main.read_transaction(version)
        if later is None or not isinstance(later.operation, lance.LanceOperation.CreateIndex):
            return False
    return True


@dataclass(frozen=True)
class _PromotionContext:
    """Hold validated candidate publication state.

    .. attribute :: main

        Current main snapshot.
    .. attribute :: candidate

        Validated candidate snapshot.
    .. attribute :: source_version

        Tagged common parent version.
    .. attribute :: storage_options

        Optional object-store credentials.
    """

    main: lance.LanceDataset
    candidate: lance.LanceDataset
    source_version: int
    storage_options: dict[str, str] | None


def _selected_promotion_specs(columns: Sequence[str]) -> tuple[EmbeddingSpec, ...]:
    """Resolve complete registry policies selected for promotion.

    :param columns: Requested top-level candidate columns.
    :returns: Registry policies in registry order.
    :raises ValueError: A column is unknown or only part of a policy is selected.
    """
    from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY, _output_columns

    selected = set(columns)
    column_owners = {
        column: spec
        for spec in EMBEDDING_REGISTRY.values()
        for column in _output_columns(spec)
    }
    unknown = selected - set(column_owners)
    if unknown:
        raise ValueError(f"unknown embedding columns {sorted(unknown)}")

    specs: list[EmbeddingSpec] = []
    for spec in EMBEDDING_REGISTRY.values():
        output_columns = set(_output_columns(spec))
        present = selected & output_columns
        if present and present != output_columns:
            raise ValueError(f"partial {spec.name} columns selected: {sorted(present)}")
        if present:
            specs.append(spec)
    return tuple(specs)


def _validate_promotion(config: EmbeddingPromotionConfig) -> _PromotionContext:
    """Resolve and validate candidate ancestry and schema isolation.

    :param config: Candidate branch, rollback identity, and output columns.
    :returns: Validated publication snapshots.
    :raises ValueError: Branch ancestry, rollback identity, or schemas are incompatible.
    """
    import lance
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import _output_columns

    specs = _selected_promotion_specs(config.columns)
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
        raise ValueError(
            f"candidate has unselected columns {sorted(candidate_names - expected_names)}"
        )
    source_schema_changed = candidate.schema.metadata != source.schema.metadata or any(
        candidate.schema.field(name) != source.schema.field(name) for name in source.schema.names
    )
    if source_schema_changed:
        raise ValueError("candidate changed the rollback source schema")
    for spec in specs:
        policy_schema = pa.schema(
            [candidate.schema.field(column) for column in _output_columns(spec)]
        )
        primary_metadata = policy_schema.field(spec.column).metadata or {}
        artifact = primary_metadata.get(_EMBEDDING_ARTIFACT_METADATA_KEY, b"")
        _validate_embedding_output_schema(policy_schema, spec, artifact)
    return _PromotionContext(
        main=main,
        candidate=candidate,
        source_version=source_version,
        storage_options=storage_options,
    )


def _main_has_exact_candidate(
    main: lance.LanceDataset,
    candidate: lance.LanceDataset,
    *,
    source_version: int,
    columns: Sequence[str],
) -> bool:
    """Prove selected schemas and merge provenance match the candidate.

    :param main: Current main snapshot.
    :param candidate: Validated candidate snapshot.
    :param source_version: Common parent version.
    :param columns: Selected candidate output columns.
    :returns: Whether current main contains the exact candidate merge.
    """
    selected = set(columns)
    if not selected.issubset(main.schema.names):
        return False
    if any(main.schema.field(name) != candidate.schema.field(name) for name in columns):
        return False
    return _main_contains_candidate_merge(main, candidate, source_version)


def _publish_candidate(
    config: EmbeddingPromotionConfig, context: _PromotionContext
) -> tuple[lance.LanceDataset, bool]:
    """Publish a validated candidate or prove its prior publication.

    :param config: Candidate branch and selected output columns.
    :param context: Validated candidate publication state.
    :returns: Main snapshot and whether publication was already complete.
    :raises OSError: The commit failed without publishing the exact candidate merge.
    :raises RuntimeError: The commit failed without publishing the exact candidate merge.
    :raises ValueError: Main advanced incompatibly or source fragments changed.
    """
    import lance

    main = context.main
    candidate = context.candidate
    present_main = set(config.columns) & set(main.schema.names)
    if present_main:
        if present_main != set(config.columns):
            raise ValueError(f"main has partial candidate columns {sorted(present_main)}")
        if not _main_has_exact_candidate(
            main,
            candidate,
            source_version=context.source_version,
            columns=config.columns,
        ):
            raise ValueError("main does not contain the validated candidate merge")
        return main, True
    if main.version != context.source_version:
        raise ValueError(
            f"main advanced from rollback version {context.source_version} to {main.version}"
        )
    main_ids = [fragment.metadata.id for fragment in main.get_fragments()]
    candidate_ids = [fragment.metadata.id for fragment in candidate.get_fragments()]
    if main_ids != candidate_ids or main.count_rows() != candidate.count_rows():
        raise ValueError("candidate fragments do not match the rollback source")
    operation = _main_merge_operation(candidate, config.candidate_branch, context.source_version)
    try:
        committed = lance.LanceDataset.commit(
            main,
            operation,
            read_version=context.source_version,
            storage_options=context.storage_options,
            commit_message="Publish validated CLAP and MeanAudio embeddings",
        )
    except (OSError, RuntimeError, ValueError):
        try:
            recovered = lance.dataset(
                config.lance_uri, storage_options=context.storage_options
            ).checkout_version((None, None))
            is_exact = _main_has_exact_candidate(
                recovered,
                candidate,
                source_version=context.source_version,
                columns=config.columns,
            )
        except (OSError, RuntimeError, ValueError):
            is_exact = False
        if is_exact:
            return recovered, True
        raise
    return committed, False


def promote_embedding_candidate(
    config: EmbeddingPromotionConfig,
) -> EmbeddingPromotionResult:
    """Publish validated branch fragment metadata through one main merge commit.

    :param config: Candidate branch, rollback identity, and output columns.
    :returns: Promotion version identities, provenance, and idempotency state.
    """
    identity = _new_run_identity()
    context = _validate_promotion(config)
    committed, already_complete = _publish_candidate(config, context)
    return EmbeddingPromotionResult(
        run_id=identity.run_id,
        git_commit=identity.git_commit,
        implementation_revision=identity.implementation_revision,
        source_version=context.source_version,
        candidate_version=context.candidate.version,
        committed_version=committed.version,
        already_complete=already_complete,
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
