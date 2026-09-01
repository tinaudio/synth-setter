"""Distribute stored full-resolution sketch pooling across Ray workers.

Typical usage::

    synth-setter-backfill-sketch-pool --lance-uri r2://bucket/train.lance \\
        --workers 32 --rollback-tag pre-sketch-pool
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import structlog
from pydantic import BaseModel, ConfigDict, Field

from synth_setter.data.vst.shapes import SKETCH_STRUCT_FIELD, SKETCH_VEC_CHILD
from synth_setter.pipeline.data.lance_retry import retry_lance_io as _retry
from synth_setter.utils.logging_utils import resolve_git_sha

if TYPE_CHECKING:
    import lance
    import pyarrow as pa
    import ray
    from lance.fragment import FragmentMetadata
    from lance.progress import IndexProgress


class _ColumnRename(TypedDict):
    """Describe the required subset of one Lance column alteration.

    .. attribute :: path

        Existing top-level column path.

    .. attribute :: name

        Replacement column name.
    """

    path: str
    name: str


class _AlterColumnsDataset(Protocol):
    """Expose Lance's runtime variadic column-alteration contract."""

    def alter_columns(self, *alterations: _ColumnRename) -> None:
        """Apply each column alteration.

        :param *alterations: Column renames to commit together.
        """


class _IndexDescriptor(TypedDict):
    """Describe the Lance index fields used by compatibility checks.

    .. attribute :: name

        Index name.

    .. attribute :: type

        Lance index type.

    .. attribute :: fields

        Indexed column paths.
    """

    name: str
    type: str
    fields: list[str]


logger = structlog.get_logger(__name__)
_INDEX_NAME = "sketch_pool_vec_idx"
_PROGRESS_INTERVAL_SECONDS = 30.0


class SketchPoolBackfillConfig(BaseModel):
    """Validate one distributed stored-sketch migration.

    .. attribute :: model_config

        Strict immutable trust-boundary configuration.

    .. attribute :: lance_uri

        Existing Lance dataset URI or local path.

    .. attribute :: branch

        Existing Lance branch to mutate; ``main`` selects the default branch.

    .. attribute :: workers

        Ray fragment workers.

    .. attribute :: batch_size

        Source rows decoded by each fragment callback.

    .. attribute :: tasks_per_worker

        Fragment tasks served before Ray recycles a worker process.

    .. attribute :: rollback_tag

        Immutable pre-migration tag created before the rename commit.

    .. attribute :: build_index

        Whether to build the canonical sketch vector index after pooling.

    .. attribute :: num_partitions

        IVF partition override, or ``None`` for a row-derived count.

    .. attribute :: resume_dir

        Worker-report cache, or ``None`` for a dataset-keyed user cache.

    .. attribute :: resume_uri

        Shared reconciliation prefix, or ``None`` for an R2 dataset-derived prefix.

    .. attribute :: timeout_seconds

        Overall fragment-task deadline.

    .. attribute :: result

        Optional JSON result path.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(min_length=1)
    branch: str = Field(default="main", min_length=1)
    workers: int = Field(ge=1)
    batch_size: int = Field(default=128, ge=1)
    tasks_per_worker: int = Field(default=4, ge=1)
    rollback_tag: str = Field(min_length=1)
    build_index: bool = True
    num_partitions: int | None = Field(default=None, ge=1)
    resume_dir: Path | None = None
    resume_uri: str | None = None
    timeout_seconds: float = Field(default=21_600.0, gt=0.0)
    result: Path | None = None


class _FragmentTask(BaseModel):
    """Validate one Ray fragment request at the process boundary.

    .. attribute :: model_config

        Strict immutable trust-boundary configuration.

    .. attribute :: uri

        Lance-openable dataset target.

    .. attribute :: storage_options

        Optional object-store credentials.

    .. attribute :: branch

        Source branch.

    .. attribute :: source_version

        Immutable branch-local source version.

    .. attribute :: fragment_id

        Source fragment ID.

    .. attribute :: batch_size

        Rows per callback batch.

    .. attribute :: artifact

        Pooling-policy identity.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    uri: str
    storage_options: dict[str, str] | None
    branch: str
    source_version: int
    fragment_id: int
    batch_size: int
    artifact: str


class _FragmentReport(BaseModel):
    """Validate one resumable Ray fragment report at the process boundary.

    .. attribute :: model_config

        Strict immutable trust-boundary configuration.

    .. attribute :: fragment_id

        Transformed source fragment ID.

    .. attribute :: metadata_json

        Lance fragment metadata JSON.

    .. attribute :: schema_ipc

        Base64 Arrow schema IPC.

    .. attribute :: row_count

        Transformed rows.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    fragment_id: int
    metadata_json: str
    schema_ipc: str
    row_count: int = Field(ge=0)


class _SubIndexStats(BaseModel):
    """Validate the PQ details returned by Lance index statistics.

    .. attribute :: model_config

        Strict immutable trust-boundary configuration.

    .. attribute :: num_sub_vectors

        PQ sub-vector count.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    num_sub_vectors: int


class _IndexStats(BaseModel):
    """Validate one IVF-PQ segment returned by Lance index statistics.

    .. attribute :: model_config

        Strict immutable trust-boundary configuration.

    .. attribute :: num_partitions

        IVF partition count.

    .. attribute :: metric_type

        Distance metric.

    .. attribute :: sub_index

        PQ details.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    num_partitions: int
    metric_type: str
    sub_index: _SubIndexStats


class _IndexStatistics(BaseModel):
    """Validate the Lance statistics envelope used for compatibility checks.

    .. attribute :: model_config

        Strict immutable trust-boundary configuration.

    .. attribute :: indices

        Physical IVF-PQ segment statistics.

    .. attribute :: num_unindexed_rows

        Rows not covered by the index.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    indices: list[_IndexStats] = Field(min_length=1)
    num_unindexed_rows: int = Field(ge=0)


class _CacheIdentity(BaseModel):
    """Bind cached fragment reports to one immutable migration input.

    .. attribute :: model_config

        Strict immutable trust-boundary configuration.

    .. attribute :: lance_uri

        Public dataset URI.

    .. attribute :: branch

        Source branch.

    .. attribute :: source_version

        Immutable branch-local source version.

    .. attribute :: batch_size

        Rows per callback batch.

    .. attribute :: artifact

        Pooling-policy identity.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str
    branch: str
    source_version: int
    batch_size: int
    artifact: str


@dataclass(frozen=True)
class _ReportStore:
    """Locate local staging and shared reconciliation reports.

    .. attribute :: local_dir

        Host-local atomic-write staging directory.

    .. attribute :: remote_uri

        Shared R2 prefix, or ``None`` for local-only migrations.
    """

    local_dir: Path
    remote_uri: str | None


@dataclass(frozen=True)
class _DispatchState:
    """Hold mutable state while the driver collects Ray reports.

    .. attribute :: pending

        Outstanding Ray object references.

    .. attribute :: reports

        Durable reports keyed by fragment ID.

    .. attribute :: fragment_ids

        Complete expected source fragment ID set.

    .. attribute :: total_rows

        Source rows for progress reporting.

    .. attribute :: report_store

        Local staging and shared report locations.
    """

    pending: list[ray.ObjectRef]
    reports: dict[int, _FragmentReport]
    fragment_ids: set[int]
    total_rows: int
    report_store: _ReportStore


@dataclass(frozen=True)
class SketchPoolBackfillResult:
    """Summarize a committed or already-complete migration.

    .. attribute :: run_id

        Unique invocation ID.

    .. attribute :: git_commit

        Implementation Git revision.

    .. attribute :: branch

        Mutated Lance branch.

    .. attribute :: rows

        Rows in the final snapshot.

    .. attribute :: fragments

        Fragments in the final snapshot.

    .. attribute :: source_version

        Version read by the data operation.

    .. attribute :: committed_version

        Final version after data and index publication.

    .. attribute :: elapsed_seconds

        End-to-end wall time.

    .. attribute :: rows_per_second

        End-to-end row throughput.

    .. attribute :: already_complete

        Whether data publication was skipped.

    .. attribute :: index_built

        Whether this invocation built the index.

    .. attribute :: index_requested

        Whether index construction was requested.

    .. attribute :: index_skip_reason

        Why no index exists, or ``None`` when one exists.

    .. attribute :: workers

        Configured Ray workers.

    .. attribute :: batch_size

        Rows per callback batch.

    .. attribute :: tasks_per_worker

        Tasks before worker recycling.

    .. attribute :: index_name

        Canonical index name when requested.

    .. attribute :: index_metric

        Canonical distance metric when requested.

    .. attribute :: num_partitions

        IVF partition count when requested.

    .. attribute :: num_sub_vectors

        PQ sub-vector count when requested.
    """

    run_id: str
    git_commit: str
    branch: str
    rows: int
    fragments: int
    source_version: int
    committed_version: int
    elapsed_seconds: float
    rows_per_second: float
    already_complete: bool
    index_built: bool
    index_requested: bool
    index_skip_reason: str | None
    workers: int
    batch_size: int
    tasks_per_worker: int
    index_name: str | None
    index_metric: str | None
    num_partitions: int | None
    num_sub_vectors: int | None


def _branch_reference(branch: str, version: int | None) -> tuple[str | None, int | None]:
    """Build a Lance branch reference with an explicit main representation.

    :param branch: Configured branch name.
    :param version: Version on that branch, or ``None`` for latest.
    :returns: Lance ``(branch, version)`` reference.
    """
    return (None if branch == "main" else branch, version)


def _lance_target(uri: str) -> tuple[str, dict[str, str] | None]:
    """Resolve a public dataset URI to the target Lance can open.

    :param uri: Lance dataset URI or local path.
    :returns: Openable URI or path plus object-store credentials when required.
    """
    from synth_setter.pipeline.r2_io import is_r2_uri, lance_target, r2_storage_options

    if is_r2_uri(uri):
        return lance_target(uri)
    if uri.startswith("s3://"):
        return uri, r2_storage_options()
    return uri, None


def _open_dataset(
    uri: str,
    storage_options: dict[str, str] | None,
    branch: str,
    version: int | None,
) -> lance.LanceDataset:
    """Open one branch snapshot under the migration retry policy.

    :param uri: Lance-openable dataset target.
    :param storage_options: Object-store credentials, when required.
    :param branch: Branch to open.
    :param version: Branch-local version, or ``None`` for latest.
    :returns: Open Lance dataset snapshot.
    """
    import lance

    return _retry(
        "open_dataset",
        lambda: lance.dataset(uri, storage_options=storage_options).checkout_version(
            _branch_reference(branch, version)
        ),
    )


def _pool_fragment_batch(batch: pa.RecordBatch, artifact: str) -> pa.RecordBatch:
    """Pool one decoded full-resolution fragment batch.

    :param batch: Batch carrying the renamed full-resolution sketch struct.
    :param artifact: Pooling-policy identity stored in field metadata.
    :returns: Batch carrying only the canonical pooled sketch struct.
    """
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import (
        SKETCH_FULL_STRUCT_FIELD,
        _encode_sketch_pool_column,
    )

    rows = batch.column(SKETCH_FULL_STRUCT_FIELD).to_numpy(zero_copy_only=False)
    encoded = _encode_sketch_pool_column(
        {SKETCH_FULL_STRUCT_FIELD: rows}, 0, lambda values: values
    )
    field = pa.field(
        SKETCH_STRUCT_FIELD,
        encoded.type,
        metadata={
            b"synth_setter.embedding.name": b"sketch_pool",
            b"synth_setter.embedding.artifact": artifact.encode(),
        },
    )
    return pa.RecordBatch.from_arrays([encoded], schema=pa.schema([field]))


def _transform_fragment(task_value: object) -> _FragmentReport:
    """Write one fragment's pooled sketch column without committing a manifest.

    :param task_value: Strictly validated fragment task from the Ray driver.
    :returns: Validated, JSON-persistable fragment metadata and row count.
    :raises ValueError: The task or source fragment is invalid.
    """
    from synth_setter.pipeline.data.add_embeddings import SKETCH_FULL_STRUCT_FIELD

    task = _FragmentTask.model_validate(task_value, strict=True)
    dataset = _open_dataset(
        task.uri,
        task.storage_options,
        task.branch,
        task.source_version,
    )
    fragment = _retry("get_fragment", lambda: dataset.get_fragment(task.fragment_id))
    if fragment is None:
        raise ValueError(f"missing fragment {task.fragment_id}")

    metadata, schema = _retry(
        "merge_fragment_columns",
        lambda: fragment.merge_columns(
            lambda batch: _pool_fragment_batch(batch, task.artifact),
            [SKETCH_FULL_STRUCT_FIELD],
            batch_size=task.batch_size,
        ),
    )
    return _FragmentReport(
        fragment_id=task.fragment_id,
        metadata_json=json.dumps(metadata.to_json(), sort_keys=True),
        schema_ipc=base64.b64encode(schema.to_pyarrow().serialize().to_pybytes()).decode("ascii"),
        row_count=_retry("count_fragment_rows", fragment.count_rows),
    )


def _is_complete(dataset: lance.LanceDataset, artifact: bytes) -> bool:
    """Check whether the current snapshot carries the expected pooled field.

    :param dataset: Open Lance dataset.
    :param artifact: Expected pooling-policy identity.
    :returns: Whether both source and canonical output are present and compatible.
    """
    from synth_setter.pipeline.data.add_embeddings import SKETCH_FULL_STRUCT_FIELD

    if {SKETCH_FULL_STRUCT_FIELD, SKETCH_STRUCT_FIELD} - set(dataset.schema.names):
        return False
    metadata = dataset.schema.field(SKETCH_STRUCT_FIELD).metadata or {}
    return (
        metadata.get(b"synth_setter.embedding.name") == b"sketch_pool"
        and metadata.get(b"synth_setter.embedding.artifact") == artifact
    )


def _index_parameters(
    dataset: lance.LanceDataset, config: SketchPoolBackfillConfig
) -> tuple[int, str, int]:
    """Resolve the canonical index parameters for validation and creation.

    :param dataset: Dataset whose row count determines default partitions.
    :param config: Index partition override.
    :returns: Partition count, metric, and PQ sub-vector count.
    :raises RuntimeError: The registry policy lacks its index specification.
    """
    from synth_setter.pipeline.data.add_embeddings import EMBEDDING_REGISTRY

    index = EMBEDDING_REGISTRY["sketch_pool"].index
    if index is None:
        raise RuntimeError("sketch_pool registry policy has no index specification")
    rows = _retry("count_index_rows", dataset.count_rows)
    partitions = config.num_partitions or max(1, round(rows**0.5))
    return partitions, index.metric, index.num_sub_vectors


def _validate_index(
    dataset: lance.LanceDataset,
    candidate: _IndexDescriptor,
    expected_partitions: int,
    expected_metric: str,
    expected_sub_vectors: int,
) -> None:
    """Verify an existing canonical index is operational and configuration-compatible.

    :param dataset: Dataset carrying the candidate index.
    :param candidate: Lance index descriptor.
    :param expected_partitions: Required IVF partition count.
    :param expected_metric: Required distance metric.
    :param expected_sub_vectors: Required PQ sub-vector count.
    :raises ValueError: Identity, type, configuration, or ANN plan is incompatible.
    """
    from synth_setter.pipeline.data.add_embeddings import SKETCH_VEC_COLUMN

    name = candidate.get("name")
    if name != _INDEX_NAME or candidate.get("type") != "IVF_PQ":
        raise ValueError(
            f"existing {SKETCH_VEC_COLUMN!r} index must be {_INDEX_NAME!r} IVF_PQ, "
            f"got name={name!r} type={candidate.get('type')!r}"
        )
    os.environ.setdefault("LANCE_INCLUDE_VECTOR_CENTROIDS", "false")
    payload = _retry("read_index_statistics", lambda: dataset.index_statistics(_INDEX_NAME))
    statistics = _IndexStatistics.model_validate(payload, strict=True)
    if len(statistics.indices) != 1:
        raise ValueError(f"existing {_INDEX_NAME!r} has {len(statistics.indices)} index segments")
    stats = statistics.indices[0]
    if statistics.num_unindexed_rows != 0:
        raise ValueError(
            f"existing {_INDEX_NAME!r} leaves {statistics.num_unindexed_rows} rows unindexed"
        )
    actual = (
        stats.num_partitions,
        stats.metric_type.lower(),
        stats.sub_index.num_sub_vectors,
    )
    expected = (expected_partitions, expected_metric.lower(), expected_sub_vectors)
    if actual != expected:
        raise ValueError(f"existing {_INDEX_NAME!r} config {actual!r} does not match {expected!r}")
    vector_field = dataset.schema.field(SKETCH_STRUCT_FIELD).type.field(SKETCH_VEC_CHILD)
    vector = [0.0] * vector_field.type.list_size
    plan = _retry(
        "explain_index_query",
        lambda: dataset.scanner(
            nearest={"column": SKETCH_VEC_COLUMN, "q": vector, "k": 1}
        ).explain_plan(),
    )
    if _INDEX_NAME not in plan:
        raise ValueError(f"ANN plan does not select {_INDEX_NAME!r}: {plan}")


def _canonical_indexes(dataset: lance.LanceDataset) -> list[_IndexDescriptor]:
    """List indexes targeting the canonical pooled vector child.

    :param dataset: Dataset snapshot to inspect.
    :returns: Canonical vector index descriptors.
    """
    from synth_setter.pipeline.data.add_embeddings import SKETCH_VEC_COLUMN

    return [
        cast("_IndexDescriptor", candidate)
        for candidate in _retry("list_indices", dataset.list_indices)
        if cast("_IndexDescriptor", candidate)["fields"] == [SKETCH_VEC_COLUMN]
    ]


def _index_progress_callback(started: float) -> Callable[[IndexProgress], None]:
    """Create a structured Lance index progress callback.

    :param started: Monotonic index-build start time.
    :returns: Callback logging stage rates and estimated time remaining.
    """

    def log_progress(progress: IndexProgress) -> None:
        elapsed = time.monotonic() - started
        rate = progress.completed / elapsed if progress.completed is not None else None
        remaining = (
            (progress.total - progress.completed) / rate
            if progress.total is not None
            and progress.completed is not None
            and rate is not None
            and rate > 0
            else None
        )
        logger.info(
            "sketch_pool_index_progress",
            progress_event=progress.event,
            stage=progress.stage,
            completed=progress.completed,
            total=progress.total,
            unit=progress.unit,
            rate=rate,
            elapsed_seconds=elapsed,
            eta_seconds=remaining,
        )

    return log_progress


def _ensure_canonical_index(
    dataset: lance.LanceDataset, config: SketchPoolBackfillConfig
) -> tuple[lance.LanceDataset, bool]:
    """Build or validate the pooled-vector index when requested.

    :param dataset: Dataset carrying canonical pooled sketches.
    :param config: Index build selection and partition override.
    :returns: Latest dataset snapshot and whether this call built an index.
    :raises ValueError: Existing canonical indexes are ambiguous or incompatible.
    """
    from synth_setter.pipeline.data.add_embeddings import MIN_ROWS_FOR_INDEX, SKETCH_VEC_COLUMN

    rows = _retry("count_index_eligibility_rows", dataset.count_rows)
    if not config.build_index:
        return dataset, False
    partitions, metric, sub_vectors = _index_parameters(dataset, config)
    existing = _canonical_indexes(dataset)
    if len(existing) > 1:
        raise ValueError(f"multiple indexes target {SKETCH_VEC_COLUMN!r}")
    if existing:
        _validate_index(dataset, existing[0], partitions, metric, sub_vectors)
        return dataset, False
    if rows < MIN_ROWS_FOR_INDEX:
        return dataset, False
    progress_callback = _index_progress_callback(time.monotonic())

    def create_or_recover() -> lance.LanceDataset:
        latest = dataset.checkout_version(_branch_reference(config.branch, None))
        recovered = _canonical_indexes(latest)
        if recovered:
            if len(recovered) != 1:
                raise ValueError(f"multiple indexes target {SKETCH_VEC_COLUMN!r}")
            _validate_index(latest, recovered[0], partitions, metric, sub_vectors)
            return latest
        return latest.create_index(
            SKETCH_VEC_COLUMN,
            index_type="IVF_PQ",
            name=_INDEX_NAME,
            metric=metric,
            num_partitions=partitions,
            num_sub_vectors=sub_vectors,
            progress_callback=progress_callback,
        )

    indexed = _retry("create_index", create_or_recover)
    created = _canonical_indexes(indexed)
    if len(created) != 1:
        raise ValueError(f"index creation produced {len(created)} canonical indexes")
    _validate_index(indexed, created[0], partitions, metric, sub_vectors)
    return indexed, True


def _resume_directory(config: SketchPoolBackfillConfig) -> Path:
    """Resolve a stable, dataset-keyed fragment-report cache directory.

    :param config: Migration identity and optional cache override.
    :returns: Cache directory unique to the public dataset URI and branch.
    """
    if config.resume_dir is not None:
        return config.resume_dir
    key = hashlib.sha256(f"{config.lance_uri}\0{config.branch}".encode()).hexdigest()[:20]
    return Path.home() / ".cache" / "synth-setter" / "sketch-pool-backfill" / key


def _report_store(
    config: SketchPoolBackfillConfig,
    identity: _CacheIdentity,
) -> _ReportStore:
    """Resolve local staging and host-independent reconciliation storage.

    :param config: Dataset and optional report-location configuration.
    :param identity: Immutable source operation identity.
    :returns: Local staging directory and optional shared R2 prefix.
    """
    from synth_setter.pipeline.r2_io import is_r2_uri

    digest = hashlib.sha256(identity.model_dump_json().encode()).hexdigest()[:20]
    remote_uri = config.resume_uri
    if remote_uri is None and is_r2_uri(config.lance_uri):
        remote_uri = f"{config.lance_uri.rstrip('/')}/metadata/workers/sketch-pool/{digest}"
    return _ReportStore(local_dir=_resume_directory(config), remote_uri=remote_uri)


def _hydrate_remote_reports(store: _ReportStore) -> None:
    """Download shared reconciliation reports into local validation staging.

    :param store: Local and optional shared report locations.
    """
    if store.remote_uri is None:
        return
    from synth_setter.pipeline.r2_io import download_to_path, list_entries

    for entry in list_entries(store.remote_uri):
        destination = store.local_dir / Path(entry.path).name
        download_to_path(f"{store.remote_uri}/{entry.path}", destination)


def _load_reports(
    store: _ReportStore, identity: _CacheIdentity, fragment_ids: set[int]
) -> dict[int, _FragmentReport]:
    """Load compatible durable worker reports for an interrupted merge.

    :param store: Local staging and shared report locations.
    :param identity: Immutable source operation identity.
    :param fragment_ids: Current source fragment IDs.
    :returns: Valid reports keyed by fragment ID.
    :raises ValueError: Existing cache belongs to a different source operation.
    """
    from synth_setter.pipeline.r2_io import upload_to_uri

    cache_dir = store.local_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    _hydrate_remote_reports(store)
    identity_path = cache_dir / "identity.json"
    if identity_path.exists():
        cached = _CacheIdentity.model_validate_json(identity_path.read_text(), strict=True)
        if cached != identity:
            raise ValueError(
                f"resume cache {cache_dir} identifies a different source operation; "
                "remove it or pass --resume-dir"
            )
    else:
        identity_path.write_text(identity.model_dump_json(indent=2) + "\n")
        if store.remote_uri is not None:
            upload_to_uri(identity_path, f"{store.remote_uri}/identity.json")
    reports: dict[int, _FragmentReport] = {}
    for path in sorted(cache_dir.glob("fragment-*.json")):
        report = _FragmentReport.model_validate_json(path.read_text(), strict=True)
        if report.fragment_id in fragment_ids:
            reports.setdefault(report.fragment_id, report)
    return reports


def _persist_report(store: _ReportStore, report: _FragmentReport) -> None:
    """Atomically persist one uniquely named worker-attempt report.

    :param store: Local staging and shared report locations.
    :param report: Strict worker result to persist.
    """
    from synth_setter.pipeline.r2_io import upload_to_uri

    attempt = uuid.uuid4().hex
    destination = store.local_dir / f"fragment-{report.fragment_id}-{attempt}.json"
    temporary = store.local_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(report.model_dump_json() + "\n")
    temporary.replace(destination)
    if store.remote_uri is not None:
        upload_to_uri(destination, f"{store.remote_uri}/{destination.name}")


def _prepare_dispatch(
    dataset: lance.LanceDataset,
    config: SketchPoolBackfillConfig,
    lance_uri: str,
    storage_options: dict[str, str] | None,
    source_version: int,
    artifact: str,
) -> _DispatchState:
    """Load durable reports and submit only missing fragment tasks.

    :param dataset: Immutable source snapshot.
    :param config: Worker, batch, and recycling controls.
    :param lance_uri: Lance-openable dataset target.
    :param storage_options: Object-store credentials, when required.
    :param source_version: Immutable branch-local source version.
    :param artifact: Pooling-policy identity.
    :returns: Initialized mutable dispatch state.
    """
    import ray

    fragments = _retry("list_fragments", dataset.get_fragments)
    fragment_ids = {fragment.metadata.id for fragment in fragments}
    identity = _CacheIdentity(
        lance_uri=config.lance_uri,
        branch=config.branch,
        source_version=source_version,
        batch_size=config.batch_size,
        artifact=artifact,
    )
    report_store = _report_store(config, identity)
    reports = _load_reports(report_store, identity, fragment_ids)
    remote_transform = ray.remote(num_cpus=1, max_calls=config.tasks_per_worker)(
        _transform_fragment
    )
    pending = [
        remote_transform.remote(
            _FragmentTask(
                uri=lance_uri,
                storage_options=storage_options,
                branch=config.branch,
                source_version=source_version,
                fragment_id=fragment_id,
                batch_size=config.batch_size,
                artifact=artifact,
            )
        )
        for fragment_id in sorted(fragment_ids - reports.keys())
    ]
    return _DispatchState(
        pending=pending,
        reports=reports,
        fragment_ids=fragment_ids,
        total_rows=_retry("count_source_rows", dataset.count_rows),
        report_store=report_store,
    )


def _poll_dispatch(
    state: _DispatchState,
    config: SketchPoolBackfillConfig,
    started: float,
) -> None:
    """Collect, validate, persist, and report one dispatch to completion.

    :param state: Prepared mutable dispatch state.
    :param config: Overall timeout policy.
    :param started: Migration start time for progress rates.
    :raises TimeoutError: The configured deadline expires; pending tasks are cancelled.
    :raises ValueError: A worker report is duplicate or unknown.
    """
    import ray

    pending = state.pending
    rows_done = sum(report.row_count for report in state.reports.values())
    last_log = started
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=10)
        if ready:
            report = _FragmentReport.model_validate(ray.get(ready[0]), strict=True)
            if report.fragment_id not in state.fragment_ids or report.fragment_id in state.reports:
                raise ValueError(f"unexpected worker report for fragment {report.fragment_id}")
            _persist_report(state.report_store, report)
            state.reports[report.fragment_id] = report
            rows_done += report.row_count
        now = time.monotonic()
        if now - started > config.timeout_seconds:
            for reference in pending:
                ray.cancel(reference, force=True)
            raise TimeoutError(
                f"fragment tasks exceeded {config.timeout_seconds} seconds with "
                f"{len(pending)} pending"
            )
        if now - last_log >= _PROGRESS_INTERVAL_SECONDS or not pending:
            elapsed = now - started
            rate = rows_done / elapsed
            logger.info(
                "sketch_pool_backfill_progress",
                rows=rows_done,
                total_rows=state.total_rows,
                rows_per_second=rate,
                fragments=len(state.reports),
                total_fragments=len(state.fragment_ids),
                elapsed_seconds=elapsed,
                eta_seconds=(state.total_rows - rows_done) / rate if rate > 0 else None,
            )
            last_log = now


def _run_fragment_tasks(
    dataset: lance.LanceDataset,
    config: SketchPoolBackfillConfig,
    lance_uri: str,
    storage_options: dict[str, str] | None,
    source_version: int,
    artifact: str,
    started: float,
) -> list[_FragmentReport]:
    """Run missing fragment tasks and durably record each successful output.

    :param dataset: Immutable source snapshot.
    :param config: Worker, batch, and recycling controls.
    :param lance_uri: Lance-openable dataset target.
    :param storage_options: Object-store credentials, when required.
    :param source_version: Immutable branch-local source version.
    :param artifact: Pooling-policy identity.
    :param started: Migration start time for progress rates.
    :returns: One validated report per source fragment.
    :raises ValueError: Worker reports do not cover every source fragment.
    """
    state = _prepare_dispatch(
        dataset,
        config,
        lance_uri,
        storage_options,
        source_version,
        artifact,
    )
    _poll_dispatch(state, config, started)
    if state.reports.keys() != state.fragment_ids:
        raise ValueError("worker reports do not cover every source fragment")
    return [state.reports[fragment_id] for fragment_id in sorted(state.fragment_ids)]


def _decode_reports(
    reports: Sequence[_FragmentReport],
) -> tuple[list[FragmentMetadata], pa.Schema]:
    """Decode validated report payloads and enforce one worker schema.

    :param reports: Validated reports loaded from the operation cache.
    :returns: Lance fragment metadata list and the common Arrow schema.
    :raises ValueError: Reports are empty or worker schemas differ.
    """
    import pyarrow as pa
    from lance.fragment import FragmentMetadata

    metadata = [FragmentMetadata.from_json(report.metadata_json) for report in reports]
    schemas = [
        pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc)))
        for report in reports
    ]
    if not schemas or any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("worker schemas differ")
    return metadata, schemas[0]


def _commit_reports(
    dataset: lance.LanceDataset,
    reports: Sequence[_FragmentReport],
    config: SketchPoolBackfillConfig,
    lance_uri: str,
    storage_options: dict[str, str] | None,
    source_version: int,
    artifact: bytes,
) -> lance.LanceDataset:
    """Commit worker reports with post-error publication recovery.

    :param dataset: Immutable source snapshot.
    :param reports: Complete validated fragment reports.
    :param config: Target branch identity.
    :param lance_uri: Lance-openable dataset target.
    :param storage_options: Object-store credentials, when required.
    :param source_version: Version read by every worker.
    :param artifact: Expected pooled-field identity.
    :returns: Committed or recovered published snapshot.
    """
    import lance

    metadata, schema = _decode_reports(reports)

    def commit_or_recover() -> lance.LanceDataset:
        latest = _open_dataset(lance_uri, storage_options, config.branch, None)
        if latest.version != source_version:
            if _is_complete(latest, artifact):
                return latest
            raise ValueError(
                f"branch advanced from source version {source_version} to "
                f"incompatible version {latest.version}"
            )
        operation = lance.LanceOperation.Merge(metadata, schema)
        return lance.LanceDataset.commit(
            dataset,
            operation,
            read_version=source_version,
            storage_options=storage_options,
            commit_message="Pool stored full-resolution sketch controls",
        )

    return _retry("commit_merge", commit_or_recover)


def _clear_resume_cache(config: SketchPoolBackfillConfig) -> None:
    """Remove consumed fragment reports after the merge commit succeeds.

    :param config: Migration whose report cache has been consumed.
    """
    cache_dir = _resume_directory(config)
    try:
        shutil.rmtree(cache_dir)
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "sketch_pool_resume_cleanup_failed",
            resume_dir=str(cache_dir),
            exc_info=True,
        )


def _write_result(
    result: SketchPoolBackfillResult, destination: Path | None
) -> SketchPoolBackfillResult:
    """Print and optionally persist one migration result.

    :param result: Completed migration summary.
    :param destination: Optional JSON output path.
    :returns: The unchanged result.
    """
    payload = json.dumps(asdict(result), sort_keys=True)
    sys.stdout.write(f"{payload}\n")
    sys.stdout.flush()
    if destination is not None:
        destination.write_text(f"{payload}\n")
    return result


def _result(
    config: SketchPoolBackfillConfig,
    dataset: lance.LanceDataset,
    source_version: int,
    started: float,
    already_complete: bool,
    index_built: bool,
    run_id: str,
) -> SketchPoolBackfillResult:
    """Build the auditable operation result from the final snapshot.

    :param config: Executed migration configuration.
    :param dataset: Final committed dataset snapshot.
    :param source_version: Version read by the data operation.
    :param started: Monotonic operation start.
    :param already_complete: Whether data publication was skipped.
    :param index_built: Whether this call published the index.
    :param run_id: Unique invocation ID.
    :returns: Persistable migration result.
    """

    elapsed = time.monotonic() - started
    partitions: int | None = None
    metric: str | None = None
    sub_vectors: int | None = None
    index_name: str | None = None
    index_skip_reason: str | None = "disabled"
    canonical_indices = _canonical_indexes(dataset)
    if config.build_index:
        partitions, metric, sub_vectors = _index_parameters(dataset, config)
        index_name = _INDEX_NAME
        index_skip_reason = None if canonical_indices else "below_min_rows"
    rows = _retry("count_result_rows", dataset.count_rows)
    return SketchPoolBackfillResult(
        run_id=run_id,
        git_commit=resolve_git_sha(),
        branch=config.branch,
        rows=rows,
        fragments=len(_retry("list_result_fragments", dataset.get_fragments)),
        source_version=source_version,
        committed_version=dataset.version,
        elapsed_seconds=elapsed,
        rows_per_second=0.0 if already_complete else rows / elapsed,
        already_complete=already_complete,
        index_built=index_built,
        index_requested=config.build_index,
        index_skip_reason=index_skip_reason,
        workers=config.workers,
        batch_size=config.batch_size,
        tasks_per_worker=config.tasks_per_worker,
        index_name=index_name,
        index_metric=metric,
        num_partitions=partitions,
        num_sub_vectors=sub_vectors,
    )


def _create_rollback_tag(
    dataset: lance.LanceDataset,
    config: SketchPoolBackfillConfig,
) -> None:
    """Create the required tag on the current pre-migration snapshot.

    :param dataset: Canonical-only source snapshot.
    :param config: Required tag and branch identity.
    :raises ValueError: The tag cannot be observed after publication.
    """
    rollback_tag = config.rollback_tag

    def create_or_recover_tag() -> None:
        if dataset.tags.list().get(rollback_tag) is not None:
            return
        dataset.tags.create(
            rollback_tag,
            _branch_reference(config.branch, dataset.version),
        )

    _retry("create_rollback_tag", create_or_recover_tag)
    if _retry("list_created_tag", dataset.tags.list).get(rollback_tag) is None:
        raise ValueError(f"rollback tag {rollback_tag!r} was not published")


def _validate_rollback_tag(
    dataset: lance.LanceDataset,
    config: SketchPoolBackfillConfig,
    lance_uri: str,
    storage_options: dict[str, str] | None,
    expected_version: int | None,
) -> None:
    """Validate an existing immutable pre-migration rollback snapshot.

    :param dataset: Current branch snapshot.
    :param config: Required tag and branch identity.
    :param lance_uri: Lance-openable dataset target.
    :param storage_options: Object-store credentials, when required.
    :param expected_version: Exact source version before rename, when known.
    :raises ValueError: The tag is missing or identifies the wrong branch, version, or schema.
    """
    from synth_setter.pipeline.data.add_embeddings import SKETCH_FULL_STRUCT_FIELD

    rollback_tag = config.rollback_tag
    existing = _retry("list_tags", dataset.tags.list).get(rollback_tag)
    if existing is None:
        raise ValueError(f"rollback tag {rollback_tag!r} must already exist")
    allowed_branches = {None, "main"} if config.branch == "main" else {config.branch}
    if existing["branch"] not in allowed_branches:
        raise ValueError(
            f"rollback tag {rollback_tag!r} identifies branch {existing['branch']!r}, "
            f"not {config.branch!r}"
        )
    if expected_version is not None and existing["version"] != expected_version:
        raise ValueError(
            f"rollback tag {rollback_tag!r} identifies version {existing['version']}, "
            f"not source version {expected_version}"
        )
    tagged = _open_dataset(
        lance_uri,
        storage_options,
        config.branch,
        existing["version"],
    )
    names = set(tagged.schema.names)
    if SKETCH_STRUCT_FIELD not in names or SKETCH_FULL_STRUCT_FIELD in names:
        raise ValueError(
            f"rollback tag {rollback_tag!r} does not identify the canonical-only "
            "pre-migration schema"
        )


def _validate_full_source_field(
    dataset: lance.LanceDataset,
    field_name: str,
) -> None:
    """Require the exact historical 401-frame nested sketch schema before mutation.

    :param dataset: Source snapshot to inspect.
    :param field_name: Canonical or renamed source field name.
    :raises ValueError: Nested children do not match the historical fixed-size layout.
    """
    import pyarrow as pa

    from synth_setter.data.vst.shapes import (
        SKETCH_CENTROID_CHILD,
        SKETCH_LOUDNESS_CHILD,
        SKETCH_PITCH_BINS,
        SKETCH_PITCH_CHILD,
    )
    from synth_setter.pipeline.data.add_embeddings import SKETCH_FULL_FRAMES

    field = dataset.schema.field(field_name)
    if not pa.types.is_struct(field.type):
        raise ValueError(f"{field_name!r} must be a nested sketch struct")
    expected_sizes = {
        SKETCH_LOUDNESS_CHILD: SKETCH_FULL_FRAMES,
        SKETCH_CENTROID_CHILD: SKETCH_FULL_FRAMES,
        SKETCH_PITCH_CHILD: SKETCH_PITCH_BINS * SKETCH_FULL_FRAMES,
    }
    actual_sizes = {}
    try:
        for child_name in expected_sizes:
            child_type = field.type.field(child_name).type
            if pa.types.is_fixed_size_list(child_type):
                actual_sizes[child_name] = child_type.list_size
            else:
                tensor_shape = getattr(child_type, "shape", None)
                actual_sizes[child_name] = (
                    math.prod(tensor_shape) if tensor_shape is not None else None
                )
    except KeyError as exc:
        raise ValueError(f"{field_name!r} is missing sketch child {exc.args[0]!r}") from exc
    if actual_sizes != expected_sizes:
        raise ValueError(
            f"{field_name!r} does not match the historical 401-frame schema: "
            f"got {actual_sizes}, expected {expected_sizes}"
        )


def _prepare_source(
    dataset: lance.LanceDataset,
    config: SketchPoolBackfillConfig,
    lance_uri: str,
    storage_options: dict[str, str] | None,
) -> lance.LanceDataset:
    """Validate, tag, and if necessary rename the migration source.

    :param dataset: Current branch snapshot.
    :param config: Rollback tag and branch configuration.
    :param lance_uri: Lance-openable dataset target.
    :param storage_options: Object-store credentials, when required.
    :returns: Snapshot carrying the renamed full-resolution source.
    :raises ValueError: Source state or rollback tag is incompatible.
    """

    from synth_setter.pipeline.data.add_embeddings import SKETCH_FULL_STRUCT_FIELD

    names = set(dataset.schema.names)
    has_canonical = SKETCH_STRUCT_FIELD in names
    has_full = SKETCH_FULL_STRUCT_FIELD in names
    if has_canonical and has_full:
        raise ValueError(f"existing {SKETCH_STRUCT_FIELD!r} field has incompatible metadata")
    if not has_canonical and not has_full:
        raise ValueError(
            f"dataset has neither {SKETCH_STRUCT_FIELD!r} nor {SKETCH_FULL_STRUCT_FIELD!r} source"
        )
    if _retry("count_prepared_rows", dataset.count_rows) == 0:
        raise ValueError("cannot backfill an empty dataset")
    _validate_full_source_field(
        dataset,
        SKETCH_STRUCT_FIELD if has_canonical else SKETCH_FULL_STRUCT_FIELD,
    )
    if has_canonical:
        _create_rollback_tag(dataset, config)
    _validate_rollback_tag(
        dataset,
        config,
        lance_uri,
        storage_options,
        expected_version=dataset.version if has_canonical else dataset.version - 1,
    )
    if not has_canonical:
        return dataset
    alteration = _ColumnRename(
        path=SKETCH_STRUCT_FIELD,
        name=SKETCH_FULL_STRUCT_FIELD,
    )

    def rename_or_recover() -> lance.LanceDataset:
        latest = _open_dataset(lance_uri, storage_options, config.branch, None)
        latest_names = set(latest.schema.names)
        if SKETCH_FULL_STRUCT_FIELD in latest_names and SKETCH_STRUCT_FIELD not in latest_names:
            return latest
        if SKETCH_STRUCT_FIELD not in latest_names or SKETCH_FULL_STRUCT_FIELD in latest_names:
            raise ValueError("branch advanced to an incompatible schema during source rename")
        cast("_AlterColumnsDataset", latest).alter_columns(alteration)
        return _open_dataset(lance_uri, storage_options, config.branch, None)

    return _retry("rename_sketch_source", rename_or_recover)


def backfill_sketch_pool(config: SketchPoolBackfillConfig) -> SketchPoolBackfillResult:
    """Pool every fragment in parallel and publish one branch-scoped Lance commit.

    :param config: Strict migration configuration.
    :returns: Committed or already-complete migration summary.
    """
    import ray

    started = time.monotonic()
    run_id = str(uuid.uuid4())
    ray.init(num_cpus=config.workers, include_dashboard=False, log_to_driver=False)
    try:
        from synth_setter.pipeline.data.add_embeddings import _sketch_pool_artifact_identity

        lance_uri, storage_options = _lance_target(config.lance_uri)
        dataset = _open_dataset(lance_uri, storage_options, config.branch, None)
        artifact = _sketch_pool_artifact_identity("")
        if _is_complete(dataset, artifact.encode()):
            _validate_rollback_tag(
                dataset,
                config,
                lance_uri,
                storage_options,
                expected_version=None,
            )
            source_version = dataset.version
            dataset, index_built = _ensure_canonical_index(dataset, config)
            return _write_result(
                _result(
                    config,
                    dataset,
                    source_version,
                    started,
                    True,
                    index_built,
                    run_id,
                ),
                config.result,
            )
        dataset = _prepare_source(dataset, config, lance_uri, storage_options)
        source_version = dataset.version
        reports = _run_fragment_tasks(
            dataset,
            config,
            lance_uri,
            storage_options,
            source_version,
            artifact,
            started,
        )
        committed = _commit_reports(
            dataset,
            reports,
            config,
            lance_uri,
            storage_options,
            source_version,
            artifact.encode(),
        )
        _clear_resume_cache(config)
        committed, index_built = _ensure_canonical_index(committed, config)
        return _write_result(
            _result(
                config,
                committed,
                source_version,
                started,
                False,
                index_built,
                run_id,
            ),
            config.result,
        )
    finally:
        ray.shutdown()


def _parse_args() -> SketchPoolBackfillConfig:
    """Parse the distributed migration CLI.

    :returns: Strict backfill configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--tasks-per-worker", default=4, type=int)
    parser.add_argument("--rollback-tag", required=True)
    parser.add_argument("--build-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-partitions", type=int)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--resume-uri")
    parser.add_argument("--timeout-seconds", type=float, default=21_600.0)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    return SketchPoolBackfillConfig.model_validate(vars(args), strict=True)


def main() -> None:
    """Execute the migration configured by process arguments."""
    backfill_sketch_pool(_parse_args())


if __name__ == "__main__":
    main()
