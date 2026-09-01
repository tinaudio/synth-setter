"""Branch-safe distributed embedding backfill and candidate promotion."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = structlog.get_logger(__name__)

_WORKER_ENCODERS: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class WorkerMetrics:
    """Capture one fragment task's resource and throughput measurements.

    .. attribute :: pid

        Worker process identifier.

    .. attribute :: rows

        Fragment rows transformed.

    .. attribute :: elapsed_seconds

        Fragment task wall time.

    .. attribute :: peak_rss_bytes

        Process peak resident memory.

    .. attribute :: peak_gpu_allocated_bytes

        PyTorch peak allocated GPU memory.

    .. attribute :: peak_gpu_reserved_bytes

        PyTorch peak reserved GPU memory.
    """

    pid: int
    rows: int
    elapsed_seconds: float
    peak_rss_bytes: int
    peak_gpu_allocated_bytes: int
    peak_gpu_reserved_bytes: int


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


def _storage_options(uri: str) -> dict[str, str] | None:
    """Resolve credentials only for object-store datasets.

    :param uri: Lance dataset URI or local path.
    :returns: R2 Lance options for ``s3://`` URIs, otherwise ``None``.
    """
    if not uri.startswith("s3://"):
        return None
    from synth_setter.pipeline.r2_io import r2_storage_options

    return r2_storage_options()


def _worker_encoder(embedding: str, checkpoint: str) -> Any:
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
    encoder = EMBEDDING_REGISTRY[embedding].load_encoder(checkpoint, config)
    _WORKER_ENCODERS[key] = encoder
    return encoder


def _transform_fragment(
    uri: str,
    storage_options: dict[str, str] | None,
    branch: str,
    source_version: int,
    fragment_id: int,
    embedding: str,
    checkpoint: str,
    sample_rate: int,
    batch_size: int,
    artifact: bytes,
) -> tuple[bytes, bytes, int, WorkerMetrics]:
    """Write one fragment's embedding data without committing a manifest.

    :param uri: Lance dataset URI or local path.
    :param storage_options: Object-store credentials when required.
    :param branch: Source branch name.
    :param source_version: Immutable source version on the branch.
    :param fragment_id: Fragment to transform.
    :param embedding: Supported embedding registry key.
    :param checkpoint: Registry checkpoint source.
    :param sample_rate: Source audio sample rate in Hz.
    :param batch_size: Rows decoded per callback.
    :param artifact: Versioned artifact identity stored in field metadata.
    :returns: Pickled fragment metadata/schema, row count, and resource metrics.
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

    started = time.monotonic()
    dataset = lance.dataset(uri, storage_options=storage_options).checkout_version(
        (None if branch == "main" else branch, source_version)
    )
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        raise ValueError(f"missing fragment {fragment_id}")
    spec = EMBEDDING_REGISTRY[embedding]
    encoder = _worker_encoder(embedding, checkpoint)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
        encoded = _encode_columns(
            _decoded_sources(batch, spec.input_fields),
            sample_rate,
            [spec],
            [encoder],
        )
        metadata = {
            b"synth_setter.embedding.name": embedding.encode(),
            b"synth_setter.embedding.artifact": artifact,
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
        batch_size=batch_size,
    )
    rows = fragment.count_rows()
    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    metrics = WorkerMetrics(
        pid=os.getpid(),
        rows=rows,
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_gpu_allocated_bytes=peak_allocated,
        peak_gpu_reserved_bytes=peak_reserved,
    )
    return pickle.dumps(metadata), pickle.dumps(schema), rows, metrics


def _write_result(result: Any, destination: Path | None) -> Any:
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


def _embedding_is_complete(dataset: Any, spec: Any, artifact: bytes) -> bool:
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


def _ensure_embedding_index(dataset: Any, spec: Any, config: Any) -> tuple[Any, bool]:
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


def backfill_embedding(config: EmbeddingBackfillConfig) -> EmbeddingBackfillResult:
    """Write one embedding through recycled Ray workers and one branch merge commit.

    :param config: Strict branch, model, worker, and index configuration.
    :returns: Distributed write result with measured resource peaks.
    :raises RuntimeError: CUDA is unavailable or Ray exceeds the process reuse bound.
    :raises ValueError: Source, artifact, or worker schemas are incompatible.
    """
    import ray
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("distributed embedding backfill requires CUDA")
    started = time.monotonic()
    ray.init(
        num_cpus=config.workers,
        num_gpus=1,
        include_dashboard=False,
        log_to_driver=False,
    )
    try:
        import lance

        from synth_setter.pipeline.data.add_embeddings import (
            EMBEDDING_REGISTRY,
            _output_columns,
            _resolve_artifact_identity,
        )
        from synth_setter.pipeline.data.lance_shard import read_shard_metadata
        from synth_setter.pipeline.schemas.add_embeddings_config import (
            AddEmbeddingsConfig,
        )

        storage_options = _storage_options(config.lance_uri)
        dataset = lance.dataset(
            config.lance_uri, storage_options=storage_options
        ).checkout_version((config.branch, None))
        spec = EMBEDDING_REGISTRY[config.embedding]
        checkpoint = config.checkpoint or spec.default_checkpoint
        checkpoints = {config.embedding: checkpoint}
        add_config = AddEmbeddingsConfig(
            lance_uri=config.lance_uri,
            embeddings=(config.embedding,),
            checkpoints=checkpoints,
            device="cuda",
            batch_size=config.batch_size,
            build_index=config.build_index,
            num_partitions=config.num_partitions,
        )
        artifact = _resolve_artifact_identity(spec, add_config).encode()
        if _embedding_is_complete(dataset, spec, artifact):
            source_version = dataset.version
            dataset, index_built = _ensure_embedding_index(dataset, spec, add_config)
            return _write_result(
                EmbeddingBackfillResult(
                    branch=config.branch,
                    embedding=config.embedding,
                    rows=dataset.count_rows(),
                    fragments=len(dataset.get_fragments()),
                    source_version=source_version,
                    data_version=source_version,
                    final_version=dataset.version,
                    elapsed_seconds=time.monotonic() - started,
                    rows_per_second=0.0,
                    worker_processes=0,
                    max_tasks_per_process=0,
                    peak_rss_bytes=0,
                    peak_gpu_allocated_bytes=0,
                    peak_gpu_reserved_bytes=0,
                    already_complete=True,
                    index_built=index_built,
                ),
                config.result,
            )
        if any(column in dataset.schema.names for column in _output_columns(spec)):
            raise ValueError(f"candidate has incompatible {config.embedding} output columns")
        total_rows = dataset.count_rows()
        if total_rows < 1:
            raise ValueError("cannot backfill an empty dataset")
        sample_rate = int(read_shard_metadata(dataset.schema).sample_rate)
        source_version = dataset.version
        fragments = dataset.get_fragments()
        transform = ray.remote(
            num_cpus=1,
            num_gpus=config.gpu_per_worker,
            max_calls=config.tasks_per_worker,
        )(_transform_fragment)
        pending = [
            transform.remote(
                config.lance_uri,
                storage_options,
                config.branch,
                source_version,
                fragment.metadata.id,
                config.embedding,
                checkpoint,
                sample_rate,
                config.batch_size,
                artifact,
            )
            for fragment in fragments
        ]
        results: list[tuple[bytes, bytes, int, WorkerMetrics]] = []
        rows_done = 0
        last_log = started
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=10)
            if ready:
                result = ray.get(ready[0])
                results.append(result)
                rows_done += result[2]
            now = time.monotonic()
            if now - last_log >= 30 or not pending:
                elapsed = now - started
                rate = rows_done / elapsed
                logger.info(
                    "embedding_backfill_progress",
                    embedding=config.embedding,
                    branch=config.branch,
                    rows=rows_done,
                    total_rows=total_rows,
                    fragments=len(results),
                    total_fragments=len(fragments),
                    rows_per_second=rate,
                    elapsed_seconds=elapsed,
                    eta_seconds=(total_rows - rows_done) / rate if rate > 0 else None,
                )
                last_log = now

        metadata = [pickle.loads(result[0]) for result in results]  # noqa: S301
        schemas = [pickle.loads(result[1]) for result in results]  # noqa: S301
        if not schemas or any(schema != schemas[0] for schema in schemas[1:]):
            raise ValueError("worker schemas differ")
        metrics = [result[3] for result in results]
        tasks_by_pid = Counter(metric.pid for metric in metrics)
        max_tasks = max(tasks_by_pid.values(), default=0)
        if max_tasks > config.tasks_per_worker:
            raise RuntimeError(
                f"Ray worker served {max_tasks} tasks, exceeding bound {config.tasks_per_worker}"
            )
        operation = lance.LanceOperation.Merge(metadata, schemas[0])
        committed = lance.LanceDataset.commit(
            dataset,
            operation,
            read_version=source_version,
            storage_options=storage_options,
            commit_message=f"Add {config.embedding} embeddings",
        )
        data_version = committed.version
        committed, index_built = _ensure_embedding_index(committed, spec, add_config)
        elapsed = time.monotonic() - started
        return _write_result(
            EmbeddingBackfillResult(
                branch=config.branch,
                embedding=config.embedding,
                rows=committed.count_rows(),
                fragments=len(committed.get_fragments()),
                source_version=source_version,
                data_version=data_version,
                final_version=committed.version,
                elapsed_seconds=elapsed,
                rows_per_second=total_rows / elapsed,
                worker_processes=len(tasks_by_pid),
                max_tasks_per_process=max_tasks,
                peak_rss_bytes=max((metric.peak_rss_bytes for metric in metrics), default=0),
                peak_gpu_allocated_bytes=max(
                    (metric.peak_gpu_allocated_bytes for metric in metrics), default=0
                ),
                peak_gpu_reserved_bytes=max(
                    (metric.peak_gpu_reserved_bytes for metric in metrics), default=0
                ),
                already_complete=False,
                index_built=index_built,
            ),
            config.result,
        )
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
    source_path = Path(uri) / "tree" / branch / "data"
    destination_path = Path(uri) / "data"
    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)


def _main_merge_operation(candidate: Any, branch: str, source_version: int) -> Any:
    """Materialize candidate files under main and rebuild its latest merge operation.

    :param candidate: Validated candidate branch dataset.
    :param branch: Candidate branch name.
    :param source_version: Candidate parent version on main.
    :returns: Lance merge operation whose data paths are main-relative.
    :raises ValueError: No merge transaction exists or a file uses an unexpected base.
    """
    import lance

    operation = None
    for version in range(candidate.version, source_version, -1):
        transaction = candidate.read_transaction(version)
        if isinstance(transaction.operation, lance.LanceOperation.Merge):
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
    _copy_candidate_data_directory(candidate.uri, branch)
    return lance.LanceOperation.Merge(fragments, candidate.schema)


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
    present_main = set(config.columns) & set(main.schema.names)
    if present_main:
        if present_main != set(config.columns):
            raise ValueError(f"main has partial candidate columns {sorted(present_main)}")
        if any(main.schema.field(name) != candidate.schema.field(name) for name in config.columns):
            raise ValueError("main candidate column schema differs from the validated branch")
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
