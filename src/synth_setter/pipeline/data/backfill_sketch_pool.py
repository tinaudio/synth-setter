"""Distributed migration from stored full-resolution sketches to canonical pooling."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import structlog
from pydantic import BaseModel, ConfigDict, Field

from synth_setter.data.vst.shapes import SKETCH_STRUCT_FIELD

logger = structlog.get_logger(__name__)


class SketchPoolBackfillConfig(BaseModel):
    """Validate one distributed stored-sketch migration.

    .. attribute :: model_config

        Pydantic model config sentinel.

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

        Optional immutable tag created before the rename commit.

    .. attribute :: build_index

        Whether to build the canonical sketch vector index after pooling.

    .. attribute :: num_partitions

        IVF partition override, or ``None`` for a row-derived count.

    .. attribute :: result

        Optional JSON result path.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(min_length=1)
    branch: str = Field(default="main", min_length=1)
    workers: int = Field(ge=1)
    batch_size: int = Field(default=128, ge=1)
    tasks_per_worker: int = Field(default=4, ge=1)
    rollback_tag: str | None = None
    build_index: bool = True
    num_partitions: int | None = Field(default=None, ge=1)
    result: Path | None = None


@dataclass(frozen=True)
class SketchPoolBackfillResult:
    """Summarize a committed or already-complete migration.

    .. attribute :: branch

        Mutated Lance branch.

    .. attribute :: rows

        Rows preserved by the migration.

    .. attribute :: fragments

        Fragments represented by the committed snapshot.

    .. attribute :: source_version

        Version read by the operation.

    .. attribute :: committed_version

        Latest version after data and index publication.

    .. attribute :: elapsed_seconds

        Total wall-clock duration.

    .. attribute :: rows_per_second

        End-to-end row throughput, or zero when data was already complete.

    .. attribute :: already_complete

        Whether the pooled data existed before this invocation.

    .. attribute :: index_built

        Whether this invocation published the canonical index.
    """

    branch: str
    rows: int
    fragments: int
    source_version: int
    committed_version: int
    elapsed_seconds: float
    rows_per_second: float
    already_complete: bool
    index_built: bool


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


def _transform_fragment(
    uri: str,
    storage_options: dict[str, str] | None,
    branch: str,
    source_version: int,
    fragment_id: int,
    batch_size: int,
    artifact: bytes,
) -> tuple[bytes, bytes, int]:
    """Write one fragment's pooled sketch column without committing a manifest.

    :param uri: Lance dataset URI or local path.
    :param storage_options: Object-store credentials, when required.
    :param branch: Source branch name.
    :param source_version: Immutable source version on the branch.
    :param fragment_id: Fragment to transform.
    :param batch_size: Rows decoded per callback.
    :param artifact: Versioned pooling-policy identity.
    :returns: Pickled fragment metadata, output schema, and transformed row count.
    :raises ValueError: The source fragment cannot be found.
    """
    import lance
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import (
        SKETCH_FULL_STRUCT_FIELD,
        _encode_sketch_pool_column,
    )

    dataset = lance.dataset(uri, storage_options=storage_options).checkout_version(
        _branch_reference(branch, source_version)
    )
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        raise ValueError(f"missing fragment {fragment_id}")

    def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
        rows = batch.column(SKETCH_FULL_STRUCT_FIELD).to_numpy(zero_copy_only=False)
        encoded = _encode_sketch_pool_column(
            {SKETCH_FULL_STRUCT_FIELD: rows}, 0, lambda values: values
        )
        field = pa.field(
            SKETCH_STRUCT_FIELD,
            encoded.type,
            metadata={
                b"synth_setter.embedding.name": b"sketch_pool",
                b"synth_setter.embedding.artifact": artifact,
            },
        )
        return pa.RecordBatch.from_arrays([encoded], schema=pa.schema([field]))

    metadata, schema = fragment.merge_columns(
        transform,
        [SKETCH_FULL_STRUCT_FIELD],
        batch_size=batch_size,
    )
    return pickle.dumps(metadata), pickle.dumps(schema), fragment.count_rows()


def _is_complete(dataset: Any, artifact: bytes) -> bool:
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


def _ensure_canonical_index(
    dataset: Any, config: SketchPoolBackfillConfig
) -> tuple[Any, bool]:
    """Build the pooled-vector index when requested and absent.

    :param dataset: Dataset carrying canonical pooled sketches.
    :param config: Index build selection and partition override.
    :returns: Latest dataset snapshot and whether this call built an index.
    :raises ValueError: Existing canonical indexes are ambiguous or incompatible.
    :raises RuntimeError: The registry policy lacks its required index specification.
    """
    from synth_setter.pipeline.data.add_embeddings import (
        EMBEDDING_REGISTRY,
        MIN_ROWS_FOR_INDEX,
        SKETCH_VEC_COLUMN,
    )

    if not config.build_index or dataset.count_rows() < MIN_ROWS_FOR_INDEX:
        return dataset, False
    existing = [
        candidate
        for candidate in dataset.list_indices()
        if candidate["fields"] == [SKETCH_VEC_COLUMN]
    ]
    if len(existing) > 1:
        raise ValueError(f"multiple indexes target {SKETCH_VEC_COLUMN!r}")
    if existing:
        if existing[0]["type"] != "IVF_PQ":
            raise ValueError(
                f"existing {SKETCH_VEC_COLUMN!r} index is {existing[0]['type']}, not IVF_PQ"
            )
        return dataset, False
    partitions = config.num_partitions or max(1, round(dataset.count_rows() ** 0.5))
    index = EMBEDDING_REGISTRY["sketch_pool"].index
    if index is None:
        raise RuntimeError("sketch_pool registry policy has no index specification")
    index_started = time.monotonic()

    def log_progress(progress: Any) -> None:
        elapsed = time.monotonic() - index_started
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
            event=progress.event,
            stage=progress.stage,
            completed=progress.completed,
            total=progress.total,
            unit=progress.unit,
            rate=rate,
            elapsed_seconds=elapsed,
            eta_seconds=remaining,
        )

    indexed = dataset.create_index(
        SKETCH_VEC_COLUMN,
        index_type="IVF_PQ",
        name="sketch_pool_vec_idx",
        metric=index.metric,
        num_partitions=partitions,
        num_sub_vectors=index.num_sub_vectors,
        progress_callback=log_progress,
    )
    return indexed, True


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


def backfill_sketch_pool(config: SketchPoolBackfillConfig) -> SketchPoolBackfillResult:
    """Pool every fragment in parallel and publish one branch-scoped Lance commit.

    :param config: Strict migration configuration.
    :returns: Committed or already-complete migration summary.
    :raises ValueError: Source columns, rollback tag, or worker schemas are incompatible.
    """
    import ray

    started = time.monotonic()
    ray.init(num_cpus=config.workers, include_dashboard=False, log_to_driver=False)
    try:
        import lance

        from synth_setter.pipeline.data.add_embeddings import (
            SKETCH_FULL_STRUCT_FIELD,
            _sketch_pool_artifact_identity,
        )

        lance_uri, storage_options = _lance_target(config.lance_uri)
        dataset = lance.dataset(
            lance_uri, storage_options=storage_options
        ).checkout_version(_branch_reference(config.branch, None))
        artifact = _sketch_pool_artifact_identity("").encode()
        if _is_complete(dataset, artifact):
            source_version = dataset.version
            dataset, index_built = _ensure_canonical_index(dataset, config)
            return _write_result(
                SketchPoolBackfillResult(
                    branch=config.branch,
                    rows=dataset.count_rows(),
                    fragments=len(dataset.get_fragments()),
                    source_version=source_version,
                    committed_version=dataset.version,
                    elapsed_seconds=time.monotonic() - started,
                    rows_per_second=0.0,
                    already_complete=True,
                    index_built=index_built,
                ),
                config.result,
            )

        names = set(dataset.schema.names)
        has_canonical = SKETCH_STRUCT_FIELD in names
        has_full = SKETCH_FULL_STRUCT_FIELD in names
        if has_canonical and has_full:
            raise ValueError(
                f"existing {SKETCH_STRUCT_FIELD!r} field has incompatible pooling metadata"
            )
        if not has_canonical and not has_full:
            raise ValueError(
                f"dataset has neither {SKETCH_STRUCT_FIELD!r} nor "
                f"{SKETCH_FULL_STRUCT_FIELD!r} source column"
            )
        if dataset.count_rows() == 0:
            raise ValueError("cannot backfill an empty dataset")

        if config.rollback_tag is not None:
            existing = dataset.tags.list().get(config.rollback_tag)
            allowed_branches = (
                {None, "main"} if config.branch == "main" else {config.branch}
            )
            rollback_version = dataset.version if has_canonical else dataset.version - 1
            if existing is None:
                if not has_canonical:
                    raise ValueError(
                        f"rollback tag {config.rollback_tag!r} must exist before resuming "
                        "a renamed source"
                    )
                dataset.tags.create(
                    config.rollback_tag,
                    _branch_reference(config.branch, dataset.version),
                )
            elif (
                existing["branch"] not in allowed_branches
                or existing["version"] != rollback_version
            ):
                raise ValueError(
                    f"rollback tag {config.rollback_tag!r} does not identify branch "
                    f"{config.branch!r} version {rollback_version}"
                )

        if has_canonical:
            dataset.alter_columns(
                cast(
                    "Any",
                    {"path": SKETCH_STRUCT_FIELD, "name": SKETCH_FULL_STRUCT_FIELD},
                )
            )
            dataset = lance.dataset(
                lance_uri, storage_options=storage_options
            ).checkout_version(_branch_reference(config.branch, None))

        source_version = dataset.version
        fragments = dataset.get_fragments()
        total_rows = dataset.count_rows()
        remote_transform = ray.remote(
            num_cpus=1, max_calls=config.tasks_per_worker
        )(_transform_fragment)
        pending = [
            remote_transform.remote(
                lance_uri,
                storage_options,
                config.branch,
                source_version,
                fragment.metadata.id,
                config.batch_size,
                artifact,
            )
            for fragment in fragments
        ]
        results: list[tuple[bytes, bytes, int]] = []
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
                rows_per_second = rows_done / elapsed
                logger.info(
                    "sketch_pool_backfill_progress",
                    rows=rows_done,
                    total_rows=total_rows,
                    rows_per_second=rows_per_second,
                    fragments=len(results),
                    total_fragments=len(fragments),
                    elapsed_seconds=elapsed,
                    eta_seconds=(total_rows - rows_done) / rows_per_second
                    if rows_per_second > 0
                    else None,
                )
                last_log = now

        metadata = [pickle.loads(result[0]) for result in results]  # noqa: S301
        schemas = [pickle.loads(result[1]) for result in results]  # noqa: S301
        if not schemas or any(schema != schemas[0] for schema in schemas[1:]):
            raise ValueError("worker schemas differ")
        operation = lance.LanceOperation.Merge(metadata, schemas[0])
        committed = lance.LanceDataset.commit(
            dataset,
            operation,
            read_version=source_version,
            storage_options=storage_options,
            commit_message="Pool stored full-resolution sketch controls",
        )
        committed, index_built = _ensure_canonical_index(committed, config)
        elapsed = time.monotonic() - started
        return _write_result(
            SketchPoolBackfillResult(
                branch=config.branch,
                rows=committed.count_rows(),
                fragments=len(committed.get_fragments()),
                source_version=source_version,
                committed_version=committed.version,
                elapsed_seconds=elapsed,
                rows_per_second=total_rows / elapsed,
                already_complete=False,
                index_built=index_built,
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
    parser.add_argument("--rollback-tag")
    parser.add_argument(
        "--build-index", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-partitions", type=int)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    return SketchPoolBackfillConfig.model_validate(vars(args), strict=True)


def main() -> None:
    """Run the distributed migration CLI."""
    backfill_sketch_pool(_parse_args())


if __name__ == "__main__":
    main()
