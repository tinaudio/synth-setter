"""Worker-side staging of Lance shard attempts (#1776).

A worker stages one shard attempt by writing its uncommitted fragment data
straight into the assigned split's dataset directory (a fragment is only
readable from the dataset whose ``data/`` dir physically holds its file), then
uploading the reconciliation contract to the shard's staging directory:
``{worker}-{attempt}.fragment.json`` (sidecar), ``.shard-stats.npz`` (Welford
state), and ``.valid`` strictly last as the staged-attempt commit point.
Finalize later selects one winner per shard and commits the winners' fragment
metadata into the split manifests — no row rewrite (design doc §7.2/§7.6).

Typical worker use calls ``write_rendering_marker(...)`` before rendering and
``stage_lance_shard_attempt(...)`` after local shard validation succeeds.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import (
    ATTEMPT_INVALID_SUFFIX,
    ATTEMPT_RENDERING_SUFFIX,
    ATTEMPT_VALID_SUFFIX,
    LANCE_FRAGMENT_SIDECAR_SUFFIX,
    LANCE_SHARD_STATS_KEYS,
    LANCE_SHARD_STATS_SUFFIX,
)
from synth_setter.pipeline.schemas.lance_attempt import LanceFragmentSidecar

if TYPE_CHECKING:
    from synth_setter.pipeline.data.add_embeddings import GenerationEmbeddingRuntime
    from synth_setter.pipeline.schemas.spec import DatasetSpec, ShardSpec, Split

# Suffixes that must all exist for an attempt to be staged-valid (design §7.2).
COMPLETE_ATTEMPT_SUFFIXES: tuple[str, ...] = (
    LANCE_FRAGMENT_SIDECAR_SUFFIX,
    LANCE_SHARD_STATS_SUFFIX,
    ATTEMPT_VALID_SUFFIX,
)

_GENERATION_RUNTIME_GUARD = threading.Lock()
_generation_runtime: GenerationEmbeddingRuntime | None = None
_generation_runtime_policy: object | None = None


def _reset_generation_embedding_runtime() -> None:
    """Release the process-local generation encoder runtime, primarily at worker shutdown."""
    with _GENERATION_RUNTIME_GUARD:
        _reset_generation_embedding_runtime_unlocked()


def _runtime_for_spec(spec: DatasetSpec) -> GenerationEmbeddingRuntime | None:
    """Return the one process-local encoder runtime shared by staging threads.

    :param spec: Frozen dataset spec carrying the optional generation policy.
    :returns: Shared runtime, or ``None`` when shard embeddings are disabled.
    """
    global _generation_runtime, _generation_runtime_policy
    policy = spec.embedding_generation
    if policy is None:
        return None
    runtime_policy = (policy, spec.render.param_spec_name)
    with _GENERATION_RUNTIME_GUARD:
        if _generation_runtime is not None and _generation_runtime_policy == runtime_policy:
            return _generation_runtime
        _reset_generation_embedding_runtime_unlocked()
        from synth_setter.pipeline.data.add_embeddings import GenerationEmbeddingRuntime

        _generation_runtime = GenerationEmbeddingRuntime(
            policy, param_spec_name=str(spec.render.param_spec_name)
        )
        _generation_runtime_policy = runtime_policy
        return _generation_runtime


def _reset_generation_embedding_runtime_unlocked() -> None:
    """Reset the runtime while the caller holds ``_GENERATION_RUNTIME_GUARD``."""
    global _generation_runtime, _generation_runtime_policy
    closer = getattr(_generation_runtime, "close", None)
    if callable(closer):
        closer()
    _generation_runtime = None
    _generation_runtime_policy = None


def split_for_shard(spec: DatasetSpec, shard_id: int) -> Split:
    """Return the split the spec deterministically assigns to ``shard_id``.

    :param spec: Validated dataset spec.
    :param shard_id: Logical shard id.
    :returns: The split whose ``split_shard_ranges`` half-open range holds the shard.
    :raises ValueError: ``shard_id`` falls outside every split range.
    """
    for split, (lo, hi) in spec.split_shard_ranges.items():
        if lo <= shard_id < hi:
            return split
    raise ValueError(f"shard_id {shard_id} outside spec ranges {spec.split_shard_ranges!r}")


def _upload_empty_marker(marker_uri: str) -> None:
    """Upload a zero-byte lifecycle marker; presence is the state.

    :param marker_uri: Destination ``r2://`` URI of the marker object.
    """
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / Path(marker_uri).name
        marker.touch()
        r2_io.upload(marker, marker_uri)


def write_rendering_marker(
    spec: DatasetSpec,
    shard_id: int,
    *,
    worker_id: str,
    attempt_uuid: str,
    attempt_staging_dir_uri: str | None = None,
) -> None:
    """Record the start of a shard attempt (``.rendering``, append-only).

    Deliberately unconsumed by reconciliation — an orphaned ``.rendering``
    with no sibling ``.valid`` is operator evidence of a crashed attempt
    (design §7.2), read by humans via ``rclone ls``, not by code.

    :param spec: Validated dataset spec.
    :param shard_id: Logical shard the attempt renders.
    :param worker_id: Worker identifier for the staging filename.
    :param attempt_uuid: Per-attempt UUID for the staging filename.
    :param attempt_staging_dir_uri: Optional feature-specific attempt directory.
    """
    marker_uri = (
        f"{attempt_staging_dir_uri}{worker_id}-{attempt_uuid}{ATTEMPT_RENDERING_SUFFIX}"
        if attempt_staging_dir_uri is not None
        else spec.r2.worker_staged_shard_uri(
            shard_id, worker_id, attempt_uuid, ATTEMPT_RENDERING_SUFFIX
        )
    )
    _upload_empty_marker(marker_uri)


def stage_lance_shard_attempt(
    spec: DatasetSpec,
    shard: ShardSpec,
    local_shard_path: Path,
    *,
    worker_id: str,
    attempt_uuid: str,
    target_lance_uri: str | None = None,
    attempt_staging_dir_uri: str | None = None,
) -> None:
    """Stage one rendered shard as an uncommitted fragment attempt.

    Streams the local shard's batches into a single fragment under the
    assigned split's dataset directory, folds its mel rows into Welford state,
    then uploads sidecar + stats and the ``.valid`` marker strictly last — an
    interrupted staging never presents as a complete attempt.

    :param spec: Validated dataset spec.
    :param shard: Shard the local dataset renders.
    :param local_shard_path: Local ``shard-NNNNNN.lance`` dataset directory.
    :param worker_id: Worker identifier for the staging filenames.
    :param attempt_uuid: Per-attempt UUID for the staging filenames.
    :param target_lance_uri: Growing branch URI receiving fragment data; ``None``
        uses the baseline split assigned by the spec.
    :param attempt_staging_dir_uri: Branch-specific sidecar directory; ``None``
        uses the finalized baseline staging namespace.
    :raises ValueError: The local shard's row count does not match the spec, or
        the shard's bytes would exceed the single-data-file bound
        (``LANCE_MAX_BYTES_PER_FILE``) the fragment write cannot split.
    """
    # Function-local so importing this module (e.g. from the launcher) never
    # pays the `lance` import cost.
    import lance
    import pyarrow as pa

    from synth_setter.data.vst.shapes import dataset_field_dtypes, dataset_field_shapes
    from synth_setter.pipeline.data.lance_shard import (
        lance_fragment,
        lance_schema,
    )
    from synth_setter.pipeline.data.stats import fold_lance_shard_into_welford

    dataset = lance.dataset(str(local_shard_path))
    rows = dataset.count_rows()
    if rows != spec.render.samples_per_shard:
        raise ValueError(
            f"local shard {local_shard_path.name} has {rows} rows; "
            f"spec expects {spec.render.samples_per_shard} per shard"
        )
    render = spec.render_for_shard(shard)
    expected_schema = lance_schema(
        dataset_field_shapes(render, spec.num_params),
        render.shard_metadata(),
        field_dtypes=dataset_field_dtypes(render),
    )
    if not dataset.schema.equals(expected_schema, check_metadata=True):
        raise ValueError(
            f"local shard {local_shard_path.name} schema does not match spec-derived schema"
        )
    growing_target = target_lance_uri is not None
    if target_lance_uri is None:
        split = split_for_shard(spec, shard.shard_id)
        target_lance_uri = spec.r2.split_lance_uri(split)
    if r2_io.is_r2_uri(target_lance_uri):
        split_target, storage_options = r2_io.lance_target(target_lance_uri)
    else:
        split_target = target_lance_uri
        storage_options = (
            r2_io.r2_storage_options() if target_lance_uri.startswith("s3://") else None
        )
    runtime = _runtime_for_spec(spec)
    output_columns = () if runtime is None else runtime.output_columns
    fragment_schema = dataset.schema
    if growing_target:
        from synth_setter.pipeline.data.lance_materialize import retry_lance_read

        branch_schema = retry_lance_read(
            "growing_branch_schema_read",
            lambda: lance.dataset(split_target, storage_options=storage_options).schema,
        )
        expected_names = [*dataset.schema.names, *output_columns]
        branch_base = pa.schema(
            [branch_schema.field(name) for name in dataset.schema.names],
            metadata=branch_schema.metadata,
        )
        if branch_schema.names != expected_names or not dataset.schema.equals(
            branch_base, check_metadata=False
        ):
            raise ValueError(
                f"growing shard fields {expected_names} do not match branch schema "
                f"{branch_schema.names}"
            )
        fragment_schema = branch_schema
    batches = dataset.to_batches(
        batch_size=None if runtime is None else runtime.batch_size
    )
    if runtime is not None:
        batches = (
            runtime.augment_batch(batch, spec.render.sample_rate) for batch in batches
        )
        first_batch = next(batches)
        if growing_target:
            for name in output_columns:
                generated = first_batch.schema.field(name)
                branch = fragment_schema.field(name)
                if generated.type != branch.type or generated.metadata != branch.metadata:
                    raise ValueError(
                        "generated embedding types or provenance do not match the growing branch"
                    )
        else:
            fragment_schema = first_batch.schema
        batches = chain((first_batch,), batches)
        if growing_target:
            batches = (
                pa.RecordBatch.from_arrays(batch.columns, schema=fragment_schema)
                for batch in batches
            )
    fragment = lance_fragment(
        split_target,
        fragment_schema,
        batches,
        storage_options=storage_options,
    )
    count, mean, m2 = fold_lance_shard_into_welford((0, 0, 0), local_shard_path)

    def _attempt_uri(suffix: str) -> str:
        if attempt_staging_dir_uri is not None:
            return f"{attempt_staging_dir_uri}{worker_id}-{attempt_uuid}{suffix}"
        return spec.r2.worker_staged_shard_uri(
            shard.shard_id, worker_id, attempt_uuid, suffix
        )

    welford_arrays = dict(zip(LANCE_SHARD_STATS_KEYS, (np.int64(count), mean, m2), strict=True))
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "shard-stats.npz"
        np.savez(stats_path, **welford_arrays)
        r2_io.upload(stats_path, _attempt_uri(LANCE_SHARD_STATS_SUFFIX))
        sidecar_path = Path(tmp) / "fragment.json"
        sidecar = LanceFragmentSidecar(
            schema_version=1, fragment_json=json.dumps(fragment.to_json())
        )
        sidecar_path.write_text(sidecar.model_dump_json())
        r2_io.upload(sidecar_path, _attempt_uri(LANCE_FRAGMENT_SIDECAR_SUFFIX))
    _upload_empty_marker(_attempt_uri(ATTEMPT_VALID_SUFFIX))


def complete_attempt_names(entry_paths: Sequence[str]) -> list[str]:
    """Return attempt names (``{worker}-{attempt}``) with a complete staged set.

    A Lance attempt is complete iff its sidecar, stats, and ``.valid`` marker
    are all present and no ``.invalid`` marker excludes it (design §7.2).

    :param entry_paths: Filenames from one shard's staging directory listing.
    :returns: Sorted attempt names with all of :data:`COMPLETE_ATTEMPT_SUFFIXES`.
    """
    names_by_suffix = {
        suffix: {path[: -len(suffix)] for path in entry_paths if path.endswith(suffix)}
        for suffix in COMPLETE_ATTEMPT_SUFFIXES
    }
    invalid_names = {
        path[: -len(ATTEMPT_INVALID_SUFFIX)]
        for path in entry_paths
        if path.endswith(ATTEMPT_INVALID_SUFFIX)
    }
    return sorted(
        name
        for name in set.intersection(*names_by_suffix.values())
        if name and name not in invalid_names
    )


def invalidate_staged_attempt(spec: DatasetSpec, shard_id: int, attempt_name: str) -> None:
    """Exclude one structurally invalid attempt from reconciliation and resume probes.

    :param spec: Validated dataset spec.
    :param shard_id: Logical shard owning the attempt.
    :param attempt_name: Staging basename without an artifact suffix.
    """
    _upload_empty_marker(
        f"{spec.r2.shard_staging_dir_uri(shard_id)}{attempt_name}{ATTEMPT_INVALID_SUFFIX}"
    )


def shard_has_complete_attempt(spec: DatasetSpec, shard_id: int) -> bool:
    """Return whether any staged-valid attempt exists for ``shard_id``.

    The worker skip-probe: a complete, non-invalidated attempt means the shard
    is already staged and need not be re-rendered (#750 resumability).

    :param spec: Validated dataset spec.
    :param shard_id: Logical shard to probe.
    :returns: ``True`` when at least one complete attempt is staged.
    """
    entries = r2_io.list_entries(spec.r2.shard_staging_dir_uri(shard_id))
    return bool(complete_attempt_names([entry.path for entry in entries]))
