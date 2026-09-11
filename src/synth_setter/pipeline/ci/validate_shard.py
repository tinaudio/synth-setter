#!/usr/bin/env python3
"""Validate dataset shards against a DatasetSpec.

Performs full per-shard validation. Each shard is dispatched by its filename
suffix via ``synth_setter.pipeline.schemas.spec.OutputFormat.from_extension``
to the Lance path (``.lance``):

- Lance path (local shard, worker-side pre-staging check): schema metadata
  parses as a strict ``ShardMetadata``; tensor fields match the writer's dtype
  and shape contracts; preview fields are non-null binary/string columns; and
  ``num_rows`` equals ``spec.render.samples_per_shard``. Tensor values must be
  finite and normalized, UUIDs must match stored audio, and MP3s must decode at
  the configured sample rate and channel count.
- Lance path (from R2): structural check of each shard's staged winner
  attempt — sidecar + stats + ``.valid`` present, sidecar round-trips through
  Lance, row counts agree, fragment data files exist under the assigned split.
  No rows are decoded; full shape/value checks already ran worker-side.

CLI usage:
    python3 -m synth_setter.pipeline.ci.validate_shard <spec.json|r2://bucket/spec.json>

Iterates `spec.shards` from R2; Lance shards are reconciled from the
`metadata/workers/shards/` staging prefix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import lance
    import pyarrow as pa

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    AUDIO_MP3_FIELD,
    AUDIO_MP3_FIELD_METADATA,
    AUDIO_UUID_FIELD,
    DATASET_FIELD_NAMES,
    PARAM_ARRAY_FIELD,
    PREVIEW_FIELD_NAMES,
    dataset_field_dtypes,
    dataset_field_shapes,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat, ShardSpec
from synth_setter.pipeline.spec_io import read_spec_text

# Bound scan buffers before NumPy checks and preview decoding amplify batch memory.
LANCE_VALIDATION_BATCH_SIZE_BYTES = 64 * 1024 * 1024


def _expected_dataset_shapes(spec: DatasetSpec) -> dict[str, tuple[int, ...]]:
    """Adapt ``spec`` to :func:`dataset_field_shapes`, the shape contract's home.

    :param spec: Dataset spec whose ``render`` config and ``num_params`` parameterize
        the per-field shapes the writer would emit for one shard.
    :returns: Mapping with one entry per writer-emitted dataset name to its full
        ``(N, ...)`` shape tuple.
    :rtype: dict[str, tuple[int, ...]]
    """
    return dataset_field_shapes(spec.render, spec.num_params)


def validate_shard(
    shard_path: Path, spec: DatasetSpec, *, expected_shard: ShardSpec | None = None
) -> list[str]:
    """Validate one shard against a DatasetSpec, dispatching by filename suffix.

    Suffix dispatch via ``OutputFormat.from_extension``: ``.lance`` -> Lance
    path. Any other suffix is rejected with an error naming the registered set
    so a typo or wrong-format file does not surface as a misleading parse error.

    :param shard_path: Local filesystem path to the shard to validate.
    :param spec: Dataset spec the shard is expected to conform to.
    :param expected_shard: Explicit identity for growing shards outside ``spec.shards``.
    :returns: List of error strings (empty = valid).
    :rtype: list[str]
    """
    if not shard_path.exists():
        return [f"shard file not found: {shard_path}"]

    fmt = OutputFormat.from_extension(shard_path.suffix)
    if fmt is OutputFormat.LANCE:
        return _validate_lance_shard(shard_path, spec, expected_shard=expected_shard)
    return [
        f"unsupported shard suffix {shard_path.suffix!r} "
        f"(expected one of: {sorted(f.extension for f in OutputFormat)})"
    ]


def _metadata_mismatch_errors(
    metadata: ShardMetadata,
    expected: ShardMetadata,
    source: str,
    present_fields: set[str] | frozenset[str],
) -> list[str]:
    """Return errors for sidecar provenance that disagrees with the input spec.

    :param metadata: Metadata parsed from the shard being validated.
    :param expected: Metadata projected from the spec used for validation.
    :param source: Human-readable metadata source for the error prefix.
    :param present_fields: Metadata fields physically present in the artifact.
    :returns: One error per mismatched field.
    """
    errors: list[str] = []
    if "render_contract_digest" not in present_fields:
        errors.append(f"{source}: render_contract_digest is missing")
    for field in (
        "base_seed",
        "sample_offset",
        "attempts_per_sample",
        "render_contract_digest",
    ):
        if field not in present_fields:
            continue
        observed = getattr(metadata, field)
        wanted = getattr(expected, field)
        if observed != wanted:
            errors.append(f"{source}: {field}={observed!r} does not match spec value {wanted!r}")
    return errors


def _expected_seed_position(shard_path: Path, spec: DatasetSpec) -> tuple[int, int]:
    """Return the seed and sample offset injected for ``shard_path``.

    :param shard_path: Local path being validated.
    :param spec: Dataset spec whose ``shards`` define seed positions.
    :returns: Matching shard seed and offset, or render defaults for ad hoc paths.
    """
    for shard in spec.shards:
        if shard.filename == shard_path.name:
            return shard.seed, shard.sample_offset
    return spec.render.base_seed, spec.render.sample_offset


def _expected_shard_metadata(
    spec: DatasetSpec,
    *,
    base_seed: int | None = None,
    sample_offset: int | None = None,
) -> ShardMetadata:
    """Project the spec's render config onto the shard metadata contract.

    :param spec: Dataset spec whose render config the shard should match.
    :param base_seed: Seed injected into the renderer; defaults to the render config.
    :param sample_offset: Split-local offset injected into the renderer; defaults to the render
        config.
    :returns: Strict shard metadata expected for rendered shards.
    """
    render = spec.render.model_copy(
        update={
            "base_seed": spec.render.base_seed if base_seed is None else base_seed,
            "sample_offset": (
                spec.render.sample_offset if sample_offset is None else sample_offset
            ),
        }
    )
    return render.shard_metadata()


def _validate_lance_shard(
    shard_path: Path, spec: DatasetSpec, *, expected_shard: ShardSpec | None = None
) -> list[str]:
    """Validate a Lance shard dataset's schema, metadata, and row count.

    :param shard_path: Local filesystem path to the Lance shard dataset directory.
    :param spec: Dataset spec the shard is expected to conform to.
    :param expected_shard: Explicit identity for growing shards outside ``spec.shards``.
    :returns: List of error strings (empty = valid).
    """
    import lance

    try:
        dataset = lance.dataset(str(shard_path))
    except (OSError, ValueError, RuntimeError) as exc:
        return [f"path is not a valid Lance dataset: {shard_path}: {exc}"]
    base_seed, sample_offset = (
        (expected_shard.seed, expected_shard.sample_offset)
        if expected_shard is not None
        else _expected_seed_position(shard_path, spec)
    )
    return _validate_lance_dataset(
        dataset,
        spec,
        base_seed=base_seed,
        sample_offset=sample_offset,
    )


def _validate_lance_dataset(
    dataset: lance.LanceDataset,
    spec: DatasetSpec,
    *,
    base_seed: int | None = None,
    sample_offset: int | None = None,
) -> list[str]:
    """Validate an open Lance shard dataset's schema, metadata, and row count.

    Local-path validation only (the worker's pre-staging check); the from-R2
    path validates staged winner attempts instead — see
    :func:`_validate_all_lance_shards_from_r2`.

    :param dataset: Open Lance dataset handle for one shard.
    :param spec: Dataset spec the shard is expected to conform to.
    :param base_seed: Seed expected in schema metadata.
    :param sample_offset: Split-local sample offset expected in schema metadata.
    :returns: List of error strings (empty = valid).
    """
    from synth_setter.pipeline.data.lance_shard import read_shard_metadata

    errors: list[str] = []
    schema = dataset.schema
    expected_metadata = _expected_shard_metadata(
        spec, base_seed=base_seed, sample_offset=sample_offset
    )
    try:
        metadata = read_shard_metadata(schema)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        errors.extend(
            _metadata_mismatch_errors(
                metadata,
                expected_metadata,
                "Lance schema metadata",
                metadata.model_fields_set,
            )
        )

    num_rows = dataset.count_rows()
    if num_rows != spec.render.samples_per_shard:
        errors.append(f"dataset has {num_rows} rows, expected {spec.render.samples_per_shard}")

    expected_shapes = _expected_dataset_shapes(spec)
    expected_dtypes = dataset_field_dtypes(spec.render)
    schema_errors: list[str] = []
    for name in DATASET_FIELD_NAMES:
        field = schema.field(name) if name in schema.names else None
        if field is None:
            schema_errors.append(f"missing column: {name!r}")
            continue
        schema_errors.extend(
            _validate_lance_field(
                name,
                field,
                expected_shapes[name],
                expected_dtype=expected_dtypes[name],
            )
        )
    for name in PREVIEW_FIELD_NAMES:
        field = schema.field(name) if name in schema.names else None
        if field is None:
            schema_errors.append(f"missing column: {name!r}")
            continue
        schema_errors.extend(_validate_preview_field(name, field))
    errors.extend(schema_errors)
    if not schema_errors:
        errors.extend(_validate_lance_values(dataset, spec))
    return errors


def _validate_lance_values(dataset: lance.LanceDataset, spec: DatasetSpec) -> list[str]:
    """Validate tensor and preview values before a worker stages a shard.

    :param dataset: Structurally valid local Lance shard dataset.
    :param spec: Dataset contract supplying playback rate and channel count.
    :returns: One error per violated field value contract.
    """
    errors: set[str] = set()
    row_offset = 0
    columns = [*DATASET_FIELD_NAMES, *PREVIEW_FIELD_NAMES]
    for batch in dataset.to_batches(
        columns=columns,
        batch_size_bytes=LANCE_VALIDATION_BATCH_SIZE_BYTES,
    ):
        errors.update(_validate_tensor_batch_values(batch))
        errors.update(_validate_preview_batch_values(batch, spec, row_offset))
        row_offset += batch.num_rows
    return sorted(errors)


def _validate_tensor_batch_values(batch: pa.RecordBatch) -> set[str]:
    """Validate finite and normalized tensor values in one record batch.

    :param batch: Structurally valid shard rows.
    :returns: Violated tensor value contracts.
    """
    errors: set[str] = set()
    for name in DATASET_FIELD_NAMES:
        values = batch.column(name).to_numpy_ndarray()
        if not np.isfinite(values).all():
            errors.add(f"column {name!r} contains non-finite values")
            continue
        if name == AUDIO_FIELD and ((values < -1) | (values > 1)).any():
            errors.add(f"column {name!r} contains values outside [-1, 1]")
        if name == PARAM_ARRAY_FIELD and ((values < 0) | (values > 1)).any():
            errors.add(f"column {name!r} contains values outside [0, 1]")
    return errors


def _validate_preview_batch_values(
    batch: pa.RecordBatch,
    spec: DatasetSpec,
    row_offset: int,
) -> set[str]:
    """Validate UUID integrity and complete MP3 playback geometry for one batch.

    :param batch: Structurally valid shard rows.
    :param spec: Dataset contract supplying playback rate and channel count.
    :param row_offset: Dataset row index of the batch's first row.
    :returns: Violated preview value contracts.
    """
    from synth_setter.data.vst.audio_preview import audio_uuid

    errors: set[str] = set()
    audio_rows = batch.column(AUDIO_FIELD).to_numpy_ndarray()
    mp3_rows = batch.column(AUDIO_MP3_FIELD).to_pylist()
    uuid_rows = batch.column(AUDIO_UUID_FIELD).to_pylist()
    for batch_index, (audio, mp3, stored_uuid) in enumerate(
        zip(audio_rows, mp3_rows, uuid_rows, strict=True)
    ):
        row_index = row_offset + batch_index
        if stored_uuid != audio_uuid(audio):
            errors.add(f"column {AUDIO_UUID_FIELD!r} row {row_index} does not match audio")
        errors.update(_validate_mp3_payload(mp3, audio.shape[-1], spec, row_index))
    return errors


def _validate_mp3_payload(
    payload: bytes,
    expected_frames: int,
    spec: DatasetSpec,
    row_index: int,
) -> set[str]:
    """Decode one complete MP3 payload and validate its playback contract.

    :param payload: Stored MP3 byte string.
    :param expected_frames: Minimum decoded frame count from the source audio row.
    :param spec: Dataset contract supplying playback rate and channel count.
    :param row_index: Dataset row index used in diagnostics.
    :returns: Violated MP3 playback contracts.
    """
    import io

    from pedalboard.io import AudioFile

    errors: set[str] = set()
    try:
        with AudioFile(io.BytesIO(payload)) as audio_file:
            decoded = audio_file.read(audio_file.frames)
            decoded_rate = int(audio_file.samplerate)
            decoded_channels = audio_file.num_channels
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {f"column {AUDIO_MP3_FIELD!r} row {row_index} is not decodable: {exc}"}
    decoded_frames = decoded.shape[1]
    if decoded_frames < expected_frames:
        errors.add(
            f"column {AUDIO_MP3_FIELD!r} row {row_index} decoded {decoded_frames} frames, "
            f"expected at least {expected_frames}"
        )
    if decoded_rate != spec.render.sample_rate:
        errors.add(
            f"column {AUDIO_MP3_FIELD!r} row {row_index} has sample rate "
            f"{decoded_rate}, expected {spec.render.sample_rate}"
        )
    if decoded_channels != spec.render.channels:
        errors.add(
            f"column {AUDIO_MP3_FIELD!r} row {row_index} has {decoded_channels} "
            f"channels, expected {spec.render.channels}"
        )
    return errors


def _validate_lance_field(
    name: str,
    field: object,
    expected_shape: tuple[int, ...],
    *,
    expected_dtype: np.dtype,
) -> list[str]:
    """Validate one Lance fixed-shape tensor field against the writer contract.

    :param name: Column name being checked.
    :param field: Arrow schema field read from the Lance file.
    :param expected_shape: Full expected shape including leading row axis.
    :param expected_dtype: Physical scalar dtype required by the dataset spec.
    :returns: List of error strings for this field.
    :rtype: list[str]
    """
    import pyarrow as pa

    if not isinstance(field, pa.Field):
        return [f"column {name!r} schema entry is not an Arrow field: {field!r}"]
    arrow_field = cast(pa.Field, field)
    errors: list[str] = []
    field_type = arrow_field.type
    if not isinstance(field_type, pa.FixedShapeTensorType):
        return [f"column {name!r} has type {field_type}, expected fixed-shape tensor"]
    expected_inner = expected_shape[1:]
    if tuple(field_type.shape) != expected_inner:
        errors.append(
            f"column {name!r} has inner shape {tuple(field_type.shape)}, expected {expected_inner}"
        )
    expected_arrow_dtype = pa.from_numpy_dtype(expected_dtype)
    if field_type.value_type != expected_arrow_dtype:
        errors.append(
            f"column {name!r} has value type {field_type.value_type}, "
            f"expected {expected_arrow_dtype}"
        )
    return errors


def _validate_preview_field(name: str, field: object) -> list[str]:
    """Validate one preview field's Arrow type, nullability, and metadata.

    :param name: Preview column name.
    :param field: Arrow schema field read from the Lance dataset.
    :returns: Schema contract violations for the preview column.
    """
    import pyarrow as pa

    if not isinstance(field, pa.Field):
        return [f"column {name!r} schema entry is not an Arrow field: {field!r}"]
    arrow_field = cast(pa.Field, field)
    expected_type = pa.binary() if name == AUDIO_MP3_FIELD else pa.string()
    errors = []
    if arrow_field.type != expected_type:
        errors.append(f"column {name!r} has type {arrow_field.type}, expected {expected_type}")
    if arrow_field.nullable:
        errors.append(f"column {name!r} must be non-nullable")
    if name == AUDIO_MP3_FIELD and arrow_field.metadata != AUDIO_MP3_FIELD_METADATA:
        errors.append(
            f"column {name!r} has metadata {arrow_field.metadata}, "
            f"expected {AUDIO_MP3_FIELD_METADATA}"
        )
    return errors


def _load_spec(spec_arg: str) -> DatasetSpec:
    """Load a spec from a local path, ``file://`` URI, or ``r2://`` URI.

    :param spec_arg: Local filesystem path, ``file://`` URI, or ``r2://...`` URI
        pointing at the spec JSON file.
    :returns: Parsed ``DatasetSpec`` instance.
    :rtype: DatasetSpec
    """
    return DatasetSpec.model_validate_json(read_spec_text(spec_arg))


def validate_all_shards_from_r2(spec: DatasetSpec) -> list[str]:
    """Validate every shard in ``spec.shards`` from R2.

    Delegates to :func:`_validate_all_lance_shards_from_r2`, which structurally
    checks each shard's staged winner attempt (#1776) without decoding any rows.

    :param spec: Dataset spec whose ``shards`` list drives the iteration; each
        listed shard lives under ``r2://{spec.r2.bucket}/{spec.r2.prefix}``.
    :returns: Aggregated error strings across all shards (empty = all valid). Each
        error is prefixed with the shard filename so the source is obvious.
    :raises ValueError: ``spec.output_format`` is not a supported shard format.
    """
    if spec.output_format is OutputFormat.LANCE:
        return _validate_all_lance_shards_from_r2(spec)
    raise ValueError(f"unsupported output_format: {spec.output_format!r}")


def _validate_all_lance_shards_from_r2(spec: DatasetSpec) -> list[str]:
    """Structurally validate every Lance shard's staged winner attempt (#1776).

    Lance workers stage per-attempt fragments, not per-shard datasets, so this
    checks each shard's would-be winner the same way finalize will: complete
    attempt set present, sidecar round-trips through Lance, row count matches
    spec and stats, fragment data files exist under the assigned split. Full
    shape/value validation already ran worker-side against the local render —
    re-reading rows here would decode data the design keeps untouched.
    Structural failures aggregate per shard; environmental failures (rclone
    auth/network, surfacing as ``CalledProcessError``) propagate and abort per
    ``r2_io``'s fail-fast contract, so an outage never reads as bad data.

    :param spec: Dataset spec whose ``shards`` list drives the iteration.
    :returns: Aggregated error strings across all shards, each prefixed with the
        shard filename.
    """
    from synth_setter.pipeline.data.lance_finalize import (
        select_checked_winner,
        staged_complete_attempts,
    )

    attempts = staged_complete_attempts(spec)
    errors: list[str] = []
    for shard in spec.shards:
        shard_attempts = attempts.get(shard.shard_id)
        if not shard_attempts:
            errors.append(
                f"{shard.filename}: no staged-valid attempt under "
                f"{spec.r2.shard_staging_dir_uri(shard.shard_id)}"
            )
            continue
        try:
            select_checked_winner(spec, shard_attempts)
        except ValueError as exc:
            errors.append(f"{shard.filename}: {exc}")
    return errors


def main() -> None:
    """CLI entry point: validate every shard referenced by a spec.

    The single argument is a spec JSON path, a ``file://`` URI, or an
    ``r2://bucket/key.json`` URI. Each shard listed in ``spec.shards`` is
    fetched from R2 and validated.
    """
    if len(sys.argv) != 2:
        sys.stderr.write(
            f"Usage: {sys.argv[0]} <spec.json|file:///abs/path/spec.json|r2://bucket/spec.json>\n"
        )
        sys.exit(1)

    spec_arg = sys.argv[1]
    spec = _load_spec(spec_arg)

    errors = validate_all_shards_from_r2(spec)

    if errors:
        for error in errors:
            sys.stderr.write(f"FAIL: {error}\n")
        sys.exit(1)

    sys.stdout.write(f"OK: all {len(spec.shards)} shards valid\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
