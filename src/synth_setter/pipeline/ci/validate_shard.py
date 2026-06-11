#!/usr/bin/env python3
"""Validate dataset shards against a DatasetSpec.

Performs full per-shard validation. Each shard file is dispatched by its
filename suffix via ``synth_setter.pipeline.schemas.spec.OutputFormat.from_extension``
to either the HDF5 path (``.h5``) or the wds tar path (``.tar``):

- HDF5 path: each top-level dataset's full ``.shape`` matches the writer's
  source-of-truth shape helpers in ``synth_setter.data.vst.shapes`` —
  ``(N, C, time)`` for audio, ``(N, C, n_mels, n_frames)`` for the mel
  spectrogram, and ``(N, num_params)`` for the param array.
- wds tar path: ``metadata.json`` is present and parses as a strict
  ``ShardMetadata``; every ``<batch_start_idx:08d>.<field>.npy`` member loads
  as a numpy array whose trailing dims (``arr.shape[1:]``) match the same
  shape helpers; and the summed row count per field equals
  ``spec.render.samples_per_shard``.

CLI usage:
    python3 -m synth_setter.pipeline.ci.validate_shard <spec.json|r2://bucket/spec.json>

Iterates `spec.shards` and downloads each shard from R2 (under
`r2://{spec.r2.bucket}/{spec.r2.prefix}{shard.filename}`) before validating.
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
from pathlib import Path
from typing import cast

import click
import h5py
import numpy as np
from pydantic import ValidationError

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    audio_dataset_shape,
    mel_dataset_shape,
    param_array_dataset_shape,
)
from synth_setter.pipeline.r2_io import downloaded_to_tempfile
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat
from synth_setter.pipeline.spec_io import read_spec_text

_TAR_METADATA_MEMBER = "metadata.json"
_TAR_NPY_NAME_RE = re.compile(r"^(?P<batch_key>\d{8})\.(?P<field>[^./]+)\.npy$")


def _check_dataset_dtype(shard: Path, key: str, observed: np.dtype) -> None:
    """Reject a dataset whose on-disk dtype disagrees with the writer's contract.

    :param shard: Shard path used to attribute the error to a specific file.
    :param key: Dataset name being checked; lookup key for the per-field dtype map.
    :param observed: dtype read from the open ``h5py.Dataset`` node.
    :raises click.ClickException: If ``observed`` differs from
        ``DATASET_FIELD_DTYPES[key]``.
    """
    expected = DATASET_FIELD_DTYPES[key]
    if observed != expected:
        raise click.ClickException(
            f"shard {shard}: dataset {key!r} has dtype {observed}, expected {expected}."
        )


def check_shards_present(shard_paths: list[Path]) -> None:
    """Fail loud before any output handle opens if a spec-named shard is missing.

    :param shard_paths: All shards a downstream tool will source from, in spec order.
    :raises click.ClickException: Listing each missing path.
    """
    missing = [p for p in shard_paths if not p.is_file()]
    if missing:
        formatted = "\n  ".join(str(p) for p in missing)
        raise click.ClickException(
            f"{len(missing)} shard(s) named by ``spec.shards`` are missing under "
            f"dataset_root:\n  {formatted}"
        )


def check_shard_ids_match_spec_order(spec: DatasetSpec) -> None:
    """Catch a tampered spec whose ``shards[i].shard_id`` no longer equals ``i``.

    :param spec: Loaded ``DatasetSpec``.
    :raises click.ClickException: If any shard's ``shard_id`` disagrees with its index.
    """
    for index, shard in enumerate(spec.shards):
        if shard.shard_id != index:
            raise click.ClickException(
                f"spec.shards[{index}].shard_id={shard.shard_id} disagrees with its "
                f"position; spec.shards must be in shard_id order."
            )


def check_shard_contracts(
    shard_paths: list[Path],
    samples_per_shard: int,
) -> dict[str, tuple[int, ...]]:
    """Validate every shard's structure and return the per-dataset trailing shape.

    Catches a drifted or partial worker upload at the trust boundary, before a
    downstream tool wires the file into a ``VirtualSource`` (which would
    otherwise either silently return fill values or surface as a low-signal
    h5py error mid-run). The returned tails let callers skip reopening the
    first shard to discover the per-dataset trailing shape.

    :param shard_paths: Spec-ordered list of shard files.
    :param samples_per_shard: Required leading-axis length for every dataset.
    :returns: Trailing shape for each required dataset key.
    :raises click.ClickException: With the offending shard, key, and observed
        value (shape, dtype, or row count).
    """
    expected_tails: dict[str, tuple[int, ...]] = {}
    for shard in shard_paths:
        with h5py.File(shard, "r") as f:
            for key in DATASET_FIELD_NAMES:
                if key not in f:
                    raise click.ClickException(
                        f"shard {shard} is missing required dataset {key!r}; "
                        f"present: {sorted(f.keys())}."
                    )
                node = f[key]
                if not isinstance(node, h5py.Dataset):
                    raise click.ClickException(
                        f"shard {shard}: key {key!r} is a {type(node).__name__}, not a Dataset."
                    )
                _check_dataset_dtype(shard, key, node.dtype)
                if node.shape[0] != samples_per_shard:
                    raise click.ClickException(
                        f"shard {shard}: dataset {key!r} has {node.shape[0]} rows, "
                        f"expected samples_per_shard={samples_per_shard}."
                    )
                tail = tuple(node.shape[1:])
                expected_tail = expected_tails.setdefault(key, tail)
                if tail != expected_tail:
                    raise click.ClickException(
                        f"shard {shard}: dataset {key!r} trailing shape {tail} disagrees "
                        f"with first shard's {expected_tail}."
                    )
    return expected_tails


def _expected_dataset_shapes(spec: DatasetSpec) -> dict[str, tuple[int, ...]]:
    """Full per-field shapes (N + inner) the writer emits for ``spec``.

    Keys match ``DATASET_FIELD_NAMES``; values come from the writer's own
    shape helpers in ``synth_setter.data.vst.shapes`` so a future renderer
    change that drifts the audio / mel / param shapes fails fast here.

    :param spec: Dataset spec whose ``render`` config and ``num_params`` parameterize
        the per-field shapes the writer would emit for one shard.
    :returns: Mapping with one entry per writer-emitted dataset name to its full
        ``(N, ...)`` shape tuple.
    :rtype: dict[str, tuple[int, ...]]
    """
    render = spec.render
    num_samples = render.samples_per_shard
    return {
        AUDIO_FIELD: audio_dataset_shape(
            num_samples, render.channels, render.sample_rate, render.signal_duration_seconds
        ),
        MEL_SPEC_FIELD: mel_dataset_shape(
            num_samples, render.channels, render.sample_rate, render.signal_duration_seconds
        ),
        PARAM_ARRAY_FIELD: param_array_dataset_shape(num_samples, spec.num_params),
    }


def validate_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Validate one shard against a DatasetSpec, dispatching by filename suffix.

    Suffix dispatch via ``OutputFormat.from_extension``: ``.h5`` -> HDF5 path,
    ``.tar`` -> tar/wds path. Any other suffix is rejected with an error
    naming the registered set so a typo or wrong-format file does not
    surface as a misleading "not valid HDF5".

    :param shard_path: Local filesystem path to the shard to validate.
    :param spec: Dataset spec the shard is expected to conform to.
    :returns: List of error strings (empty = valid).
    :rtype: list[str]
    """
    if not shard_path.exists():
        return [f"shard file not found: {shard_path}"]

    fmt = OutputFormat.from_extension(shard_path.suffix)
    if fmt is OutputFormat.HDF5:
        return _validate_h5_shard(shard_path, spec)
    if fmt is OutputFormat.WDS:
        return _validate_tar_shard(shard_path, spec)
    if fmt is OutputFormat.LANCE:
        return _validate_lance_shard(shard_path, spec)
    return [
        f"unsupported shard suffix {shard_path.suffix!r} "
        f"(expected one of: {sorted(f.extension for f in OutputFormat)})"
    ]


def _validate_h5_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Validate an HDF5 shard's datasets, row counts, and inner shapes.

    Opens the file with h5py, checks every dataset named in ``DATASET_FIELD_NAMES``
    is present, and that each dataset's full ``.shape`` matches what
    ``_expected_dataset_shapes`` predicts for ``spec``.

    :param shard_path: Local filesystem path to the HDF5 shard.
    :param spec: Dataset spec the shard is expected to conform to.
    :returns: List of error strings (empty = valid).
    :rtype: list[str]
    """
    try:
        f = h5py.File(shard_path, "r")
    except OSError:
        return [f"file is not valid HDF5: {shard_path}"]

    expected_shapes = _expected_dataset_shapes(spec)
    errors: list[str] = []
    with f:
        for name in DATASET_FIELD_NAMES:
            if name not in f:
                errors.append(f"missing dataset: {name!r}")
                continue
            actual = cast(h5py.Dataset, f[name]).shape
            expected = expected_shapes[name]
            if actual != expected:
                errors.append(f"dataset {name!r} has shape {actual}, expected {expected}")

    return errors


def _validate_tar_metadata(tar: tarfile.TarFile, member_name: str) -> list[str]:
    """Validate the tar's ``metadata.json`` parses as a strict ``ShardMetadata``.

    Trust-boundary parse: ``ShardMetadata`` is constructed with
    ``extra="forbid"`` and ``strict=True`` so unknown keys or type-coerced
    values surface as a ``ValidationError`` instead of being silently accepted.

    :param tar: Open ``TarFile`` handle. Caller owns the lifecycle.
    :param member_name: Name of the metadata member to extract.
    :returns: One error string per problem; empty if metadata is valid.
    :rtype: list[str]
    """
    extracted = tar.extractfile(member_name)
    if extracted is None:
        return [f"unable to extract tar member: {member_name!r}"]
    payload = extracted.read()
    try:
        ShardMetadata.model_validate_json(payload)
    except ValidationError as exc:
        return [f"{member_name}: invalid ShardMetadata: {exc}"]
    return []


def _validate_tar_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Validate a wds tar shard's members, metadata, row counts, and inner shapes.

    Tar member layout: per-batch ``<batch_start_idx:08d>.<field>.npy`` plus a
    single ``metadata.json``. Checks:

    1. The file opens as an uncompressed tar archive.
    2. ``metadata.json`` is present and parses as a strict ``ShardMetadata``.
    3. Every ``.npy`` member name matches ``<batch_start_idx:08d>.<field>.npy``
       (no missing or malformed batch keys).
    4. Every batch-keyed ``.npy`` member loads as a numpy array.
    5. Within each batch key, all writer fields are present and share the same
       row count (the writer's per-batch invariant).
    6. Each per-batch array's trailing dims (``arr.shape[1:]``) match the
       corresponding writer shape (dropping the N dim).
    7. The summed row count per field across all batches equals
       ``spec.render.samples_per_shard``.

    :param shard_path: Local filesystem path to the tar shard.
    :param spec: Dataset spec the shard is expected to conform to.
    :returns: List of error strings (empty = valid).
    :rtype: list[str]
    """
    try:
        tar = tarfile.open(shard_path, mode="r:")
    except tarfile.TarError:
        return [f"file is not a valid uncompressed tar archive: {shard_path}"]

    expected_inner = {field: shape[1:] for field, shape in _expected_dataset_shapes(spec).items()}
    errors: list[str] = []
    with tar:
        members = sorted(m.name for m in tar.getmembers())

        if _TAR_METADATA_MEMBER not in members:
            errors.append(f"missing tar member: {_TAR_METADATA_MEMBER!r}")
        else:
            errors.extend(_validate_tar_metadata(tar, _TAR_METADATA_MEMBER))

        rows_by_batch: dict[str, dict[str, int]] = {}
        for name in members:
            if not name.endswith(".npy"):
                continue
            match = _TAR_NPY_NAME_RE.match(name)
            if match is None:
                errors.append(
                    f"malformed tar member name: {name!r} "
                    f"(expected '<batch_start_idx:08d>.<field>.npy')"
                )
                continue
            field = match.group("field")
            if field not in expected_inner:
                errors.append(f"unknown field in tar member: {name!r}")
                continue
            arr_or_err = _load_npy_member(tar, name)
            if isinstance(arr_or_err, str):
                errors.append(arr_or_err)
                continue
            rows_by_batch.setdefault(match.group("batch_key"), {})[field] = arr_or_err.shape[0]
            if arr_or_err.shape[1:] != expected_inner[field]:
                errors.append(
                    f"{name}: inner shape {arr_or_err.shape[1:]} does not match "
                    f"expected {expected_inner[field]}"
                )

        errors.extend(_check_per_batch_invariants(rows_by_batch))
        errors.extend(_check_row_totals(rows_by_batch, spec.render.samples_per_shard))

    return errors


def _validate_lance_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Validate a Lance single-file shard's schema, metadata, and row count.

    :param shard_path: Local filesystem path to the Lance shard.
    :param spec: Dataset spec the shard is expected to conform to.
    :returns: List of error strings (empty = valid).
    :rtype: list[str]
    """
    from lance.file import LanceFileReader

    from synth_setter.pipeline.data.lance_shard import read_shard_metadata

    try:
        reader = LanceFileReader(str(shard_path))
        metadata = reader.metadata()
    except (OSError, ValueError, RuntimeError) as exc:
        return [f"file is not a valid Lance file: {shard_path}: {exc}"]

    errors: list[str] = []
    schema = metadata.schema
    try:
        read_shard_metadata(schema)
    except ValueError as exc:
        errors.append(str(exc))

    num_rows = reader.num_rows()
    if num_rows != spec.render.samples_per_shard:
        errors.append(f"file has {num_rows} rows, expected {spec.render.samples_per_shard}")

    expected_shapes = _expected_dataset_shapes(spec)
    for name in DATASET_FIELD_NAMES:
        field = schema.field(name) if name in schema.names else None
        if field is None:
            errors.append(f"missing column: {name!r}")
            continue
        errors.extend(_validate_lance_field(name, field, expected_shapes[name]))
    return errors


def _validate_lance_field(name: str, field: object, expected_shape: tuple[int, ...]) -> list[str]:
    """Validate one Lance fixed-shape tensor field against the writer contract.

    :param name: Column name being checked.
    :param field: Arrow schema field read from the Lance file.
    :param expected_shape: Full expected shape including leading row axis.
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
    expected_dtype = pa.from_numpy_dtype(DATASET_FIELD_DTYPES[name])
    if field_type.value_type != expected_dtype:
        errors.append(
            f"column {name!r} has value type {field_type.value_type}, expected {expected_dtype}"
        )
    return errors


def _check_per_batch_invariants(rows_by_batch: dict[str, dict[str, int]]) -> list[str]:
    """Check the writer's per-batch invariant: every batch key has all fields with matching rows.

    The writer emits one ``.npy`` per ``DATASET_FIELD_NAMES`` for each batch
    key, and all three are sliced from the same per-sample list — so within
    a batch the row counts must agree across fields. A drift here would mean
    misaligned WebDataset samples even if the per-field totals happen to add
    up.

    :param rows_by_batch: Mapping ``batch_key -> {field: rows}`` populated while
        iterating tar members. Missing fields surface as the batch key not
        having an entry for that field.
    :returns: List of error strings (empty = every batch key has all fields and
        a single row count).
    :rtype: list[str]
    """
    errors: list[str] = []
    for batch_key in sorted(rows_by_batch):
        per_field = rows_by_batch[batch_key]
        missing = [field for field in DATASET_FIELD_NAMES if field not in per_field]
        if missing:
            errors.append(
                f"batch {batch_key!r} missing field(s): {missing} "
                f"(expected one '.npy' per field {list(DATASET_FIELD_NAMES)})"
            )
            continue
        row_counts = {per_field[field] for field in DATASET_FIELD_NAMES}
        if len(row_counts) > 1:
            errors.append(
                f"batch {batch_key!r} row-count mismatch across fields: "
                f"{ {field: per_field[field] for field in DATASET_FIELD_NAMES} }"
            )
    return errors


def _check_row_totals(
    rows_by_batch: dict[str, dict[str, int]], samples_per_shard: int
) -> list[str]:
    """Check each field's summed row count across all batches equals ``samples_per_shard``.

    :param rows_by_batch: Mapping ``batch_key -> {field: rows}`` populated while
        iterating tar members.
    :param samples_per_shard: The writer's per-shard row total each field must sum to.
    :returns: List of error strings (empty = every field's row total matches).
    :rtype: list[str]
    """
    errors: list[str] = []
    for field in DATASET_FIELD_NAMES:
        batches_with_field = [
            per_field[field] for per_field in rows_by_batch.values() if field in per_field
        ]
        if not batches_with_field:
            errors.append(f"missing tar member: '*.{field}.npy'")
            continue
        total = sum(batches_with_field)
        if total != samples_per_shard:
            errors.append(
                f"field {field!r} summed {total} rows across "
                f"{len(batches_with_field)} batch(es), expected {samples_per_shard}"
            )
    return errors


def _load_npy_member(tar: tarfile.TarFile, name: str) -> np.ndarray | str:
    """Load a ``.npy`` tar member into a numpy array, or return an error string.

    Rejects payloads that ``np.load`` resolves to anything other than a single
    ``ndarray`` with at least one dimension (e.g. ``.npz`` bytes saved under a
    ``.npy`` name resolve to ``NpzFile``; a 0-d scalar has no row dim) — both
    would later crash the per-batch ``arr.shape[0]`` / ``arr.shape[1:]``
    accesses with an opaque traceback instead of surfacing here.

    :param tar: Open ``TarFile`` handle. Caller owns the lifecycle.
    :param name: Name of the ``.npy`` member to extract and load.
    :returns: Loaded array on success, or a single error-message string describing
        the extraction or load failure.
    :rtype: np.ndarray | str
    """
    extracted = tar.extractfile(name)
    if extracted is None:
        return f"unable to extract tar member: {name!r}"
    try:
        loaded = np.load(io.BytesIO(extracted.read()))
    except (ValueError, EOFError, OSError) as exc:
        return f"{name}: malformed npy payload: {exc}"
    if not isinstance(loaded, np.ndarray):
        return f"{name}: expected a single ndarray, got {type(loaded).__name__}"
    if loaded.ndim == 0:
        return f"{name}: expected ndarray with at least one dimension, got 0-d scalar"
    return loaded


def _load_spec(spec_arg: str) -> DatasetSpec:
    """Load a spec from a local path, ``file://`` URI, or ``r2://`` URI.

    :param spec_arg: Local filesystem path, ``file://`` URI, or ``r2://...`` URI
        pointing at the spec JSON file.
    :returns: Parsed ``DatasetSpec`` instance.
    :rtype: DatasetSpec
    """
    return DatasetSpec.model_validate_json(read_spec_text(spec_arg))


def validate_all_shards_from_r2(spec: DatasetSpec) -> list[str]:
    """Validate every shard in ``spec.shards`` by downloading from R2.

    :param spec: Dataset spec whose ``shards`` list drives the iteration; each
        listed shard is fetched from R2 under ``r2://{spec.r2.bucket}/{spec.r2.prefix}``.
    :returns: Aggregated error strings across all shards (empty = all valid). Each
        error is prefixed with the shard filename so the source is obvious.
    :rtype: list[str]
    """
    errors: list[str] = []
    for shard in spec.shards:
        shard_object_uri = spec.r2.shard_uri(shard)
        with downloaded_to_tempfile(shard_object_uri) as local_shard:
            shard_errors = validate_shard(local_shard, spec)
        for err in shard_errors:
            errors.append(f"{shard.filename}: {err}")
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
