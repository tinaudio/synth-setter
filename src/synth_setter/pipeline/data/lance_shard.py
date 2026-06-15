"""Lance dataset-shard helpers shared by writer, validator, and finalize."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.fragment import LanceFragment
from pydantic import ValidationError

from synth_setter.data.vst.shapes import BLOB_FIELDS, DATASET_FIELD_DTYPES, DATASET_FIELD_NAMES
from synth_setter.pipeline.schemas.shard_metadata import BlobFieldSpec, ShardMetadata

SHARD_METADATA_SCHEMA_KEY = b"synth_setter.shard_metadata"
# Schema-metadata key holding the per-row shape/dtype of every BLOB column, so a
# ``large_binary`` column stays self-describing (its Arrow type carries neither).
BLOB_FIELD_SPECS_SCHEMA_KEY = b"synth_setter.blob_field_specs"

# Pin the on-disk Lance file-format version so it never floats with the pylance
# default across upgrades; "2.1" equals the current default. See
# docs/design/lance-dataset-api-migration.md.
LANCE_DATA_STORAGE_VERSION = "2.1"


def lance_schema(
    field_shapes: dict[str, tuple[int, ...]],
    metadata: ShardMetadata,
) -> pa.Schema:
    """Build the Arrow schema used by one Lance shard file.

    ``BLOB_FIELDS`` become opaque ``large_binary`` columns (their shape/dtype is
    recorded under :data:`BLOB_FIELD_SPECS_SCHEMA_KEY`); the rest stay fixed-shape
    tensors.

    :param field_shapes: Full writer shapes including the leading row axis.
    :param metadata: Per-shard render metadata to embed in schema metadata.
    :returns: Arrow schema with the shard's columns and embedded metadata.
    """
    fields = []
    blob_specs: dict[str, dict[str, object]] = {}
    for field in DATASET_FIELD_NAMES:
        dtype = DATASET_FIELD_DTYPES[field]
        inner_shape = field_shapes[field][1:]
        if field in BLOB_FIELDS:
            fields.append(pa.field(field, pa.large_binary(), nullable=False))
            blob_specs[field] = BlobFieldSpec(
                shape=list(inner_shape), dtype=dtype.name
            ).model_dump()
        else:
            tensor_type = pa.fixed_shape_tensor(pa.from_numpy_dtype(dtype), inner_shape)
            fields.append(pa.field(field, tensor_type, nullable=False))
    return pa.schema(
        fields,
        metadata={
            SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode("utf-8"),
            BLOB_FIELD_SPECS_SCHEMA_KEY: json.dumps(blob_specs).encode("utf-8"),
        },
    )


def read_blob_field_specs(schema: pa.Schema) -> dict[str, BlobFieldSpec]:
    """Parse the per-column BLOB shape/dtype specs from Lance schema metadata.

    :param schema: Arrow schema read from a Lance file.
    :returns: Spec per BLOB column; empty when the key is absent (e.g. a legacy all-tensor shard
        predating the BLOB columns).
    :raises ValueError: A spec entry is present but malformed, or names a column
        that is missing from ``schema`` or is not a ``large_binary`` column
        (consumers classify a column as BLOB by its presence here, so a tampered
        spec must not be able to misclassify a tensor column).
    """
    payload = (schema.metadata or {}).get(BLOB_FIELD_SPECS_SCHEMA_KEY)
    if payload is None:
        return {}
    try:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
        specs = {name: BlobFieldSpec.model_validate(entry) for name, entry in raw.items()}
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"invalid blob field specs: {exc}") from exc
    schema_names = set(schema.names)
    for name in specs:
        if name not in schema_names:
            raise ValueError(f"blob field spec names unknown column {name!r}")
        if not pa.types.is_large_binary(schema.field(name).type):
            raise ValueError(
                f"blob field spec for non-large_binary column {name!r}: {schema.field(name).type}"
            )
    return specs


def blob_array(values: np.ndarray, spec: BlobFieldSpec) -> pa.Array:
    """Encode an ``(N, *inner)`` ndarray as a ``large_binary`` column of raw row bytes.

    Each row is serialized as C-contiguous, native-endian bytes in logical order
    (decode with :func:`decode_blob_array`); the pipeline reads and writes on one
    endianness, so the bytes are not portable across architectures.

    :param values: Rows to encode; cast to ``spec.dtype`` and required to have
        shape ``(N, *spec.shape)`` with ``N >= 1``.
    :param spec: Per-row shape and dtype the column is written with.
    :returns: Arrow ``large_binary`` array, one element per row.
    :raises ValueError: ``values`` inner axes differ from ``spec.shape``, or the batch is empty.
    """
    inner_shape = tuple(spec.shape)
    if values.shape[1:] != inner_shape:
        raise ValueError(f"blob rows have inner shape {values.shape[1:]}, expected {inner_shape}")
    if values.shape[0] == 0:
        raise ValueError(f"expected a non-empty batch of {inner_shape} rows, got 0 rows")
    rows = np.ascontiguousarray(values, dtype=np.dtype(spec.dtype))
    # Build the column from one contiguous data copy + uniform offsets, avoiding
    # an intermediate Python ``bytes`` per row (every row is the same width).
    n = rows.shape[0]
    row_nbytes = rows.dtype.itemsize * int(np.prod(inner_shape))
    data_buf = pa.py_buffer(rows.tobytes())
    offset_buf = pa.py_buffer((np.arange(n + 1, dtype=np.int64) * row_nbytes).tobytes())
    return pa.Array.from_buffers(pa.large_binary(), n, [None, offset_buf, data_buf])


def decode_blob_array(array: pa.Array | pa.ChunkedArray, spec: BlobFieldSpec) -> np.ndarray:
    """Decode a ``large_binary`` column into an owned ``(N, *spec.shape)`` ndarray.

    Reinterprets the column's contiguous value buffer in one shot — no per-row
    Python ``bytes`` and no view left pointing into Arrow memory — so it stays
    cheap and fork-safe inside DataLoader workers.

    :param array: ``large_binary`` column (one element per row); a
        ``ChunkedArray`` is combined first.
    :param spec: Per-row shape and dtype the column was written with.
    :returns: Writable ``(N, *spec.shape)`` array that owns its memory.
    :raises ValueError: ``array`` is not ``large_binary``, or its bytes are not an
        exact multiple of the per-row width (a corrupt or variable-width column).
    """
    dtype = np.dtype(spec.dtype)
    inner_shape = tuple(spec.shape)
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if not pa.types.is_large_binary(array.type):
        raise ValueError(f"expected a large_binary column, got {array.type}")
    n = len(array)
    if n == 0:
        return np.empty((0, *inner_shape), dtype=dtype)
    # Reads the value buffer directly, skipping the validity bitmap: BLOB columns
    # are written nullable=False, so every row is present and equal-width.
    _validity, offsets_buf, data_buf = array.buffers()
    # large_binary offsets are int64; honor the array's logical offset for slices.
    offsets = np.frombuffer(offsets_buf, dtype=np.int64)[array.offset : array.offset + n + 1]
    start, stop = int(offsets[0]), int(offsets[-1])
    expected_bytes = n * int(np.prod(inner_shape)) * dtype.itemsize
    if stop - start != expected_bytes:
        raise ValueError(
            f"blob column holds {stop - start} bytes, expected {expected_bytes} "
            f"for {n} rows of {inner_shape} {dtype}"
        )
    flat = np.frombuffer(
        data_buf, dtype=dtype, count=(stop - start) // dtype.itemsize, offset=start
    )
    return flat.reshape(n, *inner_shape).copy()


def tensor_array(values: np.ndarray, dtype: np.dtype, inner_shape: tuple[int, ...]) -> pa.Array:
    """Encode an ``(N, *inner_shape)`` ndarray as an Arrow fixed-shape tensor array.

    :param values: Rows to encode; cast to ``dtype`` and required to have shape
        ``(N, *inner_shape)`` with ``N >= 1`` (the leading axis is the row axis).
    :param dtype: Scalar dtype the values are cast to — the column's on-disk type.
    :param inner_shape: Schema per-row tensor shape, without the leading row axis.
    :returns: Arrow extension array compatible with :func:`lance_schema`.
    :raises ValueError: ``values`` inner axes differ from ``inner_shape``, or the batch is empty.
    """
    rows = np.ascontiguousarray(values, dtype=dtype)
    if rows.shape[1:] != inner_shape:
        raise ValueError(f"tensor rows have inner shape {rows.shape[1:]}, expected {inner_shape}")
    # Enforce N >= 1 with a clear message; the extension builder otherwise
    # rejects an empty batch with an opaque "non-empty ndarray" error.
    if rows.shape[0] == 0:
        raise ValueError(f"expected a non-empty batch of {inner_shape} tensors, got 0 rows")
    return pa.FixedShapeTensorArray.from_numpy_ndarray(rows)


def record_batch_from_arrays(
    arrays: dict[str, np.ndarray],
    schema: pa.Schema,
) -> pa.RecordBatch:
    """Build a Lance record batch from numpy arrays keyed by dataset field.

    :param arrays: Mapping with one ``(N, *inner)`` array per dataset field.
    :param schema: Schema returned by :func:`lance_schema`.
    :returns: Arrow record batch for :func:`write_lance_dataset` / :func:`lance_fragment`.
    """
    blob_specs = read_blob_field_specs(schema)
    columns = []
    for field in DATASET_FIELD_NAMES:
        if field in blob_specs:
            columns.append(blob_array(arrays[field], blob_specs[field]))
            continue
        # Read dtype and shape from the schema so an overridden field wins over
        # the global DATASET_FIELD_DTYPES default and the batch matches the file.
        tensor_type = schema.field(field).type
        np_dtype = np.dtype(tensor_type.value_type.to_pandas_dtype())
        columns.append(tensor_array(arrays[field], np_dtype, tuple(tensor_type.shape)))
    return pa.record_batch(columns, schema=schema)


def write_lance_dataset(
    uri: Path | str,
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    *,
    storage_options: dict[str, str] | None = None,
) -> None:
    """Write a Lance dataset from a pull source of pre-shaped record batches.

    Overwrites any dataset at ``uri`` (shards are immutable, never appended); the
    push-based worker loop uses :func:`lance_fragment` + :func:`commit_lance_dataset`.

    :param uri: Destination dataset directory (local path or ``s3://`` URI).
    :param schema: Arrow schema shared by every batch.
    :param batches: Record batches written in row order.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    """
    lance.write_dataset(
        iter(batches),
        str(uri),
        schema=schema,
        mode="overwrite",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=storage_options,
    )


def lance_fragment(
    uri: Path | str,
    schema: pa.Schema,
    batch: pa.RecordBatch,
    fragment_id: int,
    *,
    storage_options: dict[str, str] | None = None,
) -> lance.fragment.FragmentMetadata:
    """Write one record batch as a Lance fragment under ``uri`` (push source).

    Writes the data file immediately and returns its metadata; collect the results
    for :func:`commit_lance_dataset`. Streams batches without buffering a shard.

    :param uri: Destination dataset directory (local path or ``s3://`` URI).
    :param schema: Arrow schema shared by every fragment.
    :param batch: One record batch to persist as a fragment.
    :param fragment_id: Zero-based fragment index, contiguous within the dataset.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    :returns: Fragment metadata for the commit.
    """
    return LanceFragment.create(
        str(uri),
        batch,
        fragment_id=fragment_id,
        schema=schema,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=storage_options,
    )


def commit_lance_dataset(
    uri: Path | str,
    schema: pa.Schema,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    *,
    storage_options: dict[str, str] | None = None,
) -> None:
    """Commit fragments from :func:`lance_fragment` as a fresh Lance dataset.

    :param uri: Destination dataset directory (local path or ``s3://`` URI).
    :param schema: Arrow schema the dataset is created with.
    :param fragments: Fragment metadata from :func:`lance_fragment`, in row order.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    """
    operation = lance.LanceOperation.Overwrite(schema, list(fragments))
    lance.LanceDataset.commit(str(uri), operation, storage_options=storage_options)


def read_shard_metadata(schema: pa.Schema) -> ShardMetadata:
    """Parse ``ShardMetadata`` from Lance schema metadata.

    :param schema: Arrow schema read from a Lance file.
    :returns: Strict shard metadata payload.
    :raises ValueError: Metadata is absent or malformed.
    """
    payload = (schema.metadata or {}).get(SHARD_METADATA_SCHEMA_KEY)
    if payload is None:
        raise ValueError(f"missing schema metadata key {SHARD_METADATA_SCHEMA_KEY!r}")
    try:
        return ShardMetadata.model_validate_json(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid ShardMetadata: {exc}") from exc


def iter_lance_column_rows(
    uri: Path | str, column: str, *, storage_options: dict[str, str] | None = None
) -> Iterator[np.ndarray]:
    """Yield rows from one projected Lance column (sequential scan).

    Decodes a ``large_binary`` BLOB column via its embedded spec and a fixed-shape
    tensor column natively, so it serves both column kinds.

    :param uri: Shard dataset directory (local path or ``s3://`` URI).
    :param column: Column to project from the dataset.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    :yields: One ``(*inner_shape,)`` read-only row — copy before mutating.
    :ytype: np.ndarray
    """
    dataset = lance.dataset(str(uri), storage_options=storage_options)
    spec = read_blob_field_specs(dataset.schema).get(column)
    for batch in dataset.to_batches(columns=[column]):
        if spec is None:
            yield from batch.column(0).to_numpy_ndarray()
            continue
        decoded = decode_blob_array(batch.column(0), spec)
        # Read-only to mirror the tensor path's zero-copy contract; the owned
        # array stays alive via each row's ``base`` while the caller holds rows.
        decoded.setflags(write=False)
        yield from decoded
