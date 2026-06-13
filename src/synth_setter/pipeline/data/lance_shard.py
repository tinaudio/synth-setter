"""Lance single-file shard helpers shared by writer, validator, and finalize."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader, LanceFileWriter
from pydantic import ValidationError

from synth_setter.data.vst.shapes import DATASET_FIELD_DTYPES, DATASET_FIELD_NAMES
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

SHARD_METADATA_SCHEMA_KEY = b"synth_setter.shard_metadata"


def lance_schema(
    field_shapes: dict[str, tuple[int, ...]],
    metadata: ShardMetadata,
) -> pa.Schema:
    """Build the Arrow schema used by one Lance shard file.

    :param field_shapes: Full writer shapes including the leading row axis.
    :param metadata: Per-shard render metadata to embed in schema metadata.
    :returns: Arrow schema with fixed-shape tensor columns and shard metadata.
    :rtype: pa.Schema
    """
    fields = []
    for field in DATASET_FIELD_NAMES:
        dtype = DATASET_FIELD_DTYPES[field]
        tensor_type = pa.fixed_shape_tensor(
            pa.from_numpy_dtype(dtype),
            field_shapes[field][1:],
        )
        fields.append(pa.field(field, tensor_type, nullable=False))
    return pa.schema(
        fields,
        metadata={SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode("utf-8")},
    )


def tensor_array(values: np.ndarray, dtype: np.dtype, inner_shape: tuple[int, ...]) -> pa.Array:
    """Encode an ``(N, *inner_shape)`` ndarray as an Arrow fixed-shape tensor array.

    :param values: Batch values with a leading row axis.
    :param dtype: On-disk scalar dtype for the tensor values.
    :param inner_shape: Per-row tensor shape, excluding the leading row axis.
    :returns: Arrow extension array compatible with :func:`lance_schema`.
    :rtype: pa.Array
    :raises ValueError: ``values`` has an inner shape different from ``inner_shape``.
    """
    rows = np.asarray(values, dtype=dtype)
    if rows.ndim == len(inner_shape):
        rows = rows.reshape((1, *inner_shape))
    if rows.shape[1:] != inner_shape:
        raise ValueError(f"tensor rows have inner shape {rows.shape[1:]}, expected {inner_shape}")
    return pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(rows))


def record_batch_from_arrays(
    arrays: dict[str, np.ndarray],
    schema: pa.Schema,
) -> pa.RecordBatch:
    """Build a Lance record batch from numpy arrays keyed by dataset field.

    :param arrays: Mapping with one ``(N, *inner)`` array per dataset field.
    :param schema: Schema returned by :func:`lance_schema`.
    :returns: Arrow record batch ready for ``LanceFileWriter.write_batch``.
    :rtype: pa.RecordBatch
    """
    columns = []
    for field in DATASET_FIELD_NAMES:
        schema_field = schema.field(field)
        columns.append(
            tensor_array(
                arrays[field],
                DATASET_FIELD_DTYPES[field],
                tuple(schema_field.type.shape),
            )
        )
    return pa.record_batch(columns, schema=schema)


def write_lance_file(
    path: Path | str,
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    storage_options: dict[str, str] | None = None,
) -> None:
    """Write a single Lance file from pre-shaped Arrow record batches.

    Not atomic: a mid-stream failure closes the writer, leaving a partial object
    at ``path``; finalize's ``dataset.complete`` marker is the correctness anchor,
    and a re-run overwrites the partial split.

    :param path: Destination ``.lance`` file (local path or ``s3://`` URI).
    :param schema: Arrow schema for every batch.
    :param batches: Record batches to append in row order.
    :param storage_options: ``object_store`` kwargs (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`) when ``path`` is
        a cloud URI; ``None`` writes to local disk.
    """
    writer = LanceFileWriter(str(path), schema, storage_options=storage_options)
    try:
        for batch in batches:
            writer.write_batch(batch)
    finally:
        writer.close()


def read_shard_metadata(schema: pa.Schema) -> ShardMetadata:
    """Parse ``ShardMetadata`` from Lance schema metadata.

    :param schema: Arrow schema read from a Lance file.
    :returns: Strict shard metadata payload.
    :rtype: ShardMetadata
    :raises ValueError: Metadata is absent or malformed.
    """
    payload = (schema.metadata or {}).get(SHARD_METADATA_SCHEMA_KEY)
    if payload is None:
        raise ValueError(f"missing schema metadata key {SHARD_METADATA_SCHEMA_KEY!r}")
    try:
        return ShardMetadata.model_validate_json(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid ShardMetadata: {exc}") from exc


def tensor_chunk_to_numpy(chunk: pa.Array, inner_shape: tuple[int, ...]) -> np.ndarray:
    """Decode one fixed-shape tensor Arrow chunk to ``(N, *inner_shape)`` numpy.

    :param chunk: Arrow extension array chunk from a Lance tensor column.
    :param inner_shape: Per-row tensor shape from the schema.
    :returns: Decoded numpy array.
    :rtype: np.ndarray
    :raises TypeError: ``chunk`` is not backed by fixed-size-list tensor storage.
    """
    storage = getattr(chunk, "storage", None)
    if storage is None or not pa.types.is_fixed_size_list(storage.type):
        raise TypeError(f"expected fixed-size-list tensor storage, got {chunk.type}")
    values = storage.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(chunk), *inner_shape)


def iter_lance_column_rows(
    path: Path | str, column: str, storage_options: dict[str, str] | None = None
) -> Iterator[np.ndarray]:
    """Yield rows from one projected Lance tensor column.

    :param path: ``.lance`` shard path (local path or ``s3://`` URI).
    :param column: Column to project from the Lance file.
    :param storage_options: ``object_store`` kwargs when ``path`` is a cloud
        URI; ``None`` reads from local disk.
    :yields: One numpy tensor row at a time.
    :ytype: np.ndarray
    """
    reader = LanceFileReader(str(path), columns=[column], storage_options=storage_options)
    field = reader.metadata().schema.field(column)
    inner_shape = tuple(field.type.shape)
    for batch in reader.read_all().to_batches():
        array = tensor_chunk_to_numpy(batch.column(0), inner_shape)
        yield from array
