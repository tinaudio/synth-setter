"""Lance single-file shard helpers shared by writer, validator, and finalize."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader, LanceFileWriter
from pydantic import ValidationError

from synth_setter.data.vst.shapes import DATASET_FIELD_DTYPES, DATASET_FIELD_NAMES
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

SHARD_METADATA_SCHEMA_KEY = b"synth_setter.shard_metadata"

# Variable-length per-row MP3 preview: inline ``large_binary``, not in DATASET_FIELD_NAMES.
MP3_AUDIO_FIELD = "audio_mp3"
_MP3_AUDIO_FIELD_METADATA: MappingProxyType[bytes, bytes] = MappingProxyType(
    {b"mime_type": b"audio/mpeg"}
)


def lance_schema(
    field_shapes: dict[str, tuple[int, ...]],
    metadata: ShardMetadata,
) -> pa.Schema:
    """Build the Arrow schema used by one Lance shard file.

    The fixed-shape tensor columns (``DATASET_FIELD_NAMES``) are followed by the
    ``audio_mp3`` ``large_binary`` preview column.

    :param field_shapes: Full writer shapes including the leading row axis.
    :param metadata: Per-shard render metadata to embed in schema metadata.
    :returns: Arrow schema with the tensor columns, the MP3 preview column, and shard metadata.
    """
    fields = []
    for field in DATASET_FIELD_NAMES:
        dtype = DATASET_FIELD_DTYPES[field]
        tensor_type = pa.fixed_shape_tensor(
            pa.from_numpy_dtype(dtype),
            field_shapes[field][1:],
        )
        fields.append(pa.field(field, tensor_type, nullable=False))
    fields.append(
        pa.field(
            MP3_AUDIO_FIELD, pa.large_binary(), nullable=False, metadata=_MP3_AUDIO_FIELD_METADATA
        )
    )
    return pa.schema(
        fields,
        metadata={SHARD_METADATA_SCHEMA_KEY: metadata.model_dump_json().encode("utf-8")},
    )


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
    mp3_bytes: list[bytes],
) -> pa.RecordBatch:
    """Build a Lance record batch from per-field tensors plus the MP3 preview column.

    :param arrays: Mapping with one ``(N, *inner)`` array per ``DATASET_FIELD_NAMES`` field.
    :param schema: Schema returned by :func:`lance_schema`.
    :param mp3_bytes: One MP3 blob per row, in row order; length must equal the
        tensor batch's ``N``.
    :returns: Arrow record batch ready for ``LanceFileWriter.write_batch``.
    :raises ValueError: ``mp3_bytes`` length differs from the tensor batch's ``N``.
    """
    columns: list[pa.Array] = []
    for field in DATASET_FIELD_NAMES:
        # Read dtype and shape from the schema so an overridden field wins over
        # the global DATASET_FIELD_DTYPES default and the batch matches the file.
        tensor_type = schema.field(field).type
        np_dtype = np.dtype(tensor_type.value_type.to_pandas_dtype())
        columns.append(tensor_array(arrays[field], np_dtype, tuple(tensor_type.shape)))
    # Explicit row-count guard: Arrow's own column-length error is opaque.
    n_rows = len(columns[0])
    if len(mp3_bytes) != n_rows:
        raise ValueError(f"mp3_bytes has {len(mp3_bytes)} rows, expected {n_rows}")
    columns.append(pa.array(mp3_bytes, type=pa.large_binary()))
    return pa.record_batch(columns, schema=schema)


def write_lance_file(
    path: Path | str, schema: pa.Schema, batches: Iterable[pa.RecordBatch]
) -> None:
    """Write a single Lance file from pre-shaped Arrow record batches.

    :param path: Destination ``.lance`` file.
    :param schema: Arrow schema for every batch.
    :param batches: Record batches to append in row order.
    """
    writer = LanceFileWriter(str(path), schema)
    try:
        for batch in batches:
            writer.write_batch(batch)
    finally:
        writer.close()


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


def iter_lance_column_rows(path: Path, column: str) -> Iterator[np.ndarray]:
    """Yield rows from one projected Lance tensor column.

    :param path: Local ``.lance`` shard path.
    :param column: Column to project from the Lance file.
    :yields: One ``(*inner_shape,)`` read-only view over Arrow's buffer — copy before mutating.
    :ytype: np.ndarray
    """
    reader = LanceFileReader(str(path), columns=[column])
    for batch in reader.read_all().to_batches():
        yield from batch.column(0).to_numpy_ndarray()
