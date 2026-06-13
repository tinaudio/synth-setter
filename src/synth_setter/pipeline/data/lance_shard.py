"""Lance single-file shard helpers shared by writer, validator, and finalize."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader, LanceFileWriter
from pydantic import ValidationError

from synth_setter.data.vst.shapes import AUDIO_FIELD, DATASET_FIELD_DTYPES, DATASET_FIELD_NAMES
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

SHARD_METADATA_SCHEMA_KEY = b"synth_setter.shard_metadata"

# Finalize-only audio preview: variable-length ``binary`` (encoded sizes differ
# per row, so not a fixed-shape tensor); the mime tag lets viewers detect MP3.
MP3_PREVIEW_FIELD: str = "audio_mp3"
_MP3_PREVIEW_FIELD_METADATA: Mapping[bytes, bytes] = MappingProxyType(
    {b"mime_type": b"audio/mpeg"}
)


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


def iter_lance_column_rows(path: Path, column: str) -> Iterator[np.ndarray]:
    """Yield rows from one projected Lance tensor column.

    :param path: Local ``.lance`` shard path.
    :param column: Column to project from the Lance file.
    :yields: One numpy tensor row at a time.
    :ytype: np.ndarray
    """
    reader = LanceFileReader(str(path), columns=[column])
    field = reader.metadata().schema.field(column)
    inner_shape = tuple(field.type.shape)
    for batch in reader.read_all().to_batches():
        array = tensor_chunk_to_numpy(batch.column(0), inner_shape)
        yield from array


def schema_with_mp3_preview(schema: pa.Schema) -> pa.Schema:
    """Append the :data:`MP3_PREVIEW_FIELD` binary column to a shard schema.

    :param schema: Shard schema carrying the tensor columns and shard metadata.
    :returns: ``schema`` with the MP3 preview field appended last; the embedded
        :class:`ShardMetadata` is preserved.
    """
    preview_field = pa.field(
        MP3_PREVIEW_FIELD,
        pa.binary(),
        nullable=False,
        metadata=_MP3_PREVIEW_FIELD_METADATA,
    )
    return schema.append(preview_field)


def _encode_audio_rows_to_mp3(audio_rows: np.ndarray, sample_rate: int) -> pa.Array:
    """Encode each ``(channels, samples)`` row to MP3, surfacing the failing row index.

    :param audio_rows: Decoded audio batch, shape ``(rows, channels, samples)``.
    :param sample_rate: Source sample rate in Hz, forwarded to the encoder.
    :returns: A binary Arrow array, one MP3 payload per row.
    :raises RuntimeError: Encoding a row failed; a partial column would otherwise
        truncate the shard with no diagnosable cause.
    """
    from synth_setter.pipeline.data.audio_preview import encode_mp3_preview

    payloads: list[bytes] = []
    for index, row in enumerate(audio_rows):
        try:
            payloads.append(encode_mp3_preview(row, sample_rate))
        except Exception as exc:  # noqa: BLE001 — re-raised with the row index for triage
            raise RuntimeError(f"failed to encode mp3 preview for audio row {index}") from exc
    return pa.array(payloads, type=pa.binary())


def append_mp3_preview_column(batch: pa.RecordBatch, sample_rate: int) -> pa.RecordBatch:
    """Return ``batch`` with an MP3 preview encoded from its ``audio`` column.

    :param batch: Shard record batch holding the ``audio`` tensor column.
    :param sample_rate: Source audio sample rate in Hz, forwarded to the encoder.
    :returns: A batch with one extra :data:`MP3_PREVIEW_FIELD` column whose rows
        align with ``audio``; every other column is carried through unchanged.
    :raises KeyError: ``batch`` has no ``audio`` column to encode.
    """
    audio_index = batch.schema.get_field_index(AUDIO_FIELD)
    if audio_index < 0:
        raise KeyError(f"batch has no {AUDIO_FIELD!r} column to build an mp3 preview from")
    audio_rows = tensor_chunk_to_numpy(
        batch.column(audio_index),
        tuple(batch.schema.field(audio_index).type.shape),
    )
    previews = _encode_audio_rows_to_mp3(audio_rows, sample_rate)
    return pa.RecordBatch.from_arrays(
        [*batch.columns, previews],
        schema=schema_with_mp3_preview(batch.schema),
    )
