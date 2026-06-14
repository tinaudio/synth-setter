"""Lance single-file shard helpers shared by writer, validator, and finalize."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.fragment import LanceFragment
from pydantic import ValidationError

from synth_setter.data.vst.shapes import DATASET_FIELD_DTYPES, DATASET_FIELD_NAMES
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

SHARD_METADATA_SCHEMA_KEY = b"synth_setter.shard_metadata"

# Pin the on-disk Lance file-format version so it never floats with the pylance
# default across upgrades; "2.1" equals the current default. See
# docs/design/lance-dataset-api-migration.md.
LANCE_DATA_STORAGE_VERSION = "2.1"


def lance_schema(
    field_shapes: dict[str, tuple[int, ...]],
    metadata: ShardMetadata,
) -> pa.Schema:
    """Build the Arrow schema used by one Lance shard file.

    :param field_shapes: Full writer shapes including the leading row axis.
    :param metadata: Per-shard render metadata to embed in schema metadata.
    :returns: Arrow schema with fixed-shape tensor columns and shard metadata.
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
    :returns: Arrow record batch ready for ``LanceFileWriter.write_batch``.
    """
    columns = []
    for field in DATASET_FIELD_NAMES:
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

    Streams ``batches`` into a fresh dataset, overwriting any dataset already at
    ``uri`` (shards/splits are immutable and rewritten, never appended). Use for
    sequential producers like finalize; the push-based worker render loop uses
    :func:`lance_fragment` + :func:`commit_lance_dataset` instead.

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
    uri: Path | str, schema: pa.Schema, batch: pa.RecordBatch, fragment_id: int
) -> lance.fragment.FragmentMetadata:
    """Write one record batch as a Lance fragment under ``uri`` (push source).

    Writes the fragment's data file immediately and returns its metadata; collect
    the results and hand them to :func:`commit_lance_dataset`. Lets the push-based
    render loop stream batches without buffering a whole shard in memory.

    :param uri: Destination dataset directory the fragment data file lands under.
    :param schema: Arrow schema shared by every fragment.
    :param batch: One record batch to persist as a fragment.
    :param fragment_id: Zero-based fragment index, contiguous within the dataset.
    :returns: Fragment metadata for the commit.
    """
    return LanceFragment.create(
        str(uri),
        batch,
        fragment_id=fragment_id,
        schema=schema,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )


def commit_lance_dataset(
    uri: Path | str,
    schema: pa.Schema,
    fragments: list[lance.fragment.FragmentMetadata],
) -> None:
    """Commit fragments from :func:`lance_fragment` as a fresh Lance dataset.

    :param uri: Destination dataset directory holding the fragment data files.
    :param schema: Arrow schema the dataset is created with.
    :param fragments: Fragment metadata from :func:`lance_fragment`, in row order.
    """
    operation = lance.LanceOperation.Overwrite(schema, fragments)
    lance.LanceDataset.commit(str(uri), operation)


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
    """Yield rows from one projected Lance tensor column (sequential scan).

    :param uri: Shard dataset directory (local path or ``s3://`` URI).
    :param column: Column to project from the dataset.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    :yields: One ``(*inner_shape,)`` read-only view over Arrow's buffer — copy before mutating.
    :ytype: np.ndarray
    """
    dataset = lance.dataset(str(uri), storage_options=storage_options)
    for batch in dataset.to_batches(columns=[column]):
        yield from batch.column(0).to_numpy_ndarray()
