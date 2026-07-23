"""Lance dataset-shard helpers shared by writer, validator, and finalize."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader
from pydantic import ValidationError

from synth_setter.data.vst.seeding import seed_for_sample
from synth_setter.data.vst.shapes import DATASET_FIELD_DTYPES, DATASET_FIELD_NAMES, DEBUG_FIELD
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

SHARD_METADATA_SCHEMA_KEY = b"synth_setter.shard_metadata"
DEBUG_JSON_TYPE = pa.json_()
_LANCE_JSON_FIELD_METADATA = {
    b"ARROW:extension:name": b"lance.json",
    b"ARROW:extension:metadata": b"",
}

# Pin the Lance on-disk format instead of floating with the pylance default;
# "2.2" leads that default and needs a reader new enough to open it (#1714).
LANCE_DATA_STORAGE_VERSION = "2.2"
# Refs https://github.com/tinaudio/synth-setter/issues/1775: keep one data file
# below S3's 10k multipart-part ceiling even at 5 MiB parts.
LANCE_MAX_BYTES_PER_FILE = 32 * 1024**3


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
    # DuckDB scans reserve STANDARD_VECTOR_SIZE (2048 rows) x flattened width for every
    # fixed-shape-tensor column; audio and mel_spec can OOM SmooSense's 3 GB memory_limit (#1704).
    for field in DATASET_FIELD_NAMES:
        dtype = DATASET_FIELD_DTYPES[field]
        tensor_type = pa.fixed_shape_tensor(
            pa.from_numpy_dtype(dtype),
            field_shapes[field][1:],
        )
        fields.append(pa.field(field, tensor_type, nullable=False))
    fields.append(pa.field(DEBUG_FIELD, DEBUG_JSON_TYPE, nullable=False))
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


def seed_debug_array(
    master_seed: int,
    sample_indices: Sequence[int],
    attempts: Sequence[int],
    *,
    shard_id: int | None,
    parameter_sample_idx: int | None = None,
    parameter_attempt: int | None = None,
    parameter_source: str = "sampled",
) -> pa.Array:
    """Build row-level seed provenance as Arrow JSON documents.

    :param master_seed: Dataset or split master seed shared by these rows.
    :param sample_indices: Stable logical row indices within the seed stream.
    :param attempts: Accepted loudness-gate attempt for each row.
    :param shard_id: Logical shard number, or ``None`` for an ad hoc render.
    :param parameter_sample_idx: Seed-stream row that supplied a shard-cadence patch.
    :param parameter_attempt: Accepted attempt that supplied a shard-cadence patch.
    :param parameter_source: Whether parameters were sampled, fixed, or mixed.
    :returns: JSON documents containing each concrete seed and its derivation inputs.
    :raises ValueError: ``sample_indices`` and ``attempts`` have different lengths.
    """
    if len(sample_indices) != len(attempts):
        raise ValueError(
            f"sample_indices has length {len(sample_indices)}, attempts has length {len(attempts)}"
        )
    if (parameter_sample_idx is None) != (parameter_attempt is None):
        raise ValueError("parameter_sample_idx and parameter_attempt must be provided together")

    documents = []
    for sample_idx, attempt in zip(sample_indices, attempts, strict=True):
        document = {
            "seed": seed_for_sample(master_seed, sample_idx, attempt),
            "master_seed": master_seed,
            "sample_idx": sample_idx,
            "attempt": attempt,
            "shard_id": shard_id,
            "parameter_source": parameter_source,
        }
        if parameter_sample_idx is not None and parameter_attempt is not None:
            document.update(
                parameter_seed=seed_for_sample(
                    master_seed, parameter_sample_idx, parameter_attempt
                ),
                parameter_sample_idx=parameter_sample_idx,
                parameter_attempt=parameter_attempt,
            )
        documents.append(json.dumps(document, separators=(",", ":")))
    return pa.array(documents, type=DEBUG_JSON_TYPE)


def record_batch_from_arrays(
    arrays: dict[str, np.ndarray],
    schema: pa.Schema,
    *,
    debug: pa.Array | None,
) -> pa.RecordBatch:
    """Build a Lance record batch from numpy arrays keyed by dataset field.

    :param arrays: Mapping with one ``(N, *inner)`` array per dataset field.
    :param schema: Schema returned by :func:`lance_schema`.
    :param debug: Row-level seed provenance; ``None`` writes empty documents for fixtures.
    :returns: Arrow record batch for :func:`write_lance_dataset` / :func:`lance_fragment`.
    """
    columns = []
    for field in DATASET_FIELD_NAMES:
        # Read dtype and shape from the schema so an overridden field wins over
        # the global DATASET_FIELD_DTYPES default and the batch matches the file.
        tensor_type = schema.field(field).type
        np_dtype = np.dtype(tensor_type.value_type.to_pandas_dtype())
        columns.append(tensor_array(arrays[field], np_dtype, tuple(tensor_type.shape)))
    if debug is None:
        debug = pa.repeat(pa.scalar("{}", type=DEBUG_JSON_TYPE), len(columns[0]))
    columns.append(debug)
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
        max_bytes_per_file=LANCE_MAX_BYTES_PER_FILE,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=storage_options,
    )


def _decode_metadata_value(value: bytes | None) -> str:
    """Render one schema-metadata value for a mismatch message.

    :param value: Raw metadata bytes, or ``None`` when the key is absent on one side.
    :returns: Decoded text, or ``<absent>`` for a missing key.
    """
    return "<absent>" if value is None else value.decode("utf-8", errors="replace")


def schema_mismatch_detail(physical: pa.Schema, expected: pa.Schema) -> str:
    """Describe how a data file's physical schema diverges from the intended one.

    :param physical: Schema read from the written or staged data file.
    :param expected: Schema the current code derives from the dataset spec.
    :returns: Field and metadata differences; when the fields agree and only spec-derived schema
        metadata diverges, ends with a code-version-skew hint (the #2084 signature: writer and
        validator on different code).
    """
    parts: list[str] = []
    physical_fields = {field.name: field.type for field in physical}
    expected_fields = {field.name: field.type for field in expected}
    only_physical = sorted(set(physical_fields) - set(expected_fields))
    only_expected = sorted(set(expected_fields) - set(physical_fields))
    if only_physical:
        parts.append(f"fields only in fragment: {only_physical}")
    if only_expected:
        parts.append(f"fields only in expected: {only_expected}")
    type_diffs = [
        f"{name} fragment {physical_fields[name]} vs expected {expected_fields[name]}"
        for name in sorted(set(physical_fields) & set(expected_fields))
        if physical_fields[name] != expected_fields[name]
    ]
    if type_diffs:
        parts.append("field types differ: " + "; ".join(type_diffs))
    physical_meta = dict(physical.metadata or {})
    expected_meta = dict(expected.metadata or {})
    for key in sorted(set(physical_meta) | set(expected_meta)):
        physical_value, expected_value = physical_meta.get(key), expected_meta.get(key)
        if physical_value != expected_value:
            parts.append(
                f"metadata {key.decode('utf-8', errors='replace')!r}: "
                f"fragment={_decode_metadata_value(physical_value)} "
                f"expected={_decode_metadata_value(expected_value)}"
            )
    if not (only_physical or only_expected or type_diffs) and physical_meta != expected_meta:
        parts.append(
            "fields agree and only spec-derived schema metadata differs — likely "
            "writer/validator code-version skew (fragments staged by a different code "
            "version than this checkout, e.g. CI's shared dev-snapshot writer image vs "
            "a stale PR branch); rebase onto current main or regenerate the fragments"
        )
    # Name/type/metadata diffs above are order-insensitive and ignore field
    # flags, so an order- or nullability-only drift needs its own rendering.
    if not parts:
        parts.append(
            "fields differ in order or nullability: "
            f"fragment [{', '.join(str(field) for field in physical)}] vs "
            f"expected [{', '.join(str(field) for field in expected)}]"
        )
    return "; ".join(parts)


def _logical_fragment_schema(physical: pa.Schema, logical: pa.Schema) -> pa.Schema:
    """Restore logical JSON typing on a physical fragment schema.

    :param physical: Schema read directly from a fragment data file.
    :param logical: Schema supplied to the fragment writer and dataset commit.
    :returns: Physical schema with Lance's JSON storage field restored to its logical type.
    """
    index = logical.get_field_index(DEBUG_FIELD)
    if index < 0 or physical.names != logical.names:
        return physical
    physical_debug = physical.field(index)
    logical_debug = logical.field(index)
    if (
        logical_debug.type != DEBUG_JSON_TYPE
        or physical_debug.type != pa.large_binary()
        or physical_debug.metadata != _LANCE_JSON_FIELD_METADATA
    ):
        return physical
    restored_debug = pa.field(
        physical_debug.name,
        logical_debug.type,
        nullable=physical_debug.nullable,
        metadata=logical_debug.metadata,
    )
    return physical.set(index, restored_debug)


def fragment_schema_matches(physical: pa.Schema, logical: pa.Schema) -> bool:
    """Return whether a fragment file matches its logical dataset schema.

    Lance stores Arrow JSON as ``large_binary`` plus field metadata in fragment
    files, then restores the JSON extension from the committed dataset schema.

    :param physical: Schema read directly from a fragment data file.
    :param logical: Schema supplied to the fragment writer and dataset commit.
    :returns: Whether all logical fields have the expected physical representation.
    """
    return _logical_fragment_schema(physical, logical).equals(logical, check_metadata=True)


def fragment_schema_mismatch_detail(physical: pa.Schema, logical: pa.Schema) -> str:
    """Describe a fragment mismatch after restoring logical JSON typing.

    :param physical: Schema read directly from a fragment data file.
    :param logical: Schema supplied to the fragment writer and dataset commit.
    :returns: Field and metadata differences suitable for an operator error.
    """
    return schema_mismatch_detail(_logical_fragment_schema(physical, logical), logical)


def lance_fragment(
    uri: Path | str,
    schema: pa.Schema,
    batch: pa.RecordBatch | Iterable[pa.RecordBatch],
    *,
    storage_options: dict[str, str] | None = None,
) -> lance.fragment.FragmentMetadata:
    """Write record batches as one Lance fragment under ``uri`` (push source).

    Writes the data file immediately and returns its metadata; collect the results
    for :func:`commit_lance_dataset`. Streams batches without buffering a shard.
    Lance's object-store client retries individual requests with bounded backoff;
    replaying this whole call could consume the stream twice and leak fragments.

    :param uri: Destination dataset directory (local path or ``s3://`` URI).
    :param schema: Arrow schema shared by every fragment.
    :param batch: One record batch — or an iterable of them, streamed into a
        single fragment — to persist.
    :param storage_options: Object-store config for a cloud ``uri`` (see
        :func:`synth_setter.pipeline.r2_io.r2_storage_options`); ``None`` local.
    :returns: Fragment metadata for the commit.
    :raises ValueError: Lance splits the input into more than one fragment, or
        the written file's physical schema differs from ``schema`` (append mode
        adopts an existing committed dataset's schema — #2084).
    """
    fragments = lance.fragment.write_fragments(
        batch,
        str(uri),
        schema=schema,
        mode="append",
        max_bytes_per_file=LANCE_MAX_BYTES_PER_FILE,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=storage_options,
    )
    if len(fragments) != 1:
        raise ValueError(
            f"expected one Lance fragment under {uri}, wrote {len(fragments)}; "
            "reduce the render batch or samples per shard"
        )
    fragment = fragments[0]
    physical = (
        LanceFileReader(f"{uri}/data/{fragment.files[0].path}", storage_options=storage_options)
        .metadata()
        .schema
    )
    if not fragment_schema_matches(physical, schema):
        raise ValueError(
            f"fragment under {uri} was written with the existing dataset's schema, not the "
            "spec-derived one (Lance append mode adopts a committed dataset's schema; the "
            f"target likely holds stale data from an older code version — #2084): "
            f"{fragment_schema_mismatch_detail(physical, schema)}"
        )
    return fragment


def commit_lance_dataset(
    uri: Path | str,
    schema: pa.Schema,
    fragments: Sequence[lance.fragment.FragmentMetadata],
    *,
    storage_options: dict[str, str] | None = None,
) -> None:
    """Commit fragments from :func:`lance_fragment` as a fresh Lance dataset.

    Lance's object-store client supplies bounded request retries; the commit is
    not replayed here because its success may be ambiguous after a lost response.

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
