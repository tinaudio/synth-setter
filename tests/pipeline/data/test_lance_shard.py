"""Write→read value-fidelity tests for the Lance shard codec.

The expected arrays are constructed directly in numpy — never through the
codec under test — so a row-ordering or reshape bug in ``tensor_array`` or the
tensor decode cannot corrupt both sides identically and pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
)
from synth_setter.pipeline.data.lance_shard import (
    BLOB_FIELD_SPECS_SCHEMA_KEY,
    LANCE_DATA_STORAGE_VERSION,
    blob_array,
    commit_lance_dataset,
    decode_blob_array,
    iter_lance_column_rows,
    lance_fragment,
    lance_schema,
    read_blob_field_specs,
    record_batch_from_arrays,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.shard_metadata import BlobFieldSpec, ShardMetadata

# Small shapes: each element gets a unique value, exactly representable as float16 (<= 2048).
_FIELD_SHAPES: dict[str, tuple[int, ...]] = {
    AUDIO_FIELD: (2, 2, 5),
    MEL_SPEC_FIELD: (2, 2, 3, 4),
    PARAM_ARRAY_FIELD: (2, 7),
}

# Opaque payload to the codec — its values needn't match _FIELD_SHAPES, which
# the schema takes directly.
_METADATA = ShardMetadata(
    velocity=100,
    signal_duration_seconds=1.0,
    sample_rate=100,
    channels=2,
    min_loudness=-55.0,
)


def _arange_arrays(offset: int) -> dict[str, np.ndarray]:
    """Build one batch of distinct per-field arrays starting at ``offset``.

    :param offset: First value of every field's ``arange`` so two batches
        never share element values.
    :returns: Mapping keyed by ``DATASET_FIELD_NAMES`` with writer dtypes.
    """
    return {
        field: np.arange(
            offset, offset + np.prod(shape), dtype=DATASET_FIELD_DTYPES[field]
        ).reshape(shape)
        for field, shape in _FIELD_SHAPES.items()
    }


@pytest.mark.parametrize("field", DATASET_FIELD_NAMES)
def test_lance_round_trip_two_batches_preserves_values_and_row_order(
    field: str, tmp_path: Path
) -> None:
    """Distinct known values written in two batches read back exactly, in order.

    :param field: Writer dataset field whose column is decoded and compared.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    first = _arange_arrays(offset=0)
    second = _arange_arrays(offset=1000)
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    shard = tmp_path / "shard-000000.lance"

    write_lance_dataset(
        shard,
        schema,
        [record_batch_from_arrays(first, schema), record_batch_from_arrays(second, schema)],
    )

    decoded = np.stack(list(iter_lance_column_rows(shard, field)), axis=0)
    expected = np.concatenate([first[field], second[field]], axis=0)
    np.testing.assert_array_equal(decoded, expected)
    assert decoded.dtype == DATASET_FIELD_DTYPES[field]


def test_lance_round_trip_noncontiguous_transposed_input_preserves_values(
    tmp_path: Path,
) -> None:
    """A transposed (non-contiguous) mel input decodes to its logical values.

    The writer receives ``mel_spec`` rows transposed from their allocation
    order — the layout ``_sample_batch_arrays`` produces — so a codec that
    serialized raw buffer order instead of logical order would scramble
    values while keeping shapes intact.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    n, channels, n_mels, n_frames = _FIELD_SHAPES[MEL_SPEC_FIELD]
    transposed_mel = (
        np.arange(n * channels * n_mels * n_frames, dtype=np.float32)
        .reshape(n, n_mels, channels, n_frames)
        .transpose(0, 2, 1, 3)
    )
    assert not transposed_mel.flags["C_CONTIGUOUS"]
    arrays = _arange_arrays(offset=0)
    arrays[MEL_SPEC_FIELD] = transposed_mel
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    shard = tmp_path / "shard-000000.lance"

    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    decoded = np.stack(list(iter_lance_column_rows(shard, MEL_SPEC_FIELD)), axis=0)
    np.testing.assert_array_equal(decoded, transposed_mel)


def test_iter_lance_column_rows_yields_read_only_rows(tmp_path: Path) -> None:
    """Yielded rows are read-only, so callers must copy before mutating.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    shard = tmp_path / "shard-000000.lance"
    write_lance_dataset(
        shard, schema, [record_batch_from_arrays(_arange_arrays(offset=0), schema)]
    )

    row = next(iter_lance_column_rows(shard, AUDIO_FIELD))

    assert not row.flags.writeable


def test_write_lance_dataset_pins_data_storage_version(tmp_path: Path) -> None:
    """The written dataset reports the pinned on-disk format version, not the library default.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    shard = tmp_path / "shard-000000.lance"
    write_lance_dataset(
        shard, schema, [record_batch_from_arrays(_arange_arrays(offset=0), schema)]
    )

    assert lance.dataset(str(shard)).data_storage_version == LANCE_DATA_STORAGE_VERSION


def test_lance_fragment_commit_round_trips_values_and_pins_version(tmp_path: Path) -> None:
    """Push-path fragments commit into one dataset preserving rows, order, and the pinned version.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    first = _arange_arrays(offset=0)
    second = _arange_arrays(offset=1000)
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    shard = tmp_path / "shard-000000.lance"

    fragments = [
        lance_fragment(shard, schema, record_batch_from_arrays(first, schema), 0),
        lance_fragment(shard, schema, record_batch_from_arrays(second, schema), 1),
    ]
    commit_lance_dataset(shard, schema, fragments)

    dataset = lance.dataset(str(shard))
    assert dataset.count_rows() == 2 * _FIELD_SHAPES[AUDIO_FIELD][0]
    assert dataset.data_storage_version == LANCE_DATA_STORAGE_VERSION
    decoded = np.stack(list(iter_lance_column_rows(shard, MEL_SPEC_FIELD)), axis=0)
    expected = np.concatenate([first[MEL_SPEC_FIELD], second[MEL_SPEC_FIELD]], axis=0)
    np.testing.assert_array_equal(decoded, expected)


def test_tensor_array_missing_row_axis_raises_value_error() -> None:
    """A tensor whose ndim equals ``inner_shape``'s (no row axis) raises ValueError.

    ``(2, 7)`` under inner shape ``(2, 7)`` is rejected, not read as a single
    row of shape ``(2, 7)``.
    """
    with pytest.raises(ValueError, match=r"inner shape .+, expected \(2, 7\)"):
        tensor_array(np.zeros((2, 7), dtype=np.float16), np.dtype(np.float16), (2, 7))


def test_tensor_array_empty_batch_raises_value_error() -> None:
    """A correctly-shaped but row-empty batch raises a clear ValueError, not Arrow's.

    ``(0, 2, 7)`` passes the inner-shape check, so the explicit N >= 1 guard —
    not the opaque extension-builder error — is what must fire.
    """
    with pytest.raises(ValueError, match=r"non-empty batch .* got 0 rows"):
        tensor_array(np.zeros((0, 2, 7), dtype=np.float16), np.dtype(np.float16), (2, 7))


def test_record_batch_from_arrays_tensor_dtype_comes_from_schema_not_field_default() -> None:
    """A tensor column's dtype comes from the schema, not ``DATASET_FIELD_DTYPES``.

    ``param_array`` defaults to float32, so a schema overriding it to float16
    must yield a float16 column; sourcing the dtype from the global dict would
    emit float32 and fail ``pa.record_batch`` schema validation.
    """
    assert (
        DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD] == np.float32
    )  # guards the override's discriminating power
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    field_index = schema.get_field_index(PARAM_ARRAY_FIELD)
    float16_params = pa.field(
        PARAM_ARRAY_FIELD,
        pa.fixed_shape_tensor(pa.float16(), _FIELD_SHAPES[PARAM_ARRAY_FIELD][1:]),
        nullable=False,
    )
    schema = schema.set(field_index, float16_params)
    arrays = _arange_arrays(offset=0)
    arrays[PARAM_ARRAY_FIELD] = arrays[PARAM_ARRAY_FIELD].astype(np.float16)

    batch = record_batch_from_arrays(arrays, schema)

    assert batch.schema.field(PARAM_ARRAY_FIELD).type.value_type == pa.float16()


def test_lance_schema_stores_blob_fields_as_large_binary_and_param_array_as_tensor() -> None:
    """``audio``/``mel_spec`` become ``large_binary`` BLOB columns; ``param_array`` stays a tensor.

    Pins the property that fixes the SmooSense/DuckDB OOM: a tensor column makes the lance reader
    pre-allocate a full chunk sized by the tensor width, while a BLOB column does not.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)

    assert pa.types.is_large_binary(schema.field(AUDIO_FIELD).type)
    assert pa.types.is_large_binary(schema.field(MEL_SPEC_FIELD).type)
    assert isinstance(schema.field(PARAM_ARRAY_FIELD).type, pa.FixedShapeTensorType)


def test_read_blob_field_specs_round_trips_inner_shape_and_dtype() -> None:
    """The embedded BLOB specs recover each blob column's inner shape and dtype.

    A ``large_binary`` Arrow type carries neither, so the reader and validator
    depend on this metadata to decode and check the column.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)

    specs = read_blob_field_specs(schema)

    assert set(specs) == {AUDIO_FIELD, MEL_SPEC_FIELD}
    assert tuple(specs[AUDIO_FIELD].shape) == _FIELD_SHAPES[AUDIO_FIELD][1:]
    assert specs[AUDIO_FIELD].dtype == DATASET_FIELD_DTYPES[AUDIO_FIELD].name
    assert tuple(specs[MEL_SPEC_FIELD].shape) == _FIELD_SHAPES[MEL_SPEC_FIELD][1:]
    assert specs[MEL_SPEC_FIELD].dtype == DATASET_FIELD_DTYPES[MEL_SPEC_FIELD].name


def test_read_blob_field_specs_returns_empty_for_legacy_schema_without_key() -> None:
    """A schema with no embedded BLOB specs (legacy all-tensor shard) yields ``{}``.

    Absence is benign — such columns are fixed-shape tensors decoded natively — so the reader must
    not raise on it.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA).remove_metadata()

    assert read_blob_field_specs(schema) == {}


def test_blob_array_empty_batch_raises_value_error() -> None:
    """A correctly-shaped but row-empty blob batch raises a clear ValueError.

    Mirrors ``tensor_array``'s N >= 1 guard so an empty shard fails loudly at
    write time, not with opaque bytes downstream.
    """
    spec = BlobFieldSpec(shape=[2, 5], dtype="float16")
    with pytest.raises(ValueError, match=r"non-empty batch .* got 0 rows"):
        blob_array(np.zeros((0, 2, 5), dtype=np.float16), spec)


def test_blob_array_wrong_inner_shape_raises_value_error() -> None:
    """Rows whose inner shape differs from the spec raise ValueError, naming both shapes."""
    spec = BlobFieldSpec(shape=[2, 5], dtype="float16")
    with pytest.raises(ValueError, match=r"inner shape \(2, 6\), expected \(2, 5\)"):
        blob_array(np.zeros((3, 2, 6), dtype=np.float16), spec)


@pytest.mark.parametrize(
    ("num_rows", "inner_shape", "dtype"),
    [
        (1, (4,), "float16"),  # single row
        (3, (2, 3), "float32"),
        (3, (2, 176400), "float16"),  # production audio
        (3, (2, 128, 401), "float32"),  # production mel
    ],
)
def test_blob_array_decode_round_trips_values_shape_and_dtype(
    num_rows: int, inner_shape: tuple[int, ...], dtype: str
) -> None:
    """``blob_array`` then ``decode_blob_array`` recovers rows for production and edge shapes.

    Exercises the codec directly (not via ``iter_lance_column_rows``) so a reshape
    or dtype bug surfaces in isolation; the decoded array must own its memory.
    Random data round-trips byte-exactly, sidestepping float16 range limits.

    :param num_rows: Row count under test (1 exercises the single-row edge).
    :param inner_shape: Per-row inner shape under test.
    :param dtype: Stored numpy dtype under test.
    """
    spec = BlobFieldSpec(shape=list(inner_shape), dtype=dtype)
    rows = np.random.default_rng(0).standard_normal((num_rows, *inner_shape)).astype(dtype)

    decoded = decode_blob_array(blob_array(rows, spec), spec)

    assert decoded.shape == (num_rows, *inner_shape)
    assert decoded.dtype == np.dtype(dtype)
    assert decoded.flags.writeable
    np.testing.assert_array_equal(decoded, rows)


def test_decode_blob_array_handles_sliced_arrow_offset() -> None:
    """Decoding a sliced (non-zero ``array.offset``) ``large_binary`` array reads the right rows."""
    spec = BlobFieldSpec(shape=[2, 3], dtype="float16")
    rows = np.arange(4 * 2 * 3, dtype=np.float16).reshape(4, 2, 3)

    decoded = decode_blob_array(blob_array(rows, spec).slice(1, 2), spec)

    np.testing.assert_array_equal(decoded, rows[1:3])


def test_decode_blob_array_combines_chunked_array() -> None:
    """A ``ChunkedArray`` of BLOB rows is combined and decoded in row order."""
    spec = BlobFieldSpec(shape=[2, 3], dtype="float16")
    first = np.arange(2 * 3, dtype=np.float16).reshape(1, 2, 3)
    second = np.arange(100, 100 + 2 * 3, dtype=np.float16).reshape(1, 2, 3)
    chunked = pa.chunked_array([blob_array(first, spec), blob_array(second, spec)])

    decoded = decode_blob_array(chunked, spec)

    np.testing.assert_array_equal(decoded, np.concatenate([first, second], axis=0))


def test_decode_blob_array_byte_count_mismatch_raises_value_error() -> None:
    """A spec whose width disagrees with the stored bytes raises rather than mis-decoding."""
    written = BlobFieldSpec(shape=[2, 3], dtype="float16")
    column = blob_array(np.zeros((2, 2, 3), dtype=np.float16), written)
    wrong = BlobFieldSpec(shape=[2, 4], dtype="float16")
    with pytest.raises(ValueError, match="blob column holds"):
        decode_blob_array(column, wrong)


def test_read_blob_field_specs_malformed_payload_raises_value_error() -> None:
    """A blob-specs metadata payload that is not valid JSON raises a clear ValueError.

    The payload is read at shard-open time off an R2 trust boundary, so a corrupt entry must fail
    loudly rather than crash later in decode.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA).with_metadata(
        {BLOB_FIELD_SPECS_SCHEMA_KEY: b"not json"}
    )
    with pytest.raises(ValueError, match="invalid blob field specs"):
        read_blob_field_specs(schema)


def test_read_blob_field_specs_rejects_spec_for_non_blob_column() -> None:
    """A spec naming a fixed-shape tensor column is rejected, not silently misclassified.

    Consumers treat a column as BLOB by its presence in this mapping, so a
    tampered spec for a tensor column (e.g. ``param_array``) must fail loudly.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    tampered = dict(json.loads(schema.metadata[BLOB_FIELD_SPECS_SCHEMA_KEY]))
    tampered[PARAM_ARRAY_FIELD] = {
        "shape": list(_FIELD_SHAPES[PARAM_ARRAY_FIELD][1:]),
        "dtype": "float32",
    }
    schema = schema.with_metadata(
        {**schema.metadata, BLOB_FIELD_SPECS_SCHEMA_KEY: json.dumps(tampered).encode("utf-8")}
    )
    with pytest.raises(ValueError, match="non-large_binary column 'param_array'"):
        read_blob_field_specs(schema)


def test_read_blob_field_specs_non_object_json_raises_value_error() -> None:
    """A payload that is valid JSON but not an object (e.g. a list) raises a clear ValueError.

    Guards the ``.items()`` call from an ``AttributeError`` escaping the trust boundary.
    """
    schema = lance_schema(_FIELD_SHAPES, _METADATA).with_metadata(
        {BLOB_FIELD_SPECS_SCHEMA_KEY: b"[1, 2, 3]"}
    )
    with pytest.raises(ValueError, match="invalid blob field specs"):
        read_blob_field_specs(schema)
