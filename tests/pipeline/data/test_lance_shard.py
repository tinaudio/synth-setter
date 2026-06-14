"""Write→read value-fidelity tests for the Lance shard codec.

The expected arrays are constructed directly in numpy — never through the
codec under test — so a row-ordering or reshape bug in ``tensor_array`` or the
tensor decode cannot corrupt both sides identically and pass.
"""

from __future__ import annotations

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
    LANCE_DATA_STORAGE_VERSION,
    commit_lance_dataset,
    iter_lance_column_rows,
    lance_fragment,
    lance_schema,
    record_batch_from_arrays,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

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


def test_iter_lance_column_rows_yields_read_only_views(tmp_path: Path) -> None:
    """Yielded rows share Arrow's read-only buffer, so callers must copy to mutate.

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


def test_record_batch_from_arrays_schema_dtype_wins_over_field_default() -> None:
    """Each column's dtype comes from the schema, not ``DATASET_FIELD_DTYPES``.

    ``audio`` defaults to float16, so a schema overriding it to float32 must
    yield a float32 column; sourcing the dtype from the global dict would emit
    float16 and fail ``pa.record_batch`` schema validation.
    """
    assert (
        DATASET_FIELD_DTYPES[AUDIO_FIELD] == np.float16
    )  # guards the override's discriminating power
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    field_index = schema.get_field_index(AUDIO_FIELD)
    float32_audio = pa.field(
        AUDIO_FIELD,
        pa.fixed_shape_tensor(pa.float32(), _FIELD_SHAPES[AUDIO_FIELD][1:]),
        nullable=False,
    )
    schema = schema.set(field_index, float32_audio)
    arrays = _arange_arrays(offset=0)
    arrays[AUDIO_FIELD] = arrays[AUDIO_FIELD].astype(np.float32)

    batch = record_batch_from_arrays(arrays, schema)

    assert batch.schema.field(AUDIO_FIELD).type.value_type == pa.float32()
