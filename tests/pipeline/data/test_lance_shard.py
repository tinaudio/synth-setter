"""Write→read value-fidelity tests for the Lance shard codec.

The expected arrays are constructed directly in numpy — never through the
decoder under test — so a row-ordering or reshape bug in ``tensor_array`` /
``tensor_chunk_to_numpy`` cannot corrupt both sides identically and pass.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pedalboard.io import AudioFile

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
)
from synth_setter.pipeline.data.lance_shard import (
    MP3_PREVIEW_FIELD,
    append_mp3_preview_column,
    iter_lance_column_rows,
    lance_schema,
    read_shard_metadata,
    record_batch_from_arrays,
    schema_with_mp3_preview,
    write_lance_file,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

# Small distinct inner shapes so every element has a unique, exactly
# representable value (float16 is exact for integers up to 2048).
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

    write_lance_file(
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

    write_lance_file(shard, schema, [record_batch_from_arrays(arrays, schema)])

    decoded = np.stack(list(iter_lance_column_rows(shard, MEL_SPEC_FIELD)), axis=0)
    np.testing.assert_array_equal(decoded, transposed_mel)


# Inner audio shape (channels, samples); 441 samples at 44.1 kHz is a 10 ms
# preview — long enough to encode without exercising the resample path.
_PREVIEW_FIELD_SHAPES: dict[str, tuple[int, ...]] = {
    AUDIO_FIELD: (2, 2, 441),
    MEL_SPEC_FIELD: (2, 2, 3, 4),
    PARAM_ARRAY_FIELD: (2, 7),
}
_PREVIEW_METADATA = ShardMetadata(
    velocity=100,
    signal_duration_seconds=0.01,
    sample_rate=44100,
    channels=2,
    min_loudness=-55.0,
)


def _zero_preview_batch() -> pa.RecordBatch:
    """Build a two-row batch of zero-filled columns matching ``_PREVIEW_METADATA``.

    :returns: A record batch whose ``audio`` column has two stereo rows.
    """
    schema = lance_schema(_PREVIEW_FIELD_SHAPES, _PREVIEW_METADATA)
    arrays = {
        field: np.zeros(shape, dtype=DATASET_FIELD_DTYPES[field])
        for field, shape in _PREVIEW_FIELD_SHAPES.items()
    }
    return record_batch_from_arrays(arrays, schema)


def test_schema_with_mp3_preview_appends_binary_field() -> None:
    """The preview column is appended last as a binary field, after the tensor fields."""
    schema = lance_schema(_PREVIEW_FIELD_SHAPES, _PREVIEW_METADATA)

    augmented = schema_with_mp3_preview(schema)

    assert augmented.names == [*schema.names, MP3_PREVIEW_FIELD]
    assert pa.types.is_binary(augmented.field(MP3_PREVIEW_FIELD).type)


def test_schema_with_mp3_preview_preserves_shard_metadata() -> None:
    """Appending the preview field keeps the embedded ``ShardMetadata`` readable."""
    schema = lance_schema(_PREVIEW_FIELD_SHAPES, _PREVIEW_METADATA)

    augmented = schema_with_mp3_preview(schema)

    assert read_shard_metadata(augmented) == _PREVIEW_METADATA


def test_append_mp3_preview_column_adds_decodable_mp3_per_row() -> None:
    """Each appended row holds an MP3 that decodes back to the audio's channel count."""
    batch = _zero_preview_batch()

    augmented = append_mp3_preview_column(batch, _PREVIEW_METADATA.sample_rate)

    assert augmented.num_rows == 2
    mp3_column = augmented.column(augmented.schema.get_field_index(MP3_PREVIEW_FIELD))
    for row in range(2):
        with AudioFile(io.BytesIO(mp3_column[row].as_py())) as decoded:
            assert decoded.num_channels == 2


def test_append_mp3_preview_column_leaves_audio_column_unchanged() -> None:
    """Appending the preview never mutates the lossless ``audio`` tensor column."""
    batch = _zero_preview_batch()

    augmented = append_mp3_preview_column(batch, _PREVIEW_METADATA.sample_rate)

    original = batch.column(batch.schema.get_field_index(AUDIO_FIELD))
    carried = augmented.column(augmented.schema.get_field_index(AUDIO_FIELD))
    assert carried.to_pylist() == original.to_pylist()
