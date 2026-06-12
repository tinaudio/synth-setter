"""Write→read value-fidelity tests for the Lance shard codec.

The expected arrays are constructed directly in numpy — never through the
decoder under test — so a row-ordering or reshape bug in ``tensor_array`` /
``tensor_chunk_to_numpy`` cannot corrupt both sides identically and pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
)
from synth_setter.pipeline.data.lance_shard import (
    iter_lance_column_rows,
    lance_schema,
    record_batch_from_arrays,
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

# Carried opaquely by the codec — its values intentionally don't derive
# _FIELD_SHAPES, which the schema takes directly.
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
