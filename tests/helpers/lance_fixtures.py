"""Shared writer for Lance shard test fixtures."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa

from synth_setter.pipeline.data.lance_shard import tensor_array, write_lance_file


def write_lance_shard(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    """Write ``columns`` as a single-file Lance shard with one fixed-shape tensor column each.

    Goes through the pipeline's :func:`write_lance_file` so fixtures carry the
    exact on-disk format the finalize step emits.

    :param path: Output ``.lance`` shard file.
    :param columns: Mapping of column name to ``(num_rows, ...)`` array.
    """
    fields = [
        pa.field(
            name,
            pa.fixed_shape_tensor(pa.from_numpy_dtype(data.dtype), data.shape[1:]),
            nullable=False,
        )
        for name, data in columns.items()
    ]
    schema = pa.schema(fields)
    batch = pa.record_batch(
        [tensor_array(data, data.dtype, data.shape[1:]) for data in columns.values()],
        schema=schema,
    )
    write_lance_file(path, schema, [batch])
