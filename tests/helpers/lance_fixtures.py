"""Shared writer for Lance shard test fixtures."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa

from synth_setter.data.vst.shapes import BLOB_FIELDS
from synth_setter.pipeline.data.lance_shard import (
    BLOB_FIELD_SPECS_SCHEMA_KEY,
    blob_array,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.shard_metadata import BlobFieldSpec


def write_lance_shard(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    """Write ``columns`` as a Lance dataset directory matching the production format.

    Goes through the pipeline's :func:`write_lance_dataset` so fixtures carry the
    exact on-disk format the finalize step emits: columns in ``BLOB_FIELDS`` (e.g.
    ``audio`` / ``mel_spec``) become ``large_binary`` BLOBs with embedded specs,
    everything else stays a fixed-shape tensor.

    :param path: Output ``.lance`` dataset directory.
    :param columns: Mapping of column name to ``(num_rows, ...)`` array.
    """
    fields = []
    arrays = []
    blob_specs: dict[str, dict[str, object]] = {}
    for name, data in columns.items():
        if name in BLOB_FIELDS:
            spec = BlobFieldSpec(shape=list(data.shape[1:]), dtype=data.dtype.name)
            fields.append(pa.field(name, pa.large_binary(), nullable=False))
            arrays.append(blob_array(data, spec))
            blob_specs[name] = spec.model_dump()
        else:
            fields.append(
                pa.field(
                    name,
                    pa.fixed_shape_tensor(pa.from_numpy_dtype(data.dtype), data.shape[1:]),
                    nullable=False,
                )
            )
            arrays.append(tensor_array(data, data.dtype, data.shape[1:]))
    # Always written (even when empty), mirroring the production ``lance_schema``.
    schema = pa.schema(
        fields, metadata={BLOB_FIELD_SPECS_SCHEMA_KEY: json.dumps(blob_specs).encode("utf-8")}
    )
    write_lance_dataset(path, schema, [pa.record_batch(arrays, schema=schema)])
