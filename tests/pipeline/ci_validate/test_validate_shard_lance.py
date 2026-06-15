"""Tests for Lance shard validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    dataset_field_shapes,
)
from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.data.lance_shard import (
    BLOB_FIELD_SPECS_SCHEMA_KEY,
    SHARD_METADATA_SCHEMA_KEY,
    blob_array,
    iter_lance_column_rows,
    lance_schema,
    read_blob_field_specs,
    record_batch_from_arrays,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.finalize_shards import (
    build_lance_smoke_spec,
    smoke_shard_metadata,
    write_minimal_lance_shard,
)


def _one_row_shapes(spec: DatasetSpec) -> dict[str, tuple[int, ...]]:
    """One-row variant of the writer's shapes: same inner dims, leading axis 1.

    :param spec: Lance spec whose render config defines the inner dims.
    :returns: Per-field shapes with the leading row axis pinned to 1.
    """
    return {
        field: (1, *shape[1:])
        for field, shape in dataset_field_shapes(spec.render, spec.num_params).items()
    }


def _zero_arrays(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Build all-zero per-field arrays with the writer's on-disk dtypes.

    :param shapes: Full per-field shapes including the leading row axis.
    :returns: Mapping ready for ``record_batch_from_arrays``.
    """
    return {
        field: np.zeros(shape, dtype=DATASET_FIELD_DTYPES[field])
        for field, shape in shapes.items()
    }


def test_validate_lance_shard_accepts_valid_file(tmp_path: Path) -> None:
    """A structurally valid Lance shard returns no validation errors.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shard = tmp_path / spec.shards[0].filename
    write_minimal_lance_shard(shard, spec)

    assert validate_shard(shard, spec) == []


def test_lance_blob_column_preserves_transposed_mel_logical_values(tmp_path: Path) -> None:
    """A non-contiguous rendered mel decodes from its BLOB column to logical values.

    The mel column is ``large_binary``, so a codec serializing raw buffer order
    instead of logical order would scramble values while keeping shapes intact.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = _one_row_shapes(spec)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    n, channels, n_mels, n_frames = shapes[MEL_SPEC_FIELD]
    transposed_mel = (
        np.arange(n * channels * n_mels * n_frames, dtype=np.float32)
        .reshape(n, n_mels, channels, n_frames)
        .transpose(0, 2, 1, 3)
    )
    arrays = _zero_arrays(shapes)
    arrays[MEL_SPEC_FIELD] = transposed_mel
    shard = tmp_path / spec.shards[0].filename

    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    assert pa.types.is_large_binary(lance.dataset(str(shard)).schema.field(MEL_SPEC_FIELD).type)
    decoded = np.stack(list(iter_lance_column_rows(shard, MEL_SPEC_FIELD)), axis=0)
    np.testing.assert_array_equal(decoded, transposed_mel)


def test_validate_lance_shard_rejects_bad_suffix_payload(tmp_path: Path) -> None:
    """Garbage bytes under a ``.lance`` suffix report a Lance-open error.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shard = tmp_path / spec.shards[0].filename
    shard.write_bytes(b"not lance")

    errors = validate_shard(shard, spec)

    assert errors
    assert "valid Lance dataset" in errors[0]


def test_validate_lance_shard_reports_row_count_mismatch(tmp_path: Path) -> None:
    """A Lance shard with too few rows reports the expected row count.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    # A one-row shard disagrees with the spec's samples_per_shard.
    shapes = _one_row_shapes(spec)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    row_count_error = f"dataset has 1 rows, expected {spec.render.samples_per_shard}"
    assert any(row_count_error in error for error in errors)


def test_validate_lance_shard_reports_inner_shape_mismatch(tmp_path: Path) -> None:
    """A Lance shard whose mel column has a wrong inner shape names both shapes.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    expected_shapes = dataset_field_shapes(spec.render, spec.num_params)
    n, channels, n_mels, n_frames = expected_shapes[MEL_SPEC_FIELD]
    shapes = {**expected_shapes, MEL_SPEC_FIELD: (n, channels, n_mels + 1, n_frames)}
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    expected_inner = (channels, n_mels, n_frames)
    actual_inner = (channels, n_mels + 1, n_frames)
    assert any(
        f"column {MEL_SPEC_FIELD!r} has inner shape {actual_inner}, expected {expected_inner}"
        in error
        for error in errors
    )


def test_validate_lance_shard_reports_tensor_value_dtype_mismatch(tmp_path: Path) -> None:
    """A Lance shard whose ``param_array`` tensor is float16 reports the dtype contract.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    float16_params = pa.field(
        PARAM_ARRAY_FIELD,
        pa.fixed_shape_tensor(pa.float16(), shapes[PARAM_ARRAY_FIELD][1:]),
        nullable=False,
    )
    schema = schema.set(schema.get_field_index(PARAM_ARRAY_FIELD), float16_params)
    arrays = _zero_arrays(shapes)
    arrays[PARAM_ARRAY_FIELD] = arrays[PARAM_ARRAY_FIELD].astype(np.float16)
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    errors = validate_shard(shard, spec)

    assert any(
        f"column {PARAM_ARRAY_FIELD!r} has value type halffloat, expected float" in error
        for error in errors
    )


def test_validate_lance_shard_rejects_blob_field_stored_as_tensor(tmp_path: Path) -> None:
    """A Lance shard whose ``audio`` is a fixed-shape tensor reports the BLOB contract.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    # Capture the mel spec before mutating audio — the audio blob spec stays in
    # the schema metadata even after audio is rewritten as a fixed-shape tensor.
    mel_spec_obj = read_blob_field_specs(schema)[MEL_SPEC_FIELD]
    tensor_audio = pa.field(
        AUDIO_FIELD,
        pa.fixed_shape_tensor(
            pa.from_numpy_dtype(DATASET_FIELD_DTYPES[AUDIO_FIELD]), shapes[AUDIO_FIELD][1:]
        ),
        nullable=False,
    )
    schema = schema.set(schema.get_field_index(AUDIO_FIELD), tensor_audio)
    columns = [
        tensor_array(
            np.zeros(shapes[AUDIO_FIELD], dtype=DATASET_FIELD_DTYPES[AUDIO_FIELD]),
            DATASET_FIELD_DTYPES[AUDIO_FIELD],
            shapes[AUDIO_FIELD][1:],
        ),
        blob_array(
            np.zeros(shapes[MEL_SPEC_FIELD], dtype=DATASET_FIELD_DTYPES[MEL_SPEC_FIELD]),
            mel_spec_obj,
        ),
        tensor_array(
            np.zeros(shapes[PARAM_ARRAY_FIELD], dtype=DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD]),
            DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD],
            shapes[PARAM_ARRAY_FIELD][1:],
        ),
    ]
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [pa.record_batch(columns, schema=schema)])

    errors = validate_shard(shard, spec)

    assert any(f"non-large_binary column {AUDIO_FIELD!r}" in error for error in errors)


def test_validate_lance_shard_reports_blob_field_missing_its_spec(tmp_path: Path) -> None:
    """A ``large_binary`` BLOB column whose embedded spec is absent reports the missing spec.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    full_schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    blob_specs = read_blob_field_specs(full_schema)
    columns = [
        blob_array(
            np.zeros(shapes[AUDIO_FIELD], dtype=DATASET_FIELD_DTYPES[AUDIO_FIELD]),
            blob_specs[AUDIO_FIELD],
        ),
        blob_array(
            np.zeros(shapes[MEL_SPEC_FIELD], dtype=DATASET_FIELD_DTYPES[MEL_SPEC_FIELD]),
            blob_specs[MEL_SPEC_FIELD],
        ),
        tensor_array(
            np.zeros(shapes[PARAM_ARRAY_FIELD], dtype=DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD]),
            DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD],
            shapes[PARAM_ARRAY_FIELD][1:],
        ),
    ]
    # Strip the BLOB specs key while leaving the columns large_binary.
    metadata = {SHARD_METADATA_SCHEMA_KEY: full_schema.metadata[SHARD_METADATA_SCHEMA_KEY]}
    schema = full_schema.with_metadata(metadata)
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [pa.record_batch(columns, schema=schema)])

    errors = validate_shard(shard, spec)

    assert any(
        f"column {AUDIO_FIELD!r} is missing its blob field spec" in error for error in errors
    )


def test_validate_lance_shard_reports_blob_value_dtype_mismatch(tmp_path: Path) -> None:
    """A Lance shard whose ``audio`` BLOB spec claims float32 reports the dtype contract.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    # Tamper each BLOB field to the *other* field's dtype so both mismatch.
    tampered = {
        AUDIO_FIELD: {"shape": list(shapes[AUDIO_FIELD][1:]), "dtype": "float32"},
        MEL_SPEC_FIELD: {"shape": list(shapes[MEL_SPEC_FIELD][1:]), "dtype": "float16"},
    }
    schema = schema.with_metadata(
        {**schema.metadata, BLOB_FIELD_SPECS_SCHEMA_KEY: json.dumps(tampered).encode("utf-8")}
    )
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    assert any(
        f"column {AUDIO_FIELD!r} has value type float32, expected float16" in error
        for error in errors
    )
    assert any(
        f"column {MEL_SPEC_FIELD!r} has value type float16, expected float32" in error
        for error in errors
    )


def test_validate_lance_shard_reports_missing_schema_metadata(tmp_path: Path) -> None:
    """A Lance shard without embedded ``ShardMetadata`` reports the missing key.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    full_schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    # Drop only the ShardMetadata key, keeping the BLOB specs the writer needs.
    metadata = {k: v for k, v in full_schema.metadata.items() if k != SHARD_METADATA_SCHEMA_KEY}
    schema = full_schema.with_metadata(metadata)
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    assert any("missing schema metadata key" in error for error in errors)


def test_validate_lance_shard_reports_missing_column(tmp_path: Path) -> None:
    """A Lance shard missing one writer field reports the absent column.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    full_schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    blob_specs = read_blob_field_specs(full_schema)
    schema = full_schema.remove(full_schema.get_field_index(PARAM_ARRAY_FIELD))
    columns = [
        blob_array(np.zeros(shapes[field], dtype=DATASET_FIELD_DTYPES[field]), blob_specs[field])
        for field in (AUDIO_FIELD, MEL_SPEC_FIELD)
    ]
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [pa.record_batch(columns, schema=schema)])

    errors = validate_shard(shard, spec)

    assert any(f"missing column: {PARAM_ARRAY_FIELD!r}" in error for error in errors)
