"""Tests for Lance shard validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    dataset_field_shapes,
)
from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.data.lance_shard import (
    MP3_AUDIO_FIELD,
    lance_schema,
    record_batch_from_arrays,
    tensor_array,
    write_lance_file,
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


def _mp3_rows(shapes: Mapping[str, tuple[int, ...]]) -> list[bytes]:
    """Build placeholder ``audio_mp3`` blobs sized to a batch's row count.

    :param shapes: Per-field shapes; the ``audio`` leading axis sets the count.
    :returns: One non-decodable ``\\xff\\xfb`` placeholder blob per row, in row order.
    """
    return [b"\xff\xfb"] * shapes[AUDIO_FIELD][0]


def test_validate_lance_shard_accepts_valid_file(tmp_path: Path) -> None:
    """A structurally valid Lance shard returns no validation errors.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shard = tmp_path / spec.shards[0].filename
    write_minimal_lance_shard(shard, spec)

    assert validate_shard(shard, spec) == []


def test_lance_record_batch_preserves_transposed_tensor_shape(tmp_path: Path) -> None:
    """Non-contiguous rendered tensors keep the schema's declared shape.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = _one_row_shapes(spec)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    n, channels, n_mels, n_frames = shapes[MEL_SPEC_FIELD]
    arrays = _zero_arrays(shapes)
    arrays[MEL_SPEC_FIELD] = np.zeros((n, n_mels, channels, n_frames), dtype=np.float32).transpose(
        0, 2, 1, 3
    )
    shard = tmp_path / spec.shards[0].filename

    write_lance_file(shard, schema, [record_batch_from_arrays(arrays, schema, _mp3_rows(shapes))])

    reader = LanceFileReader(str(shard), columns=[MEL_SPEC_FIELD])
    field = reader.metadata().schema.field(MEL_SPEC_FIELD)
    assert tuple(field.type.shape) == shapes[MEL_SPEC_FIELD][1:]
    batch = next(reader.read_all().to_batches())
    decoded = batch.column(0).to_numpy_ndarray()
    assert decoded.shape == shapes[MEL_SPEC_FIELD]


def test_validate_lance_shard_rejects_bad_suffix_payload(tmp_path: Path) -> None:
    """Garbage bytes under a ``.lance`` suffix report a Lance-open error.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shard = tmp_path / spec.shards[0].filename
    shard.write_bytes(b"not lance")

    errors = validate_shard(shard, spec)

    assert errors
    assert "valid Lance file" in errors[0]


def test_validate_lance_shard_reports_row_count_mismatch(tmp_path: Path) -> None:
    """A Lance shard with too few rows reports the expected row count.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    # A one-row shard disagrees with the spec's samples_per_shard.
    shapes = _one_row_shapes(spec)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    shard = tmp_path / spec.shards[0].filename
    write_lance_file(
        shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema, _mp3_rows(shapes))]
    )

    errors = validate_shard(shard, spec)

    row_count_error = f"file has 1 rows, expected {spec.render.samples_per_shard}"
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
    write_lance_file(
        shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema, _mp3_rows(shapes))]
    )

    errors = validate_shard(shard, spec)

    expected_inner = (channels, n_mels, n_frames)
    actual_inner = (channels, n_mels + 1, n_frames)
    assert any(
        f"column {MEL_SPEC_FIELD!r} has inner shape {actual_inner}, expected {expected_inner}"
        in error
        for error in errors
    )


def test_validate_lance_shard_reports_value_dtype_mismatch(tmp_path: Path) -> None:
    """A Lance shard whose audio column is float32 reports the dtype contract.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    float32_audio = pa.field(
        AUDIO_FIELD,
        pa.fixed_shape_tensor(pa.float32(), shapes[AUDIO_FIELD][1:]),
        nullable=False,
    )
    schema = schema.set(schema.get_field_index(AUDIO_FIELD), float32_audio)
    dtypes = {**DATASET_FIELD_DTYPES, AUDIO_FIELD: np.dtype("float32")}
    columns = [
        tensor_array(
            np.zeros(shapes[field], dtype=dtypes[field]), dtypes[field], shapes[field][1:]
        )
        for field in DATASET_FIELD_NAMES
    ]
    columns.append(pa.array(_mp3_rows(shapes), type=pa.large_binary()))
    shard = tmp_path / spec.shards[0].filename
    write_lance_file(shard, schema, [pa.record_batch(columns, schema=schema)])

    errors = validate_shard(shard, spec)

    assert any(
        f"column {AUDIO_FIELD!r} has value type float, expected halffloat" in error
        for error in errors
    )


def test_validate_lance_shard_reports_missing_schema_metadata(tmp_path: Path) -> None:
    """A Lance shard without embedded ``ShardMetadata`` reports the missing key.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render)).remove_metadata()
    shard = tmp_path / spec.shards[0].filename
    write_lance_file(
        shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema, _mp3_rows(shapes))]
    )

    errors = validate_shard(shard, spec)

    assert any("missing schema metadata key" in error for error in errors)


def test_validate_lance_shard_reports_missing_column(tmp_path: Path) -> None:
    """A Lance shard missing one writer field reports the absent column.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    full_schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    schema = full_schema.remove(full_schema.get_field_index(PARAM_ARRAY_FIELD))
    columns = [
        tensor_array(
            np.zeros(shapes[field], dtype=DATASET_FIELD_DTYPES[field]),
            DATASET_FIELD_DTYPES[field],
            shapes[field][1:],
        )
        for field in (AUDIO_FIELD, MEL_SPEC_FIELD)
    ]
    columns.append(pa.array(_mp3_rows(shapes), type=pa.large_binary()))
    shard = tmp_path / spec.shards[0].filename
    write_lance_file(shard, schema, [pa.record_batch(columns, schema=schema)])

    errors = validate_shard(shard, spec)

    assert any(f"missing column: {PARAM_ARRAY_FIELD!r}" in error for error in errors)


def test_validate_lance_shard_reports_missing_audio_mp3_column(tmp_path: Path) -> None:
    """A Lance shard without the ``audio_mp3`` preview column reports it as missing.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    full_schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    schema = full_schema.remove(full_schema.get_field_index(MP3_AUDIO_FIELD))
    columns = [
        tensor_array(
            np.zeros(shapes[field], dtype=DATASET_FIELD_DTYPES[field]),
            DATASET_FIELD_DTYPES[field],
            shapes[field][1:],
        )
        for field in DATASET_FIELD_NAMES
    ]
    shard = tmp_path / spec.shards[0].filename
    write_lance_file(shard, schema, [pa.record_batch(columns, schema=schema)])

    errors = validate_shard(shard, spec)

    assert any(f"missing column: {MP3_AUDIO_FIELD!r}" in error for error in errors)


def test_validate_lance_shard_reports_wrong_audio_mp3_type(tmp_path: Path) -> None:
    """An ``audio_mp3`` column typed as anything but ``large_binary`` is rejected.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, smoke_shard_metadata(spec.render))
    binary_mp3 = pa.field(MP3_AUDIO_FIELD, pa.binary(), nullable=False)
    schema = schema.set(schema.get_field_index(MP3_AUDIO_FIELD), binary_mp3)
    columns = [
        tensor_array(
            np.zeros(shapes[field], dtype=DATASET_FIELD_DTYPES[field]),
            DATASET_FIELD_DTYPES[field],
            shapes[field][1:],
        )
        for field in DATASET_FIELD_NAMES
    ]
    columns.append(pa.array(_mp3_rows(shapes), type=pa.binary()))
    shard = tmp_path / spec.shards[0].filename
    write_lance_file(shard, schema, [pa.record_batch(columns, schema=schema)])

    errors = validate_shard(shard, spec)

    assert any(
        f"column {MP3_AUDIO_FIELD!r}" in error and "large_binary" in error for error in errors
    )
