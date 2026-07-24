"""Tests for Lance shard validation."""

from __future__ import annotations

from collections.abc import Mapping
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
    dataset_field_shapes,
)
from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.data.lance_shard import (
    lance_schema,
    record_batch_from_arrays,
    tensor_array,
    write_lance_dataset,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
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


def _first_shard_metadata(spec: DatasetSpec) -> ShardMetadata:
    """Return metadata matching the first shard's launcher-injected seed.

    :param spec: One-shard Lance smoke spec.
    :returns: Shard metadata carrying ``spec.shards[0].seed``.
    """
    render = spec.render.model_copy(update={"base_seed": spec.shards[0].seed})
    return smoke_shard_metadata(render)


def test_validate_lance_shard_accepts_split_local_sample_offset(tmp_path: Path) -> None:
    """Validation matches nonzero split-local offset provenance.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    base = build_lance_smoke_spec()
    render = base.render.model_copy(update={"samples_per_shard": 2, "samples_per_render_batch": 2})
    spec = build_lance_smoke_spec(
        train_val_test_sizes=(4, 0, 0),
        render=render,
        train_val_test_seeds=(101, 202, 303),
    )
    shard = tmp_path / spec.shards[1].filename
    write_minimal_lance_shard(shard, spec)

    assert validate_shard(shard, spec) == []


def test_validate_lance_shard_accepts_configured_signal_dtypes(tmp_path: Path) -> None:
    """Validation derives expected signal widths from the persisted render config.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    base = build_lance_smoke_spec()
    render = base.render.model_copy(update={"audio_dtype": "float32", "mel_spec_dtype": "float16"})
    spec = build_lance_smoke_spec(render=render)
    shard = tmp_path / spec.shards[0].filename
    write_minimal_lance_shard(shard, spec)

    assert validate_shard(shard, spec) == []


def test_validate_lance_shard_accepts_valid_file(tmp_path: Path) -> None:
    """A structurally valid Lance shard returns no validation errors.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shard = tmp_path / spec.shards[0].filename
    write_minimal_lance_shard(shard, spec)

    assert validate_shard(shard, spec) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (AUDIO_FIELD, np.nan, "column 'audio' contains non-finite values"),
        (MEL_SPEC_FIELD, np.inf, "column 'mel_spec' contains non-finite values"),
        (PARAM_ARRAY_FIELD, np.nan, "column 'param_array' contains non-finite values"),
        (AUDIO_FIELD, 1.01, "column 'audio' contains values outside [-1, 1]"),
        (PARAM_ARRAY_FIELD, -0.01, "column 'param_array' contains values outside [0, 1]"),
    ],
)
def test_validate_lance_shard_invalid_values_reports_field_contract(
    field: str,
    value: float,
    message: str,
    tmp_path: Path,
) -> None:
    """Non-finite or out-of-range tensor values fail the worker staging gate.

    :param field: Dataset field receiving the invalid value.
    :param value: Non-finite or out-of-range value written at the first element.
    :param message: Expected validation error naming the violated field contract.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    arrays = _zero_arrays(shapes)
    arrays[field].flat[0] = value
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    assert message in validate_shard(shard, spec)


def test_lance_record_batch_preserves_transposed_tensor_shape(tmp_path: Path) -> None:
    """Non-contiguous rendered tensors keep the schema's declared shape.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = _one_row_shapes(spec)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    n, channels, n_mels, n_frames = shapes[MEL_SPEC_FIELD]
    arrays = _zero_arrays(shapes)
    arrays[MEL_SPEC_FIELD] = np.zeros((n, n_mels, channels, n_frames), dtype=np.float32).transpose(
        0, 2, 1, 3
    )
    shard = tmp_path / spec.shards[0].filename

    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    dataset = lance.dataset(str(shard))
    field = dataset.schema.field(MEL_SPEC_FIELD)
    assert tuple(field.type.shape) == shapes[MEL_SPEC_FIELD][1:]
    batch = next(dataset.to_batches(columns=[MEL_SPEC_FIELD]))
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
    assert "valid Lance dataset" in errors[0]


def test_validate_lance_shard_reports_row_count_mismatch(tmp_path: Path) -> None:
    """A Lance shard with too few rows reports the expected row count.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    # A one-row shard disagrees with the spec's samples_per_shard.
    shapes = _one_row_shapes(spec)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
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
    schema = lance_schema(shapes, _first_shard_metadata(spec))
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


def test_validate_lance_shard_reports_value_dtype_mismatch(tmp_path: Path) -> None:
    """A Lance shard whose audio column is float32 reports the dtype contract.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
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
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [pa.record_batch(columns, schema=schema)])

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
    schema = lance_schema(shapes, _first_shard_metadata(spec)).remove_metadata()
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    assert any("missing schema metadata key" in error for error in errors)


def test_validate_lance_shard_reports_base_seed_metadata_mismatch(tmp_path: Path) -> None:
    """A Lance shard whose embedded seed differs from the spec is rejected.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    metadata = _first_shard_metadata(spec).model_copy(
        update={"base_seed": spec.render.base_seed + 1}
    )
    schema = lance_schema(shapes, metadata)
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    assert any("base_seed" in error for error in errors)


def test_validate_lance_shard_reports_sample_offset_metadata_mismatch(tmp_path: Path) -> None:
    """A shard whose sample offset differs from the spec is rejected.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    metadata = _first_shard_metadata(spec).model_copy(update={"sample_offset": 1})
    schema = lance_schema(shapes, metadata)
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    assert any("sample_offset" in error for error in errors)


def test_validate_lance_shard_reports_attempt_budget_metadata_mismatch(tmp_path: Path) -> None:
    """A Lance shard whose embedded retry budget differs from the spec is rejected.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    metadata = _first_shard_metadata(spec).model_copy(
        update={"attempts_per_sample": spec.render.attempts_per_sample + 1}
    )
    schema = lance_schema(shapes, metadata)
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    assert any("attempts_per_sample" in error for error in errors)


def test_validate_lance_shard_reports_missing_column(tmp_path: Path) -> None:
    """A Lance shard missing one writer field reports the absent column.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    full_schema = lance_schema(shapes, _first_shard_metadata(spec))
    schema = full_schema.remove(full_schema.get_field_index(PARAM_ARRAY_FIELD))
    columns = [
        tensor_array(
            np.zeros(shapes[field], dtype=DATASET_FIELD_DTYPES[field]),
            DATASET_FIELD_DTYPES[field],
            shapes[field][1:],
        )
        for field in (AUDIO_FIELD, MEL_SPEC_FIELD)
    ]
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [pa.record_batch(columns, schema=schema)])

    errors = validate_shard(shard, spec)

    assert any(f"missing column: {PARAM_ARRAY_FIELD!r}" in error for error in errors)
