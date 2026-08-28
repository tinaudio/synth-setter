"""Tests for Lance shard validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

import synth_setter.pipeline.ci.validate_shard as validate_shard_module
from synth_setter.data.vst.audio_preview import audio_uuid, encode_audio_to_mp3
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    AUDIO_MP3_FIELD,
    AUDIO_UUID_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    MEL_SPEC_FIELD,
    PARAM_ARRAY_FIELD,
    dataset_field_shapes,
)
from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.data.lance_shard import (
    lance_schema,
    write_lance_dataset,
)
from synth_setter.pipeline.data.lance_shard import (
    record_batch_from_arrays as _record_batch_from_arrays,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.finalize_shards import (
    build_lance_smoke_spec,
    smoke_shard_metadata,
    write_minimal_lance_shard,
)
from tests.helpers.lance_fixtures import with_preview_columns


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


def record_batch_from_arrays(
    arrays: Mapping[str, np.ndarray | Sequence[bytes] | Sequence[str]],
    schema: pa.Schema,
    *,
    debug: pa.Array | None = None,
) -> pa.RecordBatch:
    """Build a canonical batch, deriving previews for tensor-only test inputs.

    :param arrays: Tensor columns, optionally with explicit preview values.
    :param schema: Arrow schema for the test shard.
    :param debug: Optional row-level seed provenance.
    :returns: Record batch matching ``schema``.
    """
    if AUDIO_MP3_FIELD in arrays and AUDIO_UUID_FIELD in arrays:
        return _record_batch_from_arrays(arrays, schema, debug=debug)
    tensor_arrays = {
        name: values for name, values in arrays.items() if isinstance(values, np.ndarray)
    }
    return _record_batch_from_arrays(
        with_preview_columns(tensor_arrays, 8000),
        schema,
        debug=debug,
    )


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


def test_validate_lance_shard_bounded_batches_traverse_every_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded real-Lance batches preserve validation through the final row.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Pytest fixture reducing the production scan cap and observing batches.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    tensor_arrays = _zero_arrays(shapes)
    audio_rows = tensor_arrays[AUDIO_FIELD]
    uuid_rows = [audio_uuid(row) for row in audio_rows]
    uuid_rows[-1] = "not-the-final-audio-uuid"
    arrays: dict[str, np.ndarray | Sequence[bytes] | Sequence[str]] = {
        **tensor_arrays,
        AUDIO_MP3_FIELD: [
            encode_audio_to_mp3(row, spec.render.sample_rate, 128) for row in audio_rows
        ],
        AUDIO_UUID_FIELD: uuid_rows,
    }
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    observed_batch_rows: list[int] = []
    original_to_batches = lance.LanceDataset.to_batches

    def observe_batches(
        dataset: lance.LanceDataset,
        *,
        columns: list[str] | None = None,
        batch_size_bytes: int | None = None,
    ) -> Iterator[pa.RecordBatch]:
        for batch in original_to_batches(
            dataset,
            columns=columns,
            batch_size_bytes=batch_size_bytes,
        ):
            observed_batch_rows.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(validate_shard_module, "LANCE_VALIDATION_BATCH_SIZE_BYTES", 1)
    monkeypatch.setattr(lance.LanceDataset, "to_batches", observe_batches)

    errors = validate_shard(shard, spec)

    assert len(observed_batch_rows) > 1
    assert sum(observed_batch_rows) == spec.render.samples_per_shard
    assert f"column {AUDIO_UUID_FIELD!r} row 3 does not match audio" in errors


def test_validate_lance_shard_missing_preview_columns_reports_both(tmp_path: Path) -> None:
    """The breaking initial-write schema rejects shards without preview columns.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    for preview_field in (AUDIO_MP3_FIELD, AUDIO_UUID_FIELD):
        schema = schema.remove(schema.get_field_index(preview_field))
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema)])

    errors = validate_shard(shard, spec)

    assert f"missing column: {AUDIO_MP3_FIELD!r}" in errors
    assert f"missing column: {AUDIO_UUID_FIELD!r}" in errors


def test_validate_lance_shard_mismatched_uuid_reports_row(tmp_path: Path) -> None:
    """A UUID not derived from its stored audio fails row-level validation.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    tensor_arrays = _zero_arrays(shapes)
    audio_rows = tensor_arrays[AUDIO_FIELD]
    arrays: dict[str, np.ndarray | Sequence[bytes] | Sequence[str]] = {
        **tensor_arrays,
        AUDIO_MP3_FIELD: [
            encode_audio_to_mp3(row, spec.render.sample_rate, 128) for row in audio_rows
        ],
        AUDIO_UUID_FIELD: ["not-the-audio-uuid"] * len(audio_rows),
    }
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    errors = validate_shard(shard, spec)

    assert f"column {AUDIO_UUID_FIELD!r} row 0 does not match audio" in errors


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        (
            AUDIO_MP3_FIELD,
            pa.field(
                AUDIO_MP3_FIELD,
                pa.binary(),
                nullable=True,
                metadata={b"mime_type": b"audio/mpeg"},
            ),
            "column 'audio_mp3' must be non-nullable",
        ),
        (
            AUDIO_MP3_FIELD,
            pa.field(
                AUDIO_MP3_FIELD,
                pa.binary(),
                nullable=False,
                metadata={b"mime_type": b"application/octet-stream"},
            ),
            "column 'audio_mp3' has metadata",
        ),
        (
            AUDIO_UUID_FIELD,
            pa.field(AUDIO_UUID_FIELD, pa.binary(), nullable=False),
            "column 'audio_uuid' has type binary, expected string",
        ),
    ],
)
def test_validate_lance_shard_preview_schema_drift_reports_contract(
    field_name: str,
    replacement: pa.Field,
    message: str,
    tmp_path: Path,
) -> None:
    """Preview type, nullability, and MIME drift fail structural validation.

    :param field_name: Preview field replaced in the canonical schema.
    :param replacement: Drifted Arrow field definition.
    :param message: Expected schema error.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    schema = schema.set(schema.get_field_index(field_name), replacement)
    tensor_arrays = _zero_arrays(shapes)
    arrays = with_preview_columns(tensor_arrays, spec.render.sample_rate)
    if field_name == AUDIO_UUID_FIELD:
        arrays[AUDIO_UUID_FIELD] = [audio_uuid(row).encode() for row in tensor_arrays[AUDIO_FIELD]]
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    assert any(message in error for error in validate_shard(shard, spec))


def test_validate_lance_shard_invalid_mp3_reports_row(tmp_path: Path) -> None:
    """A binary payload that cannot decode as MP3 fails row-level validation.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    tensor_arrays = _zero_arrays(shapes)
    audio_rows = tensor_arrays[AUDIO_FIELD]
    arrays: dict[str, np.ndarray | Sequence[bytes] | Sequence[str]] = {
        **tensor_arrays,
        AUDIO_MP3_FIELD: [b"not an mp3"] * len(audio_rows),
        AUDIO_UUID_FIELD: [audio_uuid(row) for row in audio_rows],
    }
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    errors = validate_shard(shard, spec)

    assert any(f"column {AUDIO_MP3_FIELD!r} row 0 is not decodable" in error for error in errors)


@pytest.mark.parametrize(
    ("encoded_rate", "encoded_channels", "message"),
    [
        (16000, 2, "has sample rate 16000, expected 8000"),
        (8000, 1, "has 1 channels, expected 2"),
    ],
)
def test_validate_lance_shard_mp3_playback_contract_mismatch_reports_row(
    encoded_rate: int,
    encoded_channels: int,
    message: str,
    tmp_path: Path,
) -> None:
    """A decodable preview with wrong playback geometry fails validation.

    :param encoded_rate: Sample rate used to encode the drifted preview.
    :param encoded_channels: Channel count used to encode the drifted preview.
    :param message: Expected playback-contract error.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    tensor_arrays = _zero_arrays(shapes)
    audio_rows = tensor_arrays[AUDIO_FIELD]
    arrays = with_preview_columns(tensor_arrays, spec.render.sample_rate)
    arrays[AUDIO_MP3_FIELD] = [
        encode_audio_to_mp3(row[:encoded_channels], encoded_rate, 128) for row in audio_rows
    ]
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    assert any(message in error for error in validate_shard(shard, spec))


def test_validate_lance_shard_incomplete_mp3_decode_reports_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation reads the complete preview and rejects a short decoder result.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Pytest fixture replacing the codec with an incomplete decoder.
    """

    class _IncompleteAudioFile:
        samplerate = 8000
        num_channels = 2
        frames = 3200

        def __init__(self, _payload: object) -> None:
            pass

        def __enter__(self) -> _IncompleteAudioFile:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, frames: int) -> np.ndarray:
            decoded_frames = 1 if frames == 1 else frames - 1
            return np.zeros((self.num_channels, decoded_frames), dtype=np.float32)

    spec = build_lance_smoke_spec()
    shard = tmp_path / spec.shards[0].filename
    write_minimal_lance_shard(shard, spec)
    monkeypatch.setattr("pedalboard.io.AudioFile", _IncompleteAudioFile)

    errors = validate_shard(shard, spec)

    assert any("decoded 3199 frames, expected at least 3200" in error for error in errors)


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
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema, debug=None)])

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

    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema, debug=None)])

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
    write_lance_dataset(
        shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema, debug=None)]
    )

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
    write_lance_dataset(
        shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema, debug=None)]
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
    schema = lance_schema(shapes, _first_shard_metadata(spec))
    float32_audio = pa.field(
        AUDIO_FIELD,
        pa.fixed_shape_tensor(pa.float32(), shapes[AUDIO_FIELD][1:]),
        nullable=False,
    )
    schema = schema.set(schema.get_field_index(AUDIO_FIELD), float32_audio)
    dtypes = {**DATASET_FIELD_DTYPES, AUDIO_FIELD: np.dtype("float32")}
    arrays = {field: np.zeros(shapes[field], dtype=dtypes[field]) for field in DATASET_FIELD_NAMES}
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

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
    full_schema = lance_schema(shapes, _first_shard_metadata(spec))
    batch = record_batch_from_arrays(_zero_arrays(shapes), full_schema).replace_schema_metadata(
        None
    )
    schema = full_schema.remove_metadata()
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [batch])

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
    write_lance_dataset(
        shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema, debug=None)]
    )

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
    write_lance_dataset(
        shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema, debug=None)]
    )

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
    write_lance_dataset(
        shard, schema, [record_batch_from_arrays(_zero_arrays(shapes), schema, debug=None)]
    )

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
    arrays = {
        field: np.zeros(shapes[field], dtype=DATASET_FIELD_DTYPES[field])
        for field in (AUDIO_FIELD, MEL_SPEC_FIELD)
    }
    shard = tmp_path / spec.shards[0].filename
    write_lance_dataset(shard, schema, [record_batch_from_arrays(arrays, schema)])

    errors = validate_shard(shard, spec)

    assert any(f"missing column: {PARAM_ARRAY_FIELD!r}" in error for error in errors)
