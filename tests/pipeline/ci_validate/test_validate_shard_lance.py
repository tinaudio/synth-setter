"""Tests for Lance shard validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lance.file import LanceFileReader

from synth_setter.data.vst.shapes import dataset_field_shapes
from synth_setter.pipeline.ci.validate_shard import validate_shard
from synth_setter.pipeline.data.lance_shard import (
    lance_schema,
    record_batch_from_arrays,
    tensor_chunk_to_numpy,
    write_lance_file,
)
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard


def _one_row_shapes(spec: DatasetSpec) -> dict[str, tuple[int, ...]]:
    """One-row variant of the writer's shapes: same inner dims, leading axis 1.

    :param spec: Lance spec whose render config defines the inner dims.
    :returns: Per-field shapes with the leading row axis pinned to 1.
    """
    return {
        field: (1, *shape[1:])
        for field, shape in dataset_field_shapes(spec.render, spec.num_params).items()
    }


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
    render = spec.render
    shapes = _one_row_shapes(spec)
    schema = lance_schema(
        shapes,
        ShardMetadata(
            velocity=render.velocity,
            signal_duration_seconds=render.signal_duration_seconds,
            sample_rate=render.sample_rate,
            channels=render.channels,
            min_loudness=render.min_loudness,
        ),
    )
    n, channels, n_mels, n_frames = shapes["mel_spec"]
    arrays = {
        "audio": np.zeros(shapes["audio"], dtype=np.float16),
        "mel_spec": np.zeros((n, n_mels, channels, n_frames), dtype=np.float32).transpose(
            0, 2, 1, 3
        ),
        "param_array": np.zeros(shapes["param_array"], dtype=np.float32),
    }
    shard = tmp_path / spec.shards[0].filename

    write_lance_file(shard, schema, [record_batch_from_arrays(arrays, schema)])

    reader = LanceFileReader(str(shard), columns=["mel_spec"])
    field = reader.metadata().schema.field("mel_spec")
    assert tuple(field.type.shape) == shapes["mel_spec"][1:]
    batch = next(reader.read_all().to_batches())
    decoded = tensor_chunk_to_numpy(batch.column(0), shapes["mel_spec"][1:])
    assert decoded.shape == shapes["mel_spec"]


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
    render = spec.render
    # A one-row shard disagrees with the spec's samples_per_shard=4.
    shapes = _one_row_shapes(spec)
    schema = lance_schema(
        shapes,
        ShardMetadata(
            velocity=render.velocity,
            signal_duration_seconds=render.signal_duration_seconds,
            sample_rate=render.sample_rate,
            channels=render.channels,
            min_loudness=render.min_loudness,
        ),
    )
    arrays = {
        "audio": np.zeros(shapes["audio"], dtype=np.float16),
        "mel_spec": np.zeros(shapes["mel_spec"], dtype=np.float32),
        "param_array": np.zeros(shapes["param_array"], dtype=np.float32),
    }
    shard = tmp_path / spec.shards[0].filename
    write_lance_file(shard, schema, [record_batch_from_arrays(arrays, schema)])

    errors = validate_shard(shard, spec)

    assert any("expected 4" in error for error in errors)
