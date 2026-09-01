"""Production-path coverage for distributed stored-sketch pooling."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import cast

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch

from synth_setter.data.vst.shapes import (
    SKETCH_CENTROID_CHILD,
    SKETCH_LOUDNESS_CHILD,
    SKETCH_PITCH_BINS,
    SKETCH_PITCH_CHILD,
    SKETCH_STRUCT_FIELD,
)
from synth_setter.pipeline.data.add_embeddings import (
    SKETCH_FULL_STRUCT_FIELD,
    _sketch_pool_artifact_identity,
)
from synth_setter.pipeline.data.backfill_sketch_pool import (
    SketchPoolBackfillConfig,
    _parse_args,
    _transform_fragment,
    backfill_sketch_pool,
)
from synth_setter.pipeline.data.lance_shard import sketch_struct_array
from synth_setter.sketch import pool_sketch_controls


def test_backfill_sketch_pool_cli_real_lance_round_trip_is_exact(
    fake_r2_remote: Path,
) -> None:
    """The public CLI must commit exact pooled controls and remain retry-safe.

    :param fake_r2_remote: Filesystem-backed real rclone remote root.
    """
    rows = 256
    controls = np.arange(
        rows * (SKETCH_PITCH_BINS + 2) * 401, dtype=np.float32
    ).reshape(rows, SKETCH_PITCH_BINS + 2, 401)
    source = sketch_struct_array(controls)
    local_uri = fake_r2_remote / "test-bucket" / "sketches.lance"
    local_uri.parent.mkdir()
    lance.write_dataset(
        pa.table({"row_id": np.arange(rows), SKETCH_STRUCT_FIELD: source}),
        local_uri,
        max_rows_per_file=64,
        max_rows_per_group=64,
    )
    config = SketchPoolBackfillConfig(
        lance_uri="r2://test-bucket/sketches.lance",
        workers=2,
        batch_size=64,
        tasks_per_worker=1,
        rollback_tag="before-sketch-pool",
        num_partitions=2,
    )

    backfill_sketch_pool(config)
    backfill_sketch_pool(config)

    dataset = lance.dataset(local_uri)
    assert dataset.version == 4
    assert dataset.tags.get_version("before-sketch-pool") == 1
    assert dataset.take(range(rows), columns=["row_id"]).column(0).to_pylist() == list(
        range(rows)
    )
    assert {SKETCH_FULL_STRUCT_FIELD, SKETCH_STRUCT_FIELD} <= set(dataset.schema.names)
    field = dataset.schema.field(SKETCH_STRUCT_FIELD)
    assert field.metadata[b"synth_setter.embedding.name"] == b"sketch_pool"
    indices = cast("list[dict[str, object]]", dataset.list_indices())
    assert [SKETCH_STRUCT_FIELD + ".vec"] in [index["fields"] for index in indices]
    actual_rows = (
        dataset.take(range(rows), columns=[SKETCH_STRUCT_FIELD])
        .column(0)
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    loudness = np.stack([row[SKETCH_LOUDNESS_CHILD] for row in actual_rows])
    centroid = np.stack([row[SKETCH_CENTROID_CHILD] for row in actual_rows])
    pitch = np.stack([row[SKETCH_PITCH_CHILD] for row in actual_rows]).reshape(
        rows, SKETCH_PITCH_BINS, -1
    )
    actual = np.concatenate((loudness[:, None], centroid[:, None], pitch), axis=1)
    expected = pool_sketch_controls(torch.from_numpy(controls)).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_transform_fragment_real_lance_source_returns_merge_metadata(tmp_path: Path) -> None:
    """The Ray worker callable must write valid uncommitted merge metadata.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    rows = 2
    controls = np.arange(
        rows * (SKETCH_PITCH_BINS + 2) * 401, dtype=np.float32
    ).reshape(rows, SKETCH_PITCH_BINS + 2, 401)
    uri = tmp_path / "source.lance"
    dataset = lance.write_dataset(
        pa.table({SKETCH_FULL_STRUCT_FIELD: sketch_struct_array(controls)}), uri
    )
    fragment_id = dataset.get_fragments()[0].metadata.id

    metadata_bytes, schema_bytes, transformed_rows = _transform_fragment(
        str(uri),
        None,
        "main",
        dataset.version,
        fragment_id,
        2,
        _sketch_pool_artifact_identity("").encode(),
    )

    metadata = pickle.loads(metadata_bytes)  # noqa: S301
    schema = pickle.loads(schema_bytes)  # noqa: S301
    assert transformed_rows == rows
    assert metadata.id == fragment_id
    arrow_schema = schema.to_pyarrow()
    assert arrow_schema.names == [SKETCH_FULL_STRUCT_FIELD, SKETCH_STRUCT_FIELD]
    assert arrow_schema.field(SKETCH_STRUCT_FIELD).metadata[
        b"synth_setter.embedding.name"
    ] == b"sketch_pool"


def test_parse_args_with_explicit_cli_values_returns_strict_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public CLI flags must map to the strict migration boundary.

    :param monkeypatch: Fixture replacing the process argument vector.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-backfill-sketch-pool",
            "--lance-uri",
            "r2://bucket/split.lance",
            "--branch",
            "candidate",
            "--workers",
            "3",
            "--batch-size",
            "16",
            "--tasks-per-worker",
            "2",
            "--rollback-tag",
            "before",
            "--no-build-index",
            "--result",
            "result.json",
        ],
    )

    config = _parse_args()

    assert config == SketchPoolBackfillConfig(
        lance_uri="r2://bucket/split.lance",
        branch="candidate",
        workers=3,
        batch_size=16,
        tasks_per_worker=2,
        rollback_tag="before",
        build_index=False,
        result=Path("result.json"),
    )
