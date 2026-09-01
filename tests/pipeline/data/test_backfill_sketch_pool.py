"""Production-path coverage for distributed stored-sketch pooling."""

from __future__ import annotations

import subprocess
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
from synth_setter.pipeline.data.add_embeddings import SKETCH_FULL_STRUCT_FIELD
from synth_setter.pipeline.data.lance_shard import sketch_struct_array
from synth_setter.sketch import pool_sketch_controls


@pytest.mark.slow
def test_backfill_sketch_pool_cli_real_lance_round_trip_is_exact(
    tmp_path: Path,
) -> None:
    """The public CLI must commit exact pooled controls and remain retry-safe.

    :param tmp_path: Temporary directory for the real Lance dataset.
    """
    rows = 256
    controls = np.arange(
        rows * (SKETCH_PITCH_BINS + 2) * 401, dtype=np.float32
    ).reshape(rows, SKETCH_PITCH_BINS + 2, 401)
    source = sketch_struct_array(controls)
    uri = tmp_path / "sketches.lance"
    lance.write_dataset(
        pa.table({"row_id": np.arange(rows), SKETCH_STRUCT_FIELD: source}),
        uri,
        max_rows_per_file=64,
        max_rows_per_group=64,
    )
    command = [
        sys.executable,
        "-m",
        "synth_setter.pipeline.data.backfill_sketch_pool",
        "--lance-uri",
        str(uri),
        "--workers",
        "2",
        "--batch-size",
        "64",
        "--tasks-per-worker",
        "1",
        "--rollback-tag",
        "before-sketch-pool",
        "--num-partitions",
        "2",
    ]

    subprocess.run(command, check=True, timeout=180)  # noqa: S603
    subprocess.run(command, check=True, timeout=180)  # noqa: S603

    dataset = lance.dataset(uri)
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
