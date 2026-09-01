"""Production-path coverage for distributed stored-sketch pooling."""

from __future__ import annotations

import base64
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
from synth_setter.pipeline.data.add_embeddings import (
    SKETCH_FULL_STRUCT_FIELD,
    _sketch_pool_artifact_identity,
)
from synth_setter.pipeline.data.backfill_sketch_pool import (
    SketchPoolBackfillConfig,
    _AlterColumnsDataset,
    _CacheIdentity,
    _ColumnRename,
    _ensure_canonical_index,
    _ensure_rollback_tag,
    _FragmentTask,
    _load_reports,
    _parse_args,
    _persist_report,
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
    rng = np.random.default_rng(2980)
    controls = rng.random((rows, SKETCH_PITCH_BINS + 2, 401), dtype=np.float32)
    controls[:, :2] = controls[:, :2] * 2.0 - 1.0
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
        branch="candidate",
        workers=2,
        batch_size=64,
        tasks_per_worker=1,
        rollback_tag="before-sketch-pool",
        num_partitions=2,
        resume_dir=fake_r2_remote / "resume",
    )

    main_dataset = lance.dataset(local_uri)
    dataset = main_dataset.create_branch("candidate")
    dataset.tags.create("before-sketch-pool", ("candidate", dataset.version))
    cast("_AlterColumnsDataset", dataset).alter_columns(
        _ColumnRename(path=SKETCH_STRUCT_FIELD, name=SKETCH_FULL_STRUCT_FIELD)
    )
    renamed = lance.dataset(local_uri).checkout_version(("candidate", None))
    fragment_ids = {fragment.metadata.id for fragment in renamed.get_fragments()}
    first_fragment = min(fragment_ids)
    identity = _CacheIdentity(
        lance_uri=config.lance_uri,
        branch=config.branch,
        source_version=renamed.version,
        batch_size=config.batch_size,
        artifact=_sketch_pool_artifact_identity(""),
    )
    resume_dir = config.resume_dir
    assert resume_dir is not None
    _load_reports(resume_dir, identity, fragment_ids)
    _persist_report(
        resume_dir,
        _transform_fragment(
            _FragmentTask(
                uri=str(local_uri),
                storage_options=None,
                branch="candidate",
                source_version=renamed.version,
                fragment_id=first_fragment,
                batch_size=config.batch_size,
                artifact=identity.artifact,
            )
        ),
    )

    command = [
        str(Path(sys.executable).parent / "synth-setter-backfill-sketch-pool"),
        "--lance-uri",
        config.lance_uri,
        "--branch",
        config.branch,
        "--workers",
        str(config.workers),
        "--batch-size",
        str(config.batch_size),
        "--tasks-per-worker",
        str(config.tasks_per_worker),
        "--rollback-tag",
        config.rollback_tag,
        "--num-partitions",
        str(config.num_partitions),
        "--resume-dir",
        str(resume_dir),
    ]
    completed = subprocess.run(  # noqa: S603 -- test owns the executable and every argument.
        command, check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    dataset = lance.dataset(local_uri).checkout_version(("candidate", None))
    first_committed_version = dataset.version
    with pytest.raises(ValueError, match="config .* does not match"):
        _ensure_canonical_index(
            dataset,
            config.model_copy(update={"num_partitions": 1}),
        )

    backfill_sketch_pool(config)

    dataset = lance.dataset(local_uri).checkout_version(("candidate", None))
    assert dataset.version == first_committed_version
    assert lance.dataset(local_uri).version == main_dataset.version
    assert dataset.tags.get_version("before-sketch-pool") == main_dataset.version
    assert not resume_dir.exists()
    assert dataset.take(range(rows), columns=["row_id"]).column(0).to_pylist() == list(range(rows))
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


def test_ensure_rollback_tag_with_post_rename_snapshot_rejects_tag(tmp_path: Path) -> None:
    """A same-branch tag must still identify the canonical-only pre-migration schema.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    uri = tmp_path / "wrong-tag.lance"
    dataset = lance.write_dataset(pa.table({SKETCH_FULL_STRUCT_FIELD: [1]}), uri)
    dataset.tags.create("wrong", dataset.version)
    config = SketchPoolBackfillConfig(
        lance_uri=str(uri),
        workers=1,
        rollback_tag="wrong",
        build_index=False,
    )

    with pytest.raises(ValueError, match="canonical-only pre-migration schema"):
        _ensure_rollback_tag(dataset, config, str(uri), None, create=False)


def test_transform_fragment_real_lance_source_returns_merge_metadata(tmp_path: Path) -> None:
    """The Ray worker callable must write valid uncommitted merge metadata.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    rows = 2
    rng = np.random.default_rng(2980)
    controls = rng.random((rows, SKETCH_PITCH_BINS + 2, 401), dtype=np.float32)
    controls[:, :2] = controls[:, :2] * 2.0 - 1.0
    uri = tmp_path / "source.lance"
    dataset = lance.write_dataset(
        pa.table({SKETCH_FULL_STRUCT_FIELD: sketch_struct_array(controls)}), uri
    )
    fragment_id = dataset.get_fragments()[0].metadata.id

    report = _transform_fragment(
        _FragmentTask(
            uri=str(uri),
            storage_options=None,
            branch="main",
            source_version=dataset.version,
            fragment_id=fragment_id,
            batch_size=2,
            artifact=_sketch_pool_artifact_identity(""),
        )
    )

    metadata = lance.fragment.FragmentMetadata.from_json(report.metadata_json)
    arrow_schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(report.schema_ipc)))
    assert report.row_count == rows
    assert metadata.id == fragment_id
    assert arrow_schema.names == [SKETCH_FULL_STRUCT_FIELD, SKETCH_STRUCT_FIELD]
    assert (
        arrow_schema.field(SKETCH_STRUCT_FIELD).metadata[b"synth_setter.embedding.name"]
        == b"sketch_pool"
    )


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
            "--resume-dir",
            "resume",
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
        resume_dir=Path("resume"),
        result=Path("result.json"),
    )
