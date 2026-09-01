"""Production-path coverage for distributed stored-sketch pooling."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import lance
import numpy as np
import pyarrow as pa
import pytest
import ray
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
    _DispatchState,
    _ensure_canonical_index,
    _FragmentTask,
    _load_reports,
    _parse_args,
    _persist_report,
    _poll_dispatch,
    _prepare_source,
    _report_store,
    _ReportStore,
    _result,
    _ResultState,
    _resume_directory,
    _retry,
    _transform_fragment,
    _validate_rollback_tag,
    backfill_sketch_pool,
)
from synth_setter.pipeline.data.lance_shard import sketch_struct_array
from synth_setter.sketch import pool_sketch_controls


def test_backfill_sketch_pool_candidate_real_lance_round_trip_is_exact(
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
    report_store = _report_store(config, identity)
    _load_reports(report_store, identity, fragment_ids)
    _persist_report(
        report_store,
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
    shutil.rmtree(resume_dir)

    backfill_sketch_pool(config)
    dataset = lance.dataset(local_uri).checkout_version(("candidate", None))
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


def test_backfill_sketch_pool_local_source_commits_exact_pooled_values(
    tmp_path: Path,
) -> None:
    """The public operator completes its normal path without R2 test credentials.

    :param tmp_path: Temporary directory for the real local Lance source.
    """
    rows = 2
    rng = np.random.default_rng(2980)
    controls = rng.random((rows, SKETCH_PITCH_BINS + 2, 401), dtype=np.float32)
    controls[:, :2] = controls[:, :2] * 2.0 - 1.0
    uri = tmp_path / "local-source.lance"
    lance.write_dataset(
        pa.table({SKETCH_STRUCT_FIELD: sketch_struct_array(controls)}),
        uri,
    )
    config = SketchPoolBackfillConfig(
        lance_uri=str(uri),
        workers=1,
        rollback_tag="before-local-pool",
        build_index=False,
        resume_dir=tmp_path / "resume",
    )

    result = backfill_sketch_pool(config)

    dataset = lance.dataset(uri)
    committed_version = dataset.version
    pooled_rows = (
        dataset.to_table(columns=[SKETCH_STRUCT_FIELD])[SKETCH_STRUCT_FIELD]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    loudness = np.stack([row[SKETCH_LOUDNESS_CHILD] for row in pooled_rows])
    centroid = np.stack([row[SKETCH_CENTROID_CHILD] for row in pooled_rows])
    pitch = np.stack([row[SKETCH_PITCH_CHILD] for row in pooled_rows]).reshape(
        rows, SKETCH_PITCH_BINS, -1
    )
    actual = np.concatenate((loudness[:, None], centroid[:, None], pitch), axis=1)
    expected = pool_sketch_controls(torch.from_numpy(controls)).numpy()
    assert result.already_complete is False
    assert dataset.tags.get_version("before-local-pool") == 1
    np.testing.assert_array_equal(actual, expected)

    retry = backfill_sketch_pool(config)
    assert retry.already_complete is True
    assert lance.dataset(uri).version == committed_version


def test_backfill_sketch_pool_cli_with_untouched_source_tags_and_renames(
    fake_r2_remote: Path,
) -> None:
    """The console entrypoint prepares a fresh canonical source before pooling.

    :param fake_r2_remote: Filesystem-backed real rclone remote root.
    """
    rows = 2
    controls = np.zeros((rows, SKETCH_PITCH_BINS + 2, 401), dtype=np.float32)
    local_uri = fake_r2_remote / "test-bucket" / "fresh-sketches.lance"
    lance.write_dataset(
        pa.table({SKETCH_STRUCT_FIELD: sketch_struct_array(controls)}),
        local_uri,
    )
    result_path = fake_r2_remote / "fresh-result" / "result.json"
    command = [
        str(Path(sys.executable).parent / "synth-setter-backfill-sketch-pool"),
        "--lance-uri",
        "r2://test-bucket/fresh-sketches.lance",
        "--workers",
        "1",
        "--rollback-tag",
        "before-fresh-pool",
        "--no-build-index",
        "--resume-dir",
        str(fake_r2_remote / "fresh-resume"),
        "--result",
        str(result_path),
    ]

    completed = subprocess.run(  # noqa: S603 -- test owns the executable and every argument.
        command, check=False, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(result_path.read_text())
    dataset = lance.dataset(local_uri)
    assert payload["committed_version"] == dataset.version
    assert payload["rows"] == rows
    assert dataset.tags.get_version("before-fresh-pool") == 1
    assert {SKETCH_FULL_STRUCT_FIELD, SKETCH_STRUCT_FIELD} <= set(dataset.schema.names)


def test_retry_with_transient_failures_returns_eventual_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded retries recover after transient object-store failures.

    :param monkeypatch: Fixture disabling retry sleeps.
    """
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("transient")
        return "complete"

    monkeypatch.setattr(
        "synth_setter.pipeline.data.backfill_sketch_pool.time.sleep", lambda _: None
    )

    assert _retry("test", operation) == "complete"
    assert attempts == 3


def test_retry_with_transient_rclone_failure_returns_eventual_result() -> None:
    """Shared reconciliation I/O retries recognized object-store failures."""
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise subprocess.CalledProcessError(1, ["rclone"], stderr="503 Service Unavailable")
        return "ok"

    assert _retry("test_rclone", operation) == "ok"
    assert attempts == 2


def test_retry_with_permanent_failure_raises_without_retry() -> None:
    """Permission failures bypass transient transport retries."""
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError("permanent")

    with pytest.raises(PermissionError, match="permanent"):
        _retry("test", operation)
    assert attempts == 1


def test_poll_dispatch_with_short_deadline_limits_wait_and_cancels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task deadline bounds polling and force-cancels every pending reference.

    :param tmp_path: Temporary report-staging directory.
    :param monkeypatch: Fixture controlling Ray and monotonic time.
    """
    reference = cast("ray.ObjectRef", object())
    waits: list[float] = []
    cancellations: list[tuple[ray.ObjectRef, bool]] = []
    monkeypatch.setattr(
        ray,
        "wait",
        lambda pending, *, num_returns, timeout: (
            waits.append(timeout) or [],
            pending,
        ),
    )
    monkeypatch.setattr(
        ray,
        "cancel",
        lambda pending, *, force: cancellations.append((pending, force)),
    )
    clock = iter((100.0, 100.1))
    monkeypatch.setattr(
        "synth_setter.pipeline.data.backfill_sketch_pool.time.monotonic",
        lambda: next(clock),
    )
    state = _DispatchState(
        pending=[reference],
        reports={},
        fragment_ids={1},
        total_rows=1,
        report_store=_ReportStore(local_dir=tmp_path, remote_uri=None),
    )
    config = SketchPoolBackfillConfig(
        lance_uri=str(tmp_path / "source.lance"),
        workers=1,
        rollback_tag="before",
        timeout_seconds=0.05,
    )

    with pytest.raises(TimeoutError, match="0.05 seconds with 1 pending"):
        _poll_dispatch(state, config, 100.0)

    assert waits == [0.05]
    assert cancellations == [(reference, True)]


def test_resume_directory_overlapping_local_dataset_rejects_cleanup(tmp_path: Path) -> None:
    """A migration cannot designate its local dataset as disposable state.

    :param tmp_path: Temporary directory containing the protected source path.
    """
    uri = tmp_path / "source.lance"
    config = SketchPoolBackfillConfig(
        lance_uri=str(uri),
        workers=1,
        rollback_tag="before",
        resume_dir=uri,
    )

    with pytest.raises(ValueError, match="overlaps local dataset"):
        _resume_directory(config)


def test_resume_directory_with_file_uri_or_nested_result_rejects_cleanup(
    tmp_path: Path,
) -> None:
    """Local URI sources and result artifacts remain outside disposable state.

    :param tmp_path: Temporary directory containing protected paths.
    """
    uri = tmp_path / "source.lance"
    resume_dir = tmp_path / "resume"
    file_config = SketchPoolBackfillConfig(
        lance_uri=uri.as_uri(),
        workers=1,
        rollback_tag="before",
        resume_dir=uri,
    )
    result_config = SketchPoolBackfillConfig(
        lance_uri=str(uri),
        workers=1,
        rollback_tag="before",
        resume_dir=resume_dir,
        result=resume_dir / "result.json",
    )

    with pytest.raises(ValueError, match="overlaps local dataset"):
        _resume_directory(file_config)
    with pytest.raises(ValueError, match="result path .* is inside resume directory"):
        _resume_directory(result_config)


def test_resume_directory_without_override_is_dataset_keyed() -> None:
    """Default report caches separate public dataset and branch identities."""
    first = SketchPoolBackfillConfig(
        lance_uri="r2://bucket/a.lance",
        branch="candidate",
        workers=1,
        rollback_tag="before",
    )
    second = first.model_copy(update={"lance_uri": "r2://bucket/b.lance"})

    assert _resume_directory(first) != _resume_directory(second)


def test_result_when_below_index_threshold_records_skip_policy(tmp_path: Path) -> None:
    """A requested small-dataset index remains distinguishable from opt-out.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    dataset = lance.write_dataset(pa.table({"value": [1]}), tmp_path / "small.lance")
    config = SketchPoolBackfillConfig(
        lance_uri=str(dataset.uri),
        workers=1,
        rollback_tag="before",
    )

    result = _result(
        config,
        dataset,
        _ResultState(
            source_version=dataset.version,
            started=time.monotonic(),
            already_complete=True,
            index_built=False,
            run_id="run",
        ),
    )

    assert result.index_requested is True
    assert result.index_skip_reason == "below_min_rows"
    assert result.index_name == "sketch_pool_vec_idx"
    assert result.index_metric == "cosine"


def test_ensure_canonical_index_with_local_vectors_builds_and_recovers(
    tmp_path: Path,
) -> None:
    """Index publication and recovery validate the full canonical ANN contract.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    rows = 256
    dimensions = SKETCH_PITCH_BINS + 2
    vectors = np.random.default_rng(2980).random((rows, dimensions), dtype=np.float32)
    values = pa.array(vectors.reshape(-1), type=pa.float32())
    vector_array = pa.FixedSizeListArray.from_arrays(values, dimensions)
    sketch_array = pa.StructArray.from_arrays(
        [vector_array],
        fields=[pa.field("vec", vector_array.type)],
    )
    dataset = lance.write_dataset(
        pa.table({SKETCH_STRUCT_FIELD: sketch_array}),
        tmp_path / "index-source.lance",
    )
    config = SketchPoolBackfillConfig(
        lance_uri=str(dataset.uri),
        workers=1,
        rollback_tag="before",
        num_partitions=2,
    )

    indexed, built = _ensure_canonical_index(dataset, config)
    recovered, rebuilt = _ensure_canonical_index(indexed, config)

    assert built is True
    assert rebuilt is False
    assert recovered.version == indexed.version
    with pytest.raises(ValueError, match="config .* does not match"):
        _ensure_canonical_index(indexed, config.model_copy(update={"num_partitions": 1}))


def test_ensure_canonical_index_when_disabled_skips_small_dataset(tmp_path: Path) -> None:
    """An explicit no-index operation does not inspect or mutate vector indexes.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    uri = tmp_path / "no-index.lance"
    dataset = lance.write_dataset(pa.table({"value": [1]}), uri)
    config = SketchPoolBackfillConfig(
        lance_uri=str(uri),
        workers=1,
        rollback_tag="before",
        build_index=False,
    )

    unchanged, built = _ensure_canonical_index(dataset, config)

    assert unchanged.version == dataset.version
    assert built is False


def test_prepare_source_with_pooled_32_frame_sketch_rejects_before_rename(
    tmp_path: Path,
) -> None:
    """A canonical model-ready sketch cannot be renamed as historical source.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    uri = tmp_path / "pooled-source.lance"
    controls = np.zeros((1, SKETCH_PITCH_BINS + 2, 32), dtype=np.float32)
    dataset = lance.write_dataset(
        pa.table({SKETCH_STRUCT_FIELD: sketch_struct_array(controls)}),
        uri,
    )
    config = SketchPoolBackfillConfig(
        lance_uri=str(uri),
        workers=1,
        rollback_tag="before",
        build_index=False,
    )

    with pytest.raises(ValueError, match="does not match the historical 401-frame schema"):
        _prepare_source(dataset, config, str(uri), None)
    assert dataset.tags.list() == {}


def test_prepare_source_with_transposed_pitch_rejects_before_mutation(tmp_path: Path) -> None:
    """Pitch layout must preserve the required row-major bin and frame axes.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    rows = 1
    controls = np.zeros((rows, SKETCH_PITCH_BINS + 2, 401), dtype=np.float32)
    canonical = sketch_struct_array(controls)
    transposed_pitch = pa.FixedShapeTensorArray.from_numpy_ndarray(
        controls[:, 2:].transpose(0, 2, 1)
    )
    source = pa.StructArray.from_arrays(
        [canonical.field(index) for index in range(2)] + [transposed_pitch, canonical.field(3)],
        fields=[
            canonical.type.field(index)
            if index != 2
            else pa.field(SKETCH_PITCH_CHILD, transposed_pitch.type)
            for index in range(4)
        ],
    )
    uri = tmp_path / "transposed.lance"
    dataset = lance.write_dataset(pa.table({SKETCH_STRUCT_FIELD: source}), uri)
    config = SketchPoolBackfillConfig(
        lance_uri=str(uri), workers=1, rollback_tag="before", build_index=False
    )

    with pytest.raises(ValueError, match="does not match the historical 401-frame schema"):
        _prepare_source(dataset, config, str(uri), None)
    assert dataset.tags.list() == {}


def test_ensure_rollback_tag_without_existing_tag_rejects_resume(tmp_path: Path) -> None:
    """A post-rename resume cannot invent a missing rollback snapshot.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    uri = tmp_path / "missing-tag.lance"
    dataset = lance.write_dataset(pa.table({SKETCH_FULL_STRUCT_FIELD: [1]}), uri)
    config = SketchPoolBackfillConfig(
        lance_uri=str(uri),
        workers=1,
        rollback_tag="missing",
        build_index=False,
    )

    with pytest.raises(ValueError, match="must already exist"):
        _validate_rollback_tag(dataset, config, str(uri), None, expected_version=None)


def test_validate_rollback_tag_after_source_advance_rejects_old_version(tmp_path: Path) -> None:
    """A schema-compatible source commit cannot reuse an older rollback snapshot.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    uri = tmp_path / "advanced-source.lance"
    dataset = lance.write_dataset(pa.table({SKETCH_STRUCT_FIELD: [1]}), uri)
    dataset.tags.create("before", dataset.version)
    dataset = lance.write_dataset(
        pa.table({SKETCH_STRUCT_FIELD: [2]}),
        uri,
        mode="append",
    )
    config = SketchPoolBackfillConfig(
        lance_uri=str(uri),
        workers=1,
        rollback_tag="before",
        build_index=False,
    )

    with pytest.raises(ValueError, match="not source version"):
        _validate_rollback_tag(
            dataset,
            config,
            str(uri),
            None,
            expected_version=dataset.version,
        )


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
        _validate_rollback_tag(dataset, config, str(uri), None, expected_version=None)


def test_transform_fragment_with_missing_id_rejects_task(tmp_path: Path) -> None:
    """A worker cannot report a fragment absent from its immutable source version.

    :param tmp_path: Temporary directory for the real Lance source.
    """
    uri = tmp_path / "missing-fragment.lance"
    dataset = lance.write_dataset(pa.table({SKETCH_FULL_STRUCT_FIELD: [1]}), uri)

    with pytest.raises(ValueError, match="missing fragment 999"):
        _transform_fragment(
            _FragmentTask(
                uri=str(uri),
                storage_options=None,
                branch="main",
                source_version=dataset.version,
                fragment_id=999,
                batch_size=1,
                artifact=_sketch_pool_artifact_identity(""),
            )
        )


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
