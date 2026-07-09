"""Worker-side Lance attempt staging (#1776): fragment + sidecar + stats + markers.

Every test drives the real codec on real storage — local shards written with
the production ``lance_shard`` helpers, staging IO through the real ``rclone``
binary against the ``fake_r2_remote`` local remote, no mocks. Expected arrays
are built directly in NumPy so a codec bug cannot corrupt both sides.
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    DATASET_FIELD_DTYPES,
    DATASET_FIELD_NAMES,
    dataset_field_shapes,
)
from synth_setter.pipeline.data.lance_shard import (
    lance_schema,
    record_batch_from_arrays,
    write_lance_dataset,
)
from synth_setter.pipeline.data.lance_staging import (
    shard_has_complete_attempt,
    split_for_shard,
    stage_lance_shard_attempt,
    write_rendering_marker,
)
from synth_setter.pipeline.schemas.lance_attempt import LanceFragmentSidecar
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec

pytestmark = pytest.mark.usefixtures("fake_r2_remote")


def tiny_lance_spec() -> DatasetSpec:
    """Build a 4-shard lance spec (train [0,2), val [2,3), test [3,4)) with tiny rows.

    :returns: Frozen spec whose render config keeps every per-field array small.
    """
    return DatasetSpec.model_validate(
        {
            "task_name": "lance-frag-test",
            "output_format": "lance",
            "train_val_test_sizes": [4, 2, 2],
            "base_seed": 42,
            "r2": {"bucket": "intermediate-data"},
            "render": {
                "plugin_path": "plugins/Surge XT.vst3",
                "preset_path": "presets/surge-base.vstpreset",
                "param_spec_name": "surge_simple",
                "renderer_version": "1.3.4",
                "sample_rate": 100,
                "channels": 2,
                "velocity": 100,
                "signal_duration_seconds": 0.5,
                "min_loudness": -55.0,
                "samples_per_render_batch": 2,
                "samples_per_shard": 2,
                "gui_toggle_cadence": "never",
            },
        }
    )


def shard_arrays(spec: DatasetSpec, shard_id: int, value_offset: int = 0) -> dict[str, np.ndarray]:
    """Build one shard's field arrays with values disjoint across shards.

    Audio offsets stay below 2048 so every float16 value is exact (the attempt
    offset is capped for audio); the other fields use a full-field stride so
    no value repeats across shards.

    :param spec: Spec whose render config fixes the per-field shapes.
    :param shard_id: Logical shard whose values are offset.
    :param value_offset: Extra per-field offset distinguishing duplicate attempts.
    :returns: Mapping keyed by ``DATASET_FIELD_NAMES`` with writer dtypes.
    """
    shapes = dataset_field_shapes(spec.render, spec.num_params)
    arrays = {}
    for field, shape in shapes.items():
        size = int(np.prod(shape))
        stride = 300 if field == AUDIO_FIELD else size + 17
        offset = shard_id * stride + (
            min(value_offset, 100) if field == AUDIO_FIELD else value_offset
        )
        arrays[field] = np.arange(
            offset, offset + size, dtype=DATASET_FIELD_DTYPES[field]
        ).reshape(shape)
    return arrays


def write_local_shard(
    spec: DatasetSpec, shard_id: int, work_dir: Path, *, value_offset: int = 0
) -> Path:
    """Write one shard's local Lance dataset exactly as the worker's renderer does.

    :param spec: Spec supplying shapes and per-shard seed.
    :param shard_id: Logical shard to materialize.
    :param work_dir: Local scratch directory for the shard dataset.
    :param value_offset: Extra value offset distinguishing duplicate attempts.
    :returns: Path of the written ``shard-NNNNNN.lance`` dataset directory.
    """
    shard = spec.shards[shard_id]
    render = spec.render.model_copy(update={"base_seed": shard.seed})
    metadata = render.shard_metadata()
    schema = lance_schema(dataset_field_shapes(render, spec.num_params), metadata)
    batch = record_batch_from_arrays(shard_arrays(spec, shard_id, value_offset), schema)
    shard_path = work_dir / shard.filename
    write_lance_dataset(shard_path, schema, [batch])
    return shard_path


def staging_dir(fake_r2_remote: Path, spec: DatasetSpec, shard_id: int) -> Path:
    """Local path of one shard's staging directory under the fake R2 root.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param spec: Spec whose R2 location shapes the staging prefix.
    :param shard_id: Logical shard the staging directory belongs to.
    :returns: ``<root>/<bucket>/<prefix>metadata/workers/shards/shard-NNNNNN``.
    """
    return (
        fake_r2_remote
        / spec.r2.bucket
        / spec.r2.prefix
        / "metadata"
        / "workers"
        / "shards"
        / f"shard-{shard_id:06d}"
    )


def test_split_for_shard_follows_spec_split_ranges() -> None:
    spec = tiny_lance_spec()
    assert [split_for_shard(spec, i) for i in range(4)] == ["train", "train", "val", "test"]


def test_split_for_shard_out_of_range_raises() -> None:
    spec = tiny_lance_spec()
    with pytest.raises(ValueError, match="shard_id 4 outside spec ranges"):
        split_for_shard(spec, 4)


def test_stage_attempt_writes_fragment_data_into_assigned_split_dataset_dir(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    local_shard = write_local_shard(spec, 2, tmp_path)

    stage_lance_shard_attempt(
        spec, spec.shards[2], local_shard, worker_id="pod-a", attempt_uuid="a1b2"
    )

    val_data_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "val.lance" / "data"
    assert [p.suffix for p in val_data_dir.iterdir()] == [".lance"]


def test_stage_attempt_sidecar_round_trips_fragment_metadata_with_shard_row_count(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    local_shard = write_local_shard(spec, 0, tmp_path)

    stage_lance_shard_attempt(
        spec, spec.shards[0], local_shard, worker_id="pod-a", attempt_uuid="a1b2"
    )

    sidecar_path = staging_dir(fake_r2_remote, spec, 0) / "pod-a-a1b2.fragment.json"
    sidecar = LanceFragmentSidecar.model_validate_json(sidecar_path.read_text())
    assert sidecar.schema_version == 1
    fragment = lance.fragment.FragmentMetadata.from_json(sidecar.fragment_json)
    assert fragment.to_json()["physical_rows"] == spec.render.samples_per_shard


def test_stage_attempt_stats_sidecar_matches_direct_welford_over_written_mel_rows(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    local_shard = write_local_shard(spec, 0, tmp_path)

    stage_lance_shard_attempt(
        spec, spec.shards[0], local_shard, worker_id="pod-a", attempt_uuid="a1b2"
    )

    stats_path = staging_dir(fake_r2_remote, spec, 0) / "pod-a-a1b2.shard-stats.npz"
    with np.load(stats_path) as stats:
        mel_rows = shard_arrays(spec, 0)["mel_spec"].astype(np.float64)
        assert int(stats["count"]) == spec.render.samples_per_shard
        np.testing.assert_allclose(stats["mean"], mel_rows.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(
            stats["m2"], ((mel_rows - mel_rows.mean(axis=0)) ** 2).sum(axis=0), rtol=1e-9
        )


def test_stage_attempt_writes_valid_marker_alongside_sidecars(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    local_shard = write_local_shard(spec, 0, tmp_path)

    stage_lance_shard_attempt(
        spec, spec.shards[0], local_shard, worker_id="pod-a", attempt_uuid="a1b2"
    )

    assert (staging_dir(fake_r2_remote, spec, 0) / "pod-a-a1b2.valid").exists()


def test_write_rendering_marker_records_attempt_start(fake_r2_remote: Path) -> None:
    spec = tiny_lance_spec()

    write_rendering_marker(spec, 1, worker_id="pod-a", attempt_uuid="a1b2")

    assert (staging_dir(fake_r2_remote, spec, 1) / "pod-a-a1b2.rendering").exists()


def test_shard_has_complete_attempt_false_before_staging_true_after(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    local_shard = write_local_shard(spec, 0, tmp_path)
    assert shard_has_complete_attempt(spec, 0) is False

    stage_lance_shard_attempt(
        spec, spec.shards[0], local_shard, worker_id="pod-a", attempt_uuid="a1b2"
    )

    assert shard_has_complete_attempt(spec, 0) is True


def test_shard_has_complete_attempt_ignores_partial_attempt_missing_stats(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    local_shard = write_local_shard(spec, 0, tmp_path)
    stage_lance_shard_attempt(
        spec, spec.shards[0], local_shard, worker_id="pod-a", attempt_uuid="a1b2"
    )
    (staging_dir(fake_r2_remote, spec, 0) / "pod-a-a1b2.shard-stats.npz").unlink()

    assert shard_has_complete_attempt(spec, 0) is False


def test_stage_attempt_rejects_local_shard_with_wrong_row_count(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    shard = spec.shards[0]
    render = spec.render.model_copy(update={"base_seed": shard.seed, "samples_per_shard": 3})
    schema = lance_schema(dataset_field_shapes(render, spec.num_params), render.shard_metadata())
    oversized = {
        field: np.repeat(shard_arrays(spec, 0)[field][:1], 3, axis=0)
        for field in DATASET_FIELD_NAMES
    }
    shard_path = tmp_path / shard.filename
    write_lance_dataset(shard_path, schema, [record_batch_from_arrays(oversized, schema)])

    with pytest.raises(ValueError, match="row"):
        stage_lance_shard_attempt(spec, shard, shard_path, worker_id="pod-a", attempt_uuid="a1b2")

    assert not (staging_dir(fake_r2_remote, spec, 0) / "pod-a-a1b2.valid").exists()


def test_stage_attempt_rejects_shard_exceeding_single_data_file_bound(
    fake_r2_remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tiny_lance_spec()
    local_shard = write_local_shard(spec, 0, tmp_path)
    # Shrink the bound rather than materializing a >32 GiB shard; the guard's
    # comparison is what's under test (#1775).
    monkeypatch.setattr("synth_setter.pipeline.data.lance_shard.LANCE_MAX_BYTES_PER_FILE", 1024)

    with pytest.raises(ValueError, match="multipart-part ceiling"):
        stage_lance_shard_attempt(
            spec, spec.shards[0], local_shard, worker_id="pod-a", attempt_uuid="a1b2"
        )

    assert not (staging_dir(fake_r2_remote, spec, 0) / "pod-a-a1b2.valid").exists()
    train_data = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "train.lance" / "data"
    assert not train_data.exists()


def test_staged_sidecar_survives_json_round_trip_from_disk(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    local_shard = write_local_shard(spec, 0, tmp_path)
    stage_lance_shard_attempt(
        spec, spec.shards[0], local_shard, worker_id="pod-a", attempt_uuid="a1b2"
    )

    raw = (staging_dir(fake_r2_remote, spec, 0) / "pod-a-a1b2.fragment.json").read_text()
    payload = json.loads(raw)
    assert set(payload) == {"schema_version", "fragment_json"}
