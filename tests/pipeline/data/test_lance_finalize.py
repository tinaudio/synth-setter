"""Finalize-side Lance fragment commit (#1776): winners → split manifests, zero row decode.

E2e over real Lance and the real ``rclone`` binary against the ``fake_r2_remote``
local remote — no mocks. Every expected array is recomputed directly in NumPy
from the values the staging step wrote, never through the codec under test.
"""

from __future__ import annotations

import os
from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES, MEL_SPEC_FIELD
from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows
from synth_setter.pipeline.data.lance_staging import stage_lance_shard_attempt
from synth_setter.pipeline.schemas.lance_attempt import LanceDatasetCard
from tests.pipeline.data.test_lance_staging import (
    shard_arrays,
    tiny_lance_spec,
    write_local_shard,
)

pytestmark = pytest.mark.usefixtures("fake_r2_remote")


def stage_all_shards(spec, tmp_path: Path, *, worker_id: str = "pod-a") -> None:
    """Stage a complete attempt for every shard in ``spec``.

    :param spec: Spec whose shards are staged.
    :param tmp_path: Scratch dir for the local shard datasets.
    :param worker_id: Worker identifier used in every staging filename.
    """
    for shard in spec.shards:
        local = write_local_shard(spec, shard.shard_id, tmp_path / f"w-{shard.shard_id}")
        stage_lance_shard_attempt(
            spec, shard, local, worker_id=worker_id, attempt_uuid=f"u{shard.shard_id:04d}"
        )


def split_dataset_path(fake_r2_remote: Path, spec, split: str) -> Path:
    """Local path of a committed split dataset under the fake R2 root.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param spec: Spec whose R2 location shapes the run prefix.
    :param split: Split name.
    :returns: ``<root>/<bucket>/<prefix><split>.lance``.
    """
    return fake_r2_remote / spec.r2.bucket / spec.r2.prefix / f"{split}.lance"


def read_columns(uri: Path) -> dict[str, np.ndarray]:
    """Read every dataset field back from a committed Lance dataset (real data read).

    :param uri: Committed Lance dataset directory.
    :returns: Field name to row-stacked decoded array.
    """
    return {
        field: np.stack(list(iter_lance_column_rows(uri, field)), axis=0)
        for field in DATASET_FIELD_NAMES
    }


def test_finalize_commits_winners_into_three_splits_with_exact_shard_content(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)

    finalize_from_spec(spec, tmp_path / "work")

    expected_split_shards = {"train": [0, 1], "val": [2], "test": [3]}
    for split, shard_ids in expected_split_shards.items():
        decoded = read_columns(split_dataset_path(fake_r2_remote, spec, split))
        for field in DATASET_FIELD_NAMES:
            expected = np.concatenate(
                [shard_arrays(spec, sid)[field] for sid in shard_ids], axis=0
            )
            np.testing.assert_array_equal(decoded[field], expected)


def test_finalize_split_commit_is_one_atomic_manifest_version(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)

    finalize_from_spec(spec, tmp_path / "work")

    train = lance.dataset(str(split_dataset_path(fake_r2_remote, spec, "train")))
    assert train.version == 1
    assert len(train.get_fragments()) == 2


def test_finalize_stats_npz_matches_direct_recompute_over_train_mel_rows(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)

    finalize_from_spec(spec, tmp_path / "work")

    stats_path = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "stats.npz"
    stats = np.load(stats_path)
    train_mel = np.concatenate(
        [shard_arrays(spec, sid)[MEL_SPEC_FIELD] for sid in (0, 1)], axis=0
    ).astype(np.float64)
    np.testing.assert_allclose(stats["mean"], train_mel.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(stats["std"], train_mel.std(axis=0), rtol=1e-6)


def set_valid_marker_mtime(
    fake_r2_remote: Path, spec, shard_id: int, name: str, epoch: float
) -> None:
    """Pin one staged attempt's ``.valid`` LastModified via the real storage state.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param spec: Spec whose R2 location shapes the staging prefix.
    :param shard_id: Shard whose attempt marker is pinned.
    :param name: Attempt name (``{worker}-{attempt}``).
    :param epoch: POSIX timestamp to stamp on the marker.
    """
    marker = (
        fake_r2_remote
        / spec.r2.bucket
        / spec.r2.prefix
        / "metadata"
        / "workers"
        / "shards"
        / f"shard-{shard_id:06d}"
        / f"{name}.valid"
    )
    os.utime(marker, (epoch, epoch))


def stage_duplicate_attempt(
    spec, tmp_path: Path, shard_id: int, *, attempt_uuid: str, value_offset: int
) -> None:
    """Stage one more attempt for ``shard_id`` with distinguishable content.

    :param spec: Spec whose shard is re-attempted.
    :param tmp_path: Scratch dir root for the local shard dataset.
    :param shard_id: Shard receiving the duplicate attempt.
    :param attempt_uuid: Attempt UUID for the staging filenames.
    :param value_offset: Content offset so the duplicate is tell-apart-able.
    """
    local = write_local_shard(
        spec, shard_id, tmp_path / f"dup-{attempt_uuid}", value_offset=value_offset
    )
    stage_lance_shard_attempt(
        spec, spec.shards[shard_id], local, worker_id="pod-b", attempt_uuid=attempt_uuid
    )


def test_finalize_selects_earliest_valid_marker_among_duplicate_attempts(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    stage_duplicate_attempt(spec, tmp_path, 0, attempt_uuid="zzzz", value_offset=7000)
    # The duplicate's marker is older, so it must win despite staging later.
    set_valid_marker_mtime(fake_r2_remote, spec, 0, "pod-a-u0000", epoch=2_000_000_000.0)
    set_valid_marker_mtime(fake_r2_remote, spec, 0, "pod-b-zzzz", epoch=1_000_000_000.0)

    finalize_from_spec(spec, tmp_path / "work")

    decoded = read_columns(split_dataset_path(fake_r2_remote, spec, "train"))
    winner_mel = shard_arrays(spec, 0, value_offset=7000)[MEL_SPEC_FIELD]
    np.testing.assert_array_equal(decoded[MEL_SPEC_FIELD][:2], winner_mel)
    assert decoded[MEL_SPEC_FIELD].shape[0] == 4


def test_finalize_ties_on_valid_mtime_break_by_lexicographic_marker_key(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    stage_duplicate_attempt(spec, tmp_path, 0, attempt_uuid="zzzz", value_offset=7000)
    set_valid_marker_mtime(fake_r2_remote, spec, 0, "pod-a-u0000", epoch=1_500_000_000.0)
    set_valid_marker_mtime(fake_r2_remote, spec, 0, "pod-b-zzzz", epoch=1_500_000_000.0)

    finalize_from_spec(spec, tmp_path / "work")

    decoded = read_columns(split_dataset_path(fake_r2_remote, spec, "train"))
    # "pod-a-u0000.valid" < "pod-b-zzzz.valid" lexicographically → pod-a wins.
    np.testing.assert_array_equal(
        decoded[MEL_SPEC_FIELD][:2], shard_arrays(spec, 0)[MEL_SPEC_FIELD]
    )


def test_finalize_rerun_short_circuits_on_complete_marker_with_identical_content(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    finalize_from_spec(spec, tmp_path / "work")
    train_path = split_dataset_path(fake_r2_remote, spec, "train")
    first = read_columns(train_path)
    first_version = lance.dataset(str(train_path)).version

    finalize_from_spec(spec, tmp_path / "work2")

    again = read_columns(train_path)
    assert lance.dataset(str(train_path)).version == first_version
    for field in DATASET_FIELD_NAMES:
        np.testing.assert_array_equal(again[field], first[field])


def test_finalize_rerun_after_lost_marker_keeps_winner_despite_later_straggler(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    for shard in spec.shards:
        set_valid_marker_mtime(
            fake_r2_remote, spec, shard.shard_id, f"pod-a-u{shard.shard_id:04d}", 1_000_000_000.0
        )
    finalize_from_spec(spec, tmp_path / "work")
    run_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    (run_root / "dataset.complete").unlink()
    # A straggler lands with a strictly later LastModified: selection must be monotonic.
    stage_duplicate_attempt(spec, tmp_path, 0, attempt_uuid="strg", value_offset=7000)
    set_valid_marker_mtime(fake_r2_remote, spec, 0, "pod-b-strg", epoch=2_000_000_000.0)

    finalize_from_spec(spec, tmp_path / "work2")

    decoded = read_columns(split_dataset_path(fake_r2_remote, spec, "train"))
    np.testing.assert_array_equal(
        decoded[MEL_SPEC_FIELD][:2], shard_arrays(spec, 0)[MEL_SPEC_FIELD]
    )
    assert decoded[MEL_SPEC_FIELD].shape[0] == 4


def test_finalize_interrupted_before_marker_rerun_rebuilds_without_doubled_rows(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    finalize_from_spec(spec, tmp_path / "work")
    run_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    # Simulate a crash after the split commits but before the marker landed.
    (run_root / "dataset.complete").unlink()
    (run_root / "stats.npz").unlink()

    finalize_from_spec(spec, tmp_path / "work2")

    for split, expected_rows in (("train", 4), ("val", 2), ("test", 2)):
        decoded = read_columns(split_dataset_path(fake_r2_remote, spec, split))
        assert decoded[MEL_SPEC_FIELD].shape[0] == expected_rows
    assert (run_root / "dataset.complete").exists()
    assert (run_root / "stats.npz").exists()


def test_finalize_records_selected_attempts_and_valid_keys_in_dataset_json(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    stage_duplicate_attempt(spec, tmp_path, 0, attempt_uuid="zzzz", value_offset=7000)
    set_valid_marker_mtime(fake_r2_remote, spec, 0, "pod-a-u0000", epoch=2_000_000_000.0)
    set_valid_marker_mtime(fake_r2_remote, spec, 0, "pod-b-zzzz", epoch=1_000_000_000.0)

    finalize_from_spec(spec, tmp_path / "work")

    card_path = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "dataset.json"
    card = LanceDatasetCard.model_validate_json(card_path.read_text())
    assert card.run_id == spec.run_id
    selected = {attempt.shard_id: attempt for attempt in card.selected_attempts}
    assert sorted(selected) == [0, 1, 2, 3]
    assert selected[0].attempt == "pod-b-zzzz"
    assert selected[0].valid_key == (
        f"{spec.r2.prefix}metadata/workers/shards/shard-000000/pod-b-zzzz.valid"
    )


def test_finalize_with_missing_shard_reports_it_and_writes_no_marker(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.cli.finalize_dataset import finalize_from_spec

    spec = tiny_lance_spec()
    for shard in spec.shards:
        if shard.shard_id == 1:
            continue
        local = write_local_shard(spec, shard.shard_id, tmp_path / f"w-{shard.shard_id}")
        stage_lance_shard_attempt(
            spec, shard, local, worker_id="pod-a", attempt_uuid=f"u{shard.shard_id:04d}"
        )

    with pytest.raises(ValueError, match="shard-000001"):
        finalize_from_spec(spec, tmp_path / "work")

    run_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    assert not (run_root / "dataset.complete").exists()
    assert not (run_root / "train.lance" / "_versions").exists()
