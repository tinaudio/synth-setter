"""Finalize-side Lance fragment commit (#1776): winners → split manifests, zero row decode.

E2e over real Lance and the real ``rclone`` binary against the ``fake_r2_remote``
local remote — no mocks. Every expected array is recomputed directly in NumPy
from the values the staging step wrote, never through the codec under test.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.cli.finalize_dataset import finalize_from_spec
from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES, MEL_SPEC_FIELD
from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows, read_shard_metadata
from synth_setter.pipeline.data.lance_staging import stage_lance_shard_attempt
from synth_setter.pipeline.schemas.lance_attempt import LanceDatasetCard
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.pipeline.data.test_lance_staging import (
    shard_arrays,
    staging_dir,
    tiny_lance_spec,
    write_local_shard,
)

pytestmark = pytest.mark.usefixtures("fake_r2_remote")


def stage_all_shards(spec: DatasetSpec, tmp_path: Path, *, worker_id: str = "pod-a") -> None:
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


def split_dataset_path(fake_r2_remote: Path, spec: DatasetSpec, split: str) -> Path:
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
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)

    finalize_from_spec(spec, tmp_path / "work")

    expected_split_shards = {"train": [0, 1], "val": [2], "test": [3]}
    for split, shard_ids in expected_split_shards.items():
        split_path = split_dataset_path(fake_r2_remote, spec, split)
        dataset = lance.dataset(str(split_path))
        first_shard_id = shard_ids[0]
        worker_shard = lance.dataset(
            str(tmp_path / f"w-{first_shard_id}" / spec.shards[first_shard_id].filename)
        )
        assert dataset.schema == worker_shard.schema
        assert read_shard_metadata(dataset.schema).base_seed == spec.shards[first_shard_id].seed

        decoded = read_columns(split_path)
        for field in DATASET_FIELD_NAMES:
            expected = np.concatenate(
                [shard_arrays(spec, sid)[field] for sid in shard_ids], axis=0
            )
            np.testing.assert_array_equal(decoded[field], expected)


def test_finalize_split_commit_is_one_atomic_manifest_version(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)

    finalize_from_spec(spec, tmp_path / "work")

    train = lance.dataset(str(split_dataset_path(fake_r2_remote, spec, "train")))
    assert train.version == 1
    assert len(train.get_fragments()) == 2


def test_finalize_stats_npz_matches_direct_recompute_over_train_mel_rows(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)

    finalize_from_spec(spec, tmp_path / "work")

    stats_path = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "stats.npz"
    with np.load(stats_path) as stats:
        stats_mean, stats_std = stats["mean"], stats["std"]
    train_mel = np.concatenate(
        [shard_arrays(spec, sid)[MEL_SPEC_FIELD] for sid in (0, 1)], axis=0
    ).astype(np.float64)
    np.testing.assert_allclose(stats_mean, train_mel.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(stats_std, train_mel.std(axis=0), rtol=1e-6)


def set_valid_marker_mtime(
    fake_r2_remote: Path, spec: DatasetSpec, shard_id: int, name: str, epoch: float
) -> None:
    """Pin one staged attempt's ``.valid`` LastModified via the real storage state.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param spec: Spec whose R2 location shapes the staging prefix.
    :param shard_id: Shard whose attempt marker is pinned.
    :param name: Attempt name (``{worker}-{attempt}``).
    :param epoch: POSIX timestamp to stamp on the marker.
    """
    marker = staging_dir(fake_r2_remote, spec, shard_id) / f"{name}.valid"
    os.utime(marker, (epoch, epoch))


def stage_duplicate_attempt(
    spec: DatasetSpec, tmp_path: Path, shard_id: int, *, attempt_uuid: str, value_offset: int
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


@pytest.mark.parametrize("flag", [True, False])
def test_finalize_forwards_mask_degenerate_bins_to_welford_finalize(
    fake_r2_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: bool,
) -> None:
    """``spec.mask_degenerate_bins`` reaches the Welford finalize verbatim.

    Pins the wire on both polarities, mirroring the hdf5/wds forwarding tests,
    so a regression that hard-wires the kwarg fails here.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param tmp_path: Scratch dir for the local shard datasets.
    :param monkeypatch: Pytest fixture used to capture the forwarded kwarg.
    :param flag: Parametrized polarity threaded through the spec field.
    """
    from synth_setter.pipeline.data import lance_finalize

    spec = DatasetSpec.model_validate(
        {**tiny_lance_spec().model_dump(mode="json"), "mask_degenerate_bins": flag}
    )
    stage_all_shards(spec, tmp_path)
    captured: dict[str, bool] = {}
    real_finalize = lance_finalize.finalize_welford

    def capture_finalize(existing: object, mask_degenerate: bool = False) -> object:
        captured["mask_degenerate"] = mask_degenerate
        return real_finalize(existing, mask_degenerate=mask_degenerate)

    monkeypatch.setattr(lance_finalize, "finalize_welford", capture_finalize)

    finalize_from_spec(spec, tmp_path / "work")

    assert captured == {"mask_degenerate": flag}


def test_finalize_records_selected_attempts_and_valid_keys_in_dataset_json(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
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


def staging_file(fake_r2_remote: Path, spec: DatasetSpec, shard_id: int, filename: str) -> Path:
    """Local path of one staged artifact under the fake R2 root.

    :param fake_r2_remote: Root the ``r2:`` remote resolves to.
    :param spec: Spec whose R2 location shapes the staging prefix.
    :param shard_id: Shard whose staging directory holds the artifact.
    :param filename: Staged artifact filename (``{worker}-{attempt}{suffix}``).
    :returns: ``<root>/<bucket>/<prefix>metadata/workers/shards/shard-NNNNNN/<filename>``.
    """
    return staging_dir(fake_r2_remote, spec, shard_id) / filename


def test_finalize_rejects_sidecar_that_fails_strict_validation(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    sidecar = staging_file(fake_r2_remote, spec, 0, "pod-a-u0000.fragment.json")
    sidecar.write_text('{"schema_version": 1, "fragment_json": 42}')

    with pytest.raises(ValueError, match="invalid fragment sidecar"):
        finalize_from_spec(spec, tmp_path / "work")


def test_finalize_rejects_binary_fragment_sidecar_with_context(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    sidecar = staging_file(fake_r2_remote, spec, 0, "pod-a-u0000.fragment.json")
    sidecar.write_bytes(b"\xff")

    with pytest.raises(ValueError, match=r"shard 0 attempt pod-a-u0000: invalid fragment sidecar"):
        finalize_from_spec(spec, tmp_path / "work")


def test_finalize_rejects_fragment_json_that_is_not_lance_metadata(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    sidecar = staging_file(fake_r2_remote, spec, 0, "pod-a-u0000.fragment.json")
    # Passes the strict Pydantic parse (valid JSON string field) but is not a
    # Lance fragment payload — lance raises KeyError, not ValueError.
    sidecar.write_text('{"schema_version": 1, "fragment_json": "{\\"bogus\\": 1}"}')

    with pytest.raises(ValueError, match="does not deserialize as Lance fragment metadata"):
        finalize_from_spec(spec, tmp_path / "work")


def test_finalize_skips_empty_split_and_still_completes(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    # train 4 + val 2 samples at 2/shard → 3 shards, test split empty. Rebuilt
    # from a fresh dump so the frozen spec's cached computed fields re-derive.
    spec = DatasetSpec.model_validate(
        {**tiny_lance_spec().model_dump(mode="json"), "train_val_test_sizes": [4, 2, 0]}
    )
    stage_all_shards(spec, tmp_path)

    finalize_from_spec(spec, tmp_path / "work")

    run_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    assert (run_root / "train.lance" / "_versions").exists()
    assert (run_root / "val.lance" / "_versions").exists()
    assert not (run_root / "test.lance").exists()
    assert (run_root / "dataset.complete").exists()


def test_finalize_rejects_stats_sidecar_missing_welford_arrays(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    stats_path = staging_file(fake_r2_remote, spec, 0, "pod-a-u0000.shard-stats.npz")
    np.savez(stats_path, count=np.int64(2))

    with pytest.raises(ValueError, match=r"missing arrays \['mean', 'm2'\]"):
        finalize_from_spec(spec, tmp_path / "work")


def test_finalize_rejects_malformed_stats_sidecar_with_context(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    stats_path = staging_file(fake_r2_remote, spec, 0, "pod-a-u0000.shard-stats.npz")
    stats_path.write_bytes(b"")

    with pytest.raises(ValueError, match=r"shard 0 attempt pod-a-u0000: invalid shard-stats\.npz"):
        finalize_from_spec(spec, tmp_path / "work")


def test_finalize_rejects_stats_count_disagreeing_with_fragment_rows(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    stats_path = staging_file(fake_r2_remote, spec, 0, "pod-a-u0000.shard-stats.npz")
    with np.load(stats_path) as stats:
        arrays = dict(stats)
    arrays["count"] = np.int64(999)
    np.savez(stats_path, **arrays)

    with pytest.raises(ValueError, match="stats count 999"):
        finalize_from_spec(spec, tmp_path / "work")


def test_finalize_rejects_fragment_whose_rows_disagree_with_spec(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    sidecar_path = staging_file(fake_r2_remote, spec, 0, "pod-a-u0000.fragment.json")
    payload = json.loads(sidecar_path.read_text())
    fragment_meta = json.loads(payload["fragment_json"])
    fragment_meta["physical_rows"] = 7
    payload["fragment_json"] = json.dumps(fragment_meta)
    sidecar_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fragment has 7 rows"):
        finalize_from_spec(spec, tmp_path / "work")


def test_finalize_rejects_winner_whose_fragment_data_file_is_absent(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    # Shard 2 is val's only shard, so its fragment file is val.lance's only data.
    val_data = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "val.lance" / "data"
    for fragment_file in val_data.iterdir():
        fragment_file.unlink()

    with pytest.raises(ValueError, match="missing or empty under"):
        finalize_from_spec(spec, tmp_path / "work")


def test_finalize_rejects_winner_whose_fragment_data_file_is_truncated_to_zero(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    val_data = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "val.lance" / "data"
    for fragment_file in val_data.iterdir():
        fragment_file.write_bytes(b"")

    with pytest.raises(ValueError, match="missing or empty under"):
        finalize_from_spec(spec, tmp_path / "work")


def test_select_winner_prefers_earliest_valid_mtime() -> None:
    from synth_setter.pipeline.data.lance_finalize import StagedLanceAttempt, select_winner

    early = StagedLanceAttempt(
        shard_id=0,
        name="pod-z-late-key",
        valid_key="z/pod-z.valid",
        valid_mtime=datetime(2026, 1, 1, tzinfo=UTC),
    )
    late = StagedLanceAttempt(
        shard_id=0,
        name="pod-a-early-key",
        valid_key="a/pod-a.valid",
        valid_mtime=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert select_winner([late, early]) is early


def test_select_winner_breaks_mtime_ties_by_lexicographic_key() -> None:
    from synth_setter.pipeline.data.lance_finalize import StagedLanceAttempt, select_winner

    tied = datetime(2026, 1, 1, tzinfo=UTC)
    key_b = StagedLanceAttempt(shard_id=0, name="b", valid_key="dir/b.valid", valid_mtime=tied)
    key_a = StagedLanceAttempt(shard_id=0, name="a", valid_key="dir/a.valid", valid_mtime=tied)

    assert select_winner([key_b, key_a]) is key_a


def test_staged_discovery_skips_non_shard_entries_in_staging_root(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    from synth_setter.pipeline.data.lance_finalize import staged_complete_attempts

    spec = tiny_lance_spec()
    stage_all_shards(spec, tmp_path)
    staging_root = (
        fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "metadata" / "workers" / "shards"
    )
    stray_dir = staging_root / "quarantine"
    stray_dir.mkdir()
    (stray_dir / "pod-x-dead.h5").write_bytes(b"x")
    (staging_root / "stray-top-level.txt").write_bytes(b"x")
    # A full quarantined triple nested under a shard dir is not a staged attempt.
    shard_quarantine = staging_dir(fake_r2_remote, spec, 0) / "quarantine"
    shard_quarantine.mkdir()
    for suffix in (".fragment.json", ".shard-stats.npz", ".valid"):
        (shard_quarantine / f"pod-q-dead{suffix}").write_bytes(b"x")

    attempts = staged_complete_attempts(spec)

    assert sorted(attempts) == [0, 1, 2, 3]
    assert [attempt.name for attempt in attempts[0]] == ["pod-a-u0000"]


def test_finalize_with_missing_shard_reports_it_and_writes_no_marker(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
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
