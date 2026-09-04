"""Rolling Lance window and publication contracts."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest
from pydantic import ValidationError

from synth_setter.pipeline.data.lance_shard import commit_lance_dataset, lance_schema
from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES, PARAM_ARRAY_FIELD
from synth_setter.pipeline.data.rolling_lance import (
    ActiveRollingSnapshot,
    PendingRefreshRequest,
    RollingSnapshot,
    RollingWindow,
    initialize_rolling_branch,
    materialize_and_activate,
    pending_refresh_request,
    publish_rolling_branch,
)
from tests.pipeline.data.test_lance_fragment_finalize_poc import (
    _FIELD_SHAPES,
    _METADATA,
    _arange_arrays,
    _worker_writes_fragment,
)
from tests.pipeline.data.test_lance_staging import tiny_lance_spec


def test_first_refresh_advances_relative_high_watermark_and_keeps_baseline_size() -> None:
    """The first N extras replace the oldest N baseline shards."""
    window = RollingWindow.from_spec(tiny_lance_spec(), num_extra_shards=1)

    pending = window.next_refresh()

    assert window.high_watermark == 2
    assert pending.enqueue_relative_ids == (2,)
    assert pending.membership_relative_ids == (1, 2)
    assert pending.next_high_watermark == 3


def test_pending_refresh_binds_exact_source_snapshot() -> None:
    """Enqueue freezes one range against the current ready transaction."""
    current = RollingSnapshot(
        branch="rolling",
        branch_uri="/tmp/train.lance/tree/rolling",
        version=7,
        baseline_version=1,
        baseline_transaction="baseline-tx",
        transaction="ready-tx",
        window_size=2,
        num_extra_shards=1,
        high_watermark=2,
        membership_relative_ids=(0, 1),
        dataset_spec_fingerprint="spec-fingerprint",
        row_count=4,
        schema_fingerprint="schema-fingerprint",
        stats_sha256="stats-sha256",
    )

    pending = pending_refresh_request(current)

    assert pending.source_version == 7
    assert pending.source_transaction == "ready-tx"
    assert pending.enqueue_relative_ids == (2,)
    assert pending.membership_relative_ids == (1, 2)


def test_pending_refresh_rejects_unbound_external_record() -> None:
    """Pending requests are strict and bind the source snapshot identity."""
    with pytest.raises(ValidationError):
        PendingRefreshRequest.model_validate(
            {
                "enqueue_relative_ids": [2],
                "membership_relative_ids": [1, 2],
                "next_high_watermark": 3,
                "unexpected": True,
            }
        )


def test_num_extra_shards_above_baseline_window_raises() -> None:
    """A refresh cannot replace more shards than the fixed baseline window."""
    with pytest.raises(ValueError, match="between 1 and baseline window size 2"):
        RollingWindow.from_spec(tiny_lance_spec(), num_extra_shards=3)


def test_extra_shard_identity_avoids_all_baseline_split_ids() -> None:
    """Extra IDs and sample offsets start after every frozen baseline split."""
    spec = tiny_lance_spec()
    window = RollingWindow.from_spec(spec, num_extra_shards=1)

    shard = window.extra_shard(spec, 2)

    assert shard.shard_id == 4
    assert shard.sample_offset == 8
    assert shard.filename == "shard-000004.lance"


def test_repeated_refresh_uses_published_high_watermark() -> None:
    """The next refresh starts exactly at the previously published watermark."""
    window = RollingWindow.from_spec(tiny_lance_spec(), num_extra_shards=1)

    second = window.advance(window.next_refresh()).next_refresh()

    assert second.enqueue_relative_ids == (3,)
    assert second.membership_relative_ids == (2, 3)
    assert second.next_high_watermark == 4


def test_initialize_branch_pins_explicit_baseline_and_publishes_ready_tag_last(
    tmp_path: Path,
) -> None:
    """Initialization records the pinned transaction before exposing readiness.

    :param tmp_path: Isolated local Lance and metadata root.
    """
    spec = tiny_lance_spec()
    train_uri = tmp_path / "train.lance"
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    baseline = [
        _worker_writes_fragment(train_uri, schema, _arange_arrays(offset))[0]
        for offset in (0, 1000)
    ]
    commit_lance_dataset(train_uri, schema, baseline)
    metadata_root = tmp_path / "rolling-metadata"
    (metadata_root / "versions/1").mkdir(parents=True)
    np.savez(metadata_root / "versions/1/stats.npz", mean=[0.0], std=[1.0])

    snapshot = initialize_rolling_branch(
        train_uri,
        spec=spec,
        branch="rolling",
        baseline_version=1,
        metadata_root=metadata_root,
        num_extra_shards=1,
    )

    persisted = RollingSnapshot.model_validate_json(
        (tmp_path / "rolling-metadata" / "versions" / "1" / "snapshot.json").read_text()
    )
    dataset = lance.dataset(str(train_uri))
    baseline_transaction = dataset.read_transaction(1)
    assert snapshot == persisted
    assert baseline_transaction is not None
    assert snapshot.baseline_transaction == baseline_transaction.uuid
    assert snapshot.window_size == 2
    assert snapshot.high_watermark == 2
    assert snapshot.membership_relative_ids == (0, 1)
    assert lance.dataset(snapshot.branch_uri).version == snapshot.version
    assert dataset.tags.get_version("rolling-ready") == 1
    assert dataset.checkout_version(("rolling", 1)).count_rows() == 4


def test_initialize_branch_retries_after_metadata_publication_failure(tmp_path: Path) -> None:
    """A retry resumes the existing pinned branch and publishes readiness.

    :param tmp_path: Isolated local Lance and metadata root.
    """
    spec = tiny_lance_spec()
    train_uri = tmp_path / "train.lance"
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    baseline = [
        _worker_writes_fragment(train_uri, schema, _arange_arrays(offset))[0]
        for offset in (0, 1000)
    ]
    commit_lance_dataset(train_uri, schema, baseline)
    metadata_root = tmp_path / "rolling-metadata"
    (metadata_root / "versions/1").mkdir(parents=True)
    np.savez(metadata_root / "versions/1/stats.npz", mean=[0.0], std=[1.0])

    def fail_publication(_snapshot: RollingSnapshot, _version_dir: Path) -> None:
        raise RuntimeError("metadata publication failed")

    with pytest.raises(RuntimeError, match="metadata publication failed"):
        initialize_rolling_branch(
            train_uri,
            spec=spec,
            branch="rolling",
            baseline_version=1,
            metadata_root=metadata_root,
            num_extra_shards=1,
            publish_metadata=fail_publication,
        )

    snapshot = initialize_rolling_branch(
        train_uri,
        spec=spec,
        branch="rolling",
        baseline_version=1,
        metadata_root=metadata_root,
        num_extra_shards=1,
    )

    assert snapshot.version == 1
    assert lance.dataset(str(train_uri)).tags.get_version("rolling-ready") == 1


def test_initialize_branch_rejects_wrong_baseline_fragment_count(tmp_path: Path) -> None:
    """The fixed rolling window requires one baseline fragment per train shard.

    :param tmp_path: Isolated local Lance root.
    """
    spec = tiny_lance_spec()
    train_uri = tmp_path / "train.lance"
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    rows = _arange_arrays(0)
    combined = {name: np.concatenate((values, values)) for name, values in rows.items()}
    baseline = _worker_writes_fragment(train_uri, schema, combined)[0]
    commit_lance_dataset(train_uri, schema, [baseline])
    metadata_root = tmp_path / "metadata"
    (metadata_root / "versions/1").mkdir(parents=True)
    np.savez(metadata_root / "versions/1/stats.npz", mean=[0.0], std=[1.0])

    with pytest.raises(ValueError, match="baseline train fragments"):
        initialize_rolling_branch(
            train_uri,
            spec=spec,
            branch="rolling",
            baseline_version=1,
            metadata_root=metadata_root,
            num_extra_shards=1,
        )


def test_initialize_branch_rejects_baseline_with_wrong_row_count(tmp_path: Path) -> None:
    """The pinned baseline must contain exactly the frozen train sample count.

    :param tmp_path: Isolated local Lance root.
    """
    spec = tiny_lance_spec()
    train_uri = tmp_path / "train.lance"
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    baseline = _worker_writes_fragment(train_uri, schema, _arange_arrays(0))[0]
    commit_lance_dataset(train_uri, schema, [baseline])
    metadata_root = tmp_path / "metadata"
    (metadata_root / "versions/1").mkdir(parents=True)
    np.savez(metadata_root / "versions/1/stats.npz", mean=[0.0], std=[1.0])

    with pytest.raises(ValueError, match="baseline train rows"):
        initialize_rolling_branch(
            train_uri,
            spec=spec.model_copy(
                update={"train_val_test_sizes": (3, 1, 1)},
            ),
            branch="rolling",
            baseline_version=1,
            metadata_root=metadata_root,
            num_extra_shards=1,
        )


def test_publish_overwrites_branch_and_advances_ready_tag_after_metadata(
    tmp_path: Path,
) -> None:
    """Publication commits K fragments and persists version metadata before readiness.

    :param tmp_path: Isolated local Lance and metadata root.
    """
    spec = tiny_lance_spec()
    train_uri = tmp_path / "train.lance"
    metadata_root = tmp_path / "rolling-metadata"
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    baseline_fragments = [
        _worker_writes_fragment(train_uri, schema, _arange_arrays(offset))[0]
        for offset in (0, 1000)
    ]
    commit_lance_dataset(train_uri, schema, baseline_fragments)
    (metadata_root / "versions/1").mkdir(parents=True)
    np.savez(metadata_root / "versions/1/stats.npz", mean=[0.0], std=[1.0])
    initial = initialize_rolling_branch(
        train_uri,
        spec=spec,
        branch="rolling",
        baseline_version=1,
        metadata_root=metadata_root,
        num_extra_shards=1,
    )
    replacement = _worker_writes_fragment(
        Path(initial.branch_uri), schema, _arange_arrays(0)
    )[0]

    published = publish_rolling_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=(
            lance.dataset(initial.branch_uri).get_fragments()[1].metadata,
            replacement,
        ),
        welford=(
            (2, np.array([0.0]), np.array([2.0])),
            (2, np.array([2.0]), np.array([2.0])),
        ),
        metadata_root=metadata_root,
    )

    dataset = lance.dataset(str(train_uri))
    assert published.high_watermark == 3
    assert published.membership_relative_ids == (1, 2)
    assert dataset.tags.get_version("rolling-ready") == published.version
    assert dataset.checkout_version(("rolling", published.version)).count_rows() == 4
    assert (metadata_root / "versions" / str(published.version) / "stats.npz").is_file()
    assert (metadata_root / "versions" / str(published.version) / "snapshot.json").is_file()


def test_publish_rejects_fragment_file_outside_branch_before_commit(tmp_path: Path) -> None:
    """Publication does not commit metadata for an unreadable staged fragment.

    :param tmp_path: Isolated local Lance and metadata root.
    """
    spec = tiny_lance_spec()
    train_uri = tmp_path / "train.lance"
    metadata_root = tmp_path / "rolling-metadata"
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    baseline_fragments = [
        _worker_writes_fragment(train_uri, schema, _arange_arrays(offset))[0]
        for offset in (0, 1000)
    ]
    commit_lance_dataset(train_uri, schema, baseline_fragments)
    (metadata_root / "versions/1").mkdir(parents=True)
    np.savez(metadata_root / "versions/1/stats.npz", mean=[0.0], std=[1.0])
    initial = initialize_rolling_branch(
        train_uri,
        spec=spec,
        branch="rolling",
        baseline_version=1,
        metadata_root=metadata_root,
        num_extra_shards=1,
    )
    foreign_fragment = _worker_writes_fragment(
        tmp_path / "foreign.lance", schema, _arange_arrays(2000)
    )[0]

    with pytest.raises(ValueError, match="missing or unreadable"):
        publish_rolling_branch(
            train_uri,
            spec=spec,
            current=initial,
            fragments=(
                lance.dataset(initial.branch_uri).get_fragments()[1].metadata,
                foreign_fragment,
            ),
            welford=(
                (2, np.array([0.0]), np.array([2.0])),
                (2, np.array([2.0]), np.array([2.0])),
            ),
            metadata_root=metadata_root,
        )

    branch = lance.dataset(str(train_uri)).checkout_version(("rolling", None))
    assert branch.version == initial.version
    assert lance.dataset(str(train_uri)).tags.get_version("rolling-ready") == initial.version


def test_exact_version_activation_keeps_open_prior_reader_usable(tmp_path: Path) -> None:
    """Atomic activation never mutates the snapshot held by an existing reader.

    :param tmp_path: Isolated source and active snapshot roots.
    """
    spec = tiny_lance_spec()
    train_uri = tmp_path / "train.lance"
    metadata_root = tmp_path / "rolling-metadata"
    local_root = tmp_path / "local"
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    baseline_fragments = [
        _worker_writes_fragment(train_uri, schema, _arange_arrays(offset))[0]
        for offset in (0, 1000)
    ]
    commit_lance_dataset(train_uri, schema, baseline_fragments)
    initial_version_dir = metadata_root / "versions" / "1"
    initial_version_dir.mkdir(parents=True)
    np.savez(initial_version_dir / "stats.npz", mean=np.array([0.0]), std=np.array([1.0]))
    initial = initialize_rolling_branch(
        train_uri,
        spec=spec,
        branch="rolling",
        baseline_version=1,
        metadata_root=metadata_root,
        num_extra_shards=1,
    )
    first_active = materialize_and_activate(
        train_uri,
        snapshot=initial,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )
    old_reader = lance.dataset(first_active.dataset_path)
    replacement = _worker_writes_fragment(
        Path(initial.branch_uri), schema, _arange_arrays(0)
    )[0]
    published = publish_rolling_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=(
            lance.dataset(initial.branch_uri).get_fragments()[1].metadata,
            replacement,
        ),
        welford=(
            (2, np.array([0.0]), np.array([2.0])),
            (2, np.array([2.0]), np.array([2.0])),
        ),
        metadata_root=metadata_root,
    )

    second_active = materialize_and_activate(
        train_uri,
        snapshot=published,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )

    active = ActiveRollingSnapshot.model_validate_json(
        (local_root / "active.json").read_text()
    )
    assert active == second_active
    assert old_reader.count_rows() == 4
    assert old_reader.to_table(columns=[PARAM_ARRAY_FIELD]).num_rows == 4
    assert first_active.dataset_path != second_active.dataset_path
    assert Path(first_active.dataset_path).is_dir()
