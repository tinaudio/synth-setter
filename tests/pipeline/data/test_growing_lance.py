"""Growing Lance range, publication, and local hydration contracts."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.data.vst.shapes import DATASET_FIELD_NAMES, MEL_SPEC_FIELD, dataset_field_shapes
from synth_setter.pipeline.data import growing_lance
from synth_setter.pipeline.data.growing_lance import (
    ActiveGrowingSnapshot,
    GrowingPlan,
    GrowingSnapshot,
    initialize_growing_branch,
    materialize_and_activate,
    pending_refresh_request,
    publish_growing_branch,
)
from synth_setter.pipeline.data.lance_shard import commit_lance_dataset, lance_schema
from synth_setter.pipeline.data.stats import save_welford
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.pipeline.data.test_lance_fragment_finalize_poc import (
    _FIELD_SHAPES,
    _METADATA,
    _arange_arrays,
    _worker_writes_fragment,
)
from tests.pipeline.data.test_lance_staging import tiny_lance_spec


def _baseline_dataset(tmp_path: Path) -> tuple[DatasetSpec, Path, Path]:
    spec = tiny_lance_spec()
    train_uri = tmp_path / "train.lance"
    schema = lance_schema(_FIELD_SHAPES, _METADATA)
    fragments = [
        _worker_writes_fragment(train_uri, schema, _arange_arrays(offset))[0]
        for offset in (0, 1000)
    ]
    commit_lance_dataset(train_uri, schema, fragments)
    metadata_root = tmp_path / "metadata"
    version_root = metadata_root / "versions/1"
    version_root.mkdir(parents=True)
    shape = dataset_field_shapes(spec.render, spec.num_params)[MEL_SPEC_FIELD][1:]
    state = (
        4,
        np.zeros(shape, dtype=np.float32),
        np.ones(shape, dtype=np.float32),
    )
    save_welford(version_root / "welford.npz", state, expected_shape=state[1].shape)
    np.savez(version_root / "stats.npz", mean=state[1], std=np.ones_like(state[1]))
    return spec, train_uri, metadata_root


def _baseline(
    tmp_path: Path, *, max_train_shards: int = 5
) -> tuple[DatasetSpec, Path, Path, GrowingSnapshot]:
    spec, train_uri, metadata_root = _baseline_dataset(tmp_path)
    snapshot = initialize_growing_branch(
        train_uri,
        spec=spec,
        branch="growing",
        baseline_version=1,
        metadata_root=metadata_root,
        max_train_shards=max_train_shards,
        num_extra_shards=2,
    )
    return spec, train_uri, metadata_root, snapshot


def test_initialize_recovers_branch_created_without_contract_metadata(
    tmp_path: Path,
) -> None:
    """A rerun completes a branch whose metadata write crashed after creation.

    :param tmp_path: Isolated Lance and metadata roots.
    """
    spec, train_uri, metadata_root = _baseline_dataset(tmp_path)
    lance.dataset(str(train_uri)).create_branch("growing", 1)

    snapshot = initialize_growing_branch(
        train_uri,
        spec=spec,
        branch="growing",
        baseline_version=1,
        metadata_root=metadata_root,
        max_train_shards=5,
        num_extra_shards=2,
    )

    metadata = lance.dataset(str(train_uri)).branches.list()["growing"]["metadata"]
    assert metadata["synth_setter.growing_max_train_shards"] == "5"
    assert snapshot.high_watermark == 2


def _append_fragments(
    snapshot: GrowingSnapshot, count: int
) -> tuple[lance.fragment.FragmentMetadata, ...]:
    schema = lance.dataset(snapshot.branch_uri).schema
    return tuple(
        _worker_writes_fragment(
            Path(snapshot.branch_uri), schema, _arange_arrays(2000 + index * 1000)
        )[0]
        for index in range(count)
    )


def _states(count: int) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
    spec = tiny_lance_spec()
    shape = dataset_field_shapes(spec.render, spec.num_params)[MEL_SPEC_FIELD][1:]
    return tuple(
        (2, np.full(shape, 2.0 + index, dtype=np.float32), np.ones(shape, dtype=np.float32))
        for index in range(count)
    )


def test_growing_plan_baseline_five_to_five_hundred_is_bounded() -> None:
    """The maximum counts baseline train shards and the request is relative to readiness."""
    plan = GrowingPlan(
        baseline_train_shards=5,
        max_train_shards=500,
        num_extra_shards=100,
        high_watermark=5,
    )

    request = plan.next_refresh()

    assert request is not None
    assert request.enqueue_shard_ids == tuple(range(5, 105))
    assert request.next_high_watermark == 105


def test_growing_plan_final_request_is_short_then_capacity_is_noop() -> None:
    """A request truncates at max and the next request safely refuses work."""
    plan = GrowingPlan(5, 500, 100, 405)

    request = plan.next_refresh()

    assert request is not None
    assert request.enqueue_shard_ids == tuple(range(405, 500))
    assert request.next_high_watermark == 500
    assert plan.advance(request).next_refresh() is None


def test_growing_extra_shard_uses_direct_train_id_and_frozen_sample_offset() -> None:
    """Branch-local metadata permits direct train positions without baseline collisions."""
    spec = tiny_lance_spec()
    plan = GrowingPlan(2, 5, 2, 2)

    shard = plan.extra_shard(spec, 2)

    assert shard.shard_id == 2
    assert shard.filename == "shard-000002.lance"
    assert shard.sample_offset == 8


def test_initialize_persists_immutable_native_contract_and_validates_reinit(
    tmp_path: Path,
) -> None:
    """Native branch metadata binds every growth limit and baseline identity.

    :param tmp_path: Isolated Lance and metadata roots.
    """
    spec, train_uri, _, snapshot = _baseline(tmp_path)
    metadata = lance.dataset(str(train_uri)).branches.list()["growing"]["metadata"]

    assert snapshot.baseline_train_shards == 2
    assert snapshot.high_watermark == 2
    assert snapshot.max_train_shards == 5
    assert metadata["synth_setter.growing_baseline_train_shards"] == "2"
    assert metadata["synth_setter.growing_max_train_shards"] == "5"
    with pytest.raises(ValueError, match="growing contract"):
        initialize_growing_branch(
            train_uri,
            spec=spec,
            branch="growing",
            baseline_version=1,
            metadata_root=tmp_path / "metadata",
            max_train_shards=4,
            num_extra_shards=2,
        )


def test_publish_appends_exact_prefix_and_preserves_old_version(tmp_path: Path) -> None:
    """Publication uses append semantics and retains the readable baseline version.

    :param tmp_path: Isolated Lance and metadata roots.
    """
    spec, train_uri, metadata_root, initial = _baseline(tmp_path)
    old = lance.dataset(str(train_uri)).checkout_version(("growing", initial.version))
    old_files = tuple(fragment.metadata.files[0].path for fragment in old.get_fragments())

    published = publish_growing_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=_append_fragments(initial, 2),
        welford=_states(2),
        metadata_root=metadata_root,
    )

    latest = lance.dataset(str(train_uri)).checkout_version(("growing", published.version))
    latest_files = tuple(fragment.metadata.files[0].path for fragment in latest.get_fragments())
    assert latest_files[:2] == old_files
    assert len(latest_files) == 4
    assert latest.count_rows() == old.count_rows() + 4
    assert old.count_rows() == 4
    assert lance.dataset(str(train_uri)).tags.get_version("growing-ready") == published.version


def test_publish_final_short_request_then_returns_capacity_noop(tmp_path: Path) -> None:
    """The final append may be short and further publication creates no version.

    :param tmp_path: Isolated Lance and metadata roots.
    """
    spec, train_uri, metadata_root, initial = _baseline(tmp_path, max_train_shards=3)
    published = publish_growing_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=_append_fragments(initial, 1),
        welford=_states(1),
        metadata_root=metadata_root,
    )
    before = lance.dataset(str(train_uri)).checkout_version(("growing", None)).version

    noop = publish_growing_branch(
        train_uri,
        spec=spec,
        current=published,
        fragments=(),
        welford=(),
        metadata_root=metadata_root,
    )

    assert noop == published
    assert pending_refresh_request(published) is None
    assert lance.dataset(str(train_uri)).checkout_version(("growing", None)).version == before


def test_materialization_appends_to_one_local_dataset_and_keeps_old_version(
    tmp_path: Path,
) -> None:
    """Hydration scans only growth and records exact remote-to-local identities.

    :param tmp_path: Isolated source and local roots.
    """
    spec, train_uri, metadata_root, initial = _baseline(tmp_path)
    local_root = tmp_path / "local"
    first = materialize_and_activate(
        train_uri,
        snapshot=initial,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )
    published = publish_growing_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=_append_fragments(initial, 2),
        welford=_states(2),
        metadata_root=metadata_root,
    )

    second = materialize_and_activate(
        train_uri,
        snapshot=published,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )

    assert first.dataset_path == second.dataset_path == str(local_root / "train.lance")
    local = lance.dataset(second.dataset_path)
    assert local.count_rows() == 8
    assert local.checkout_version(first.local_version).count_rows() == 4
    assert second.remote_version == published.version
    assert second.local_version > first.local_version
    assert Path(second.version_stats_path) == local_root / "versions" / str(published.version)
    assert ActiveGrowingSnapshot.model_validate_json(
        (local_root / "active.json").read_bytes()
    ) == second


def test_publish_reconciles_crash_after_remote_append(tmp_path: Path) -> None:
    """A rerun recognizes its deterministic append after metadata publication crashes.

    :param tmp_path: Isolated Lance and metadata roots.
    """
    spec, train_uri, metadata_root, initial = _baseline(tmp_path)
    fragments = _append_fragments(initial, 2)
    states = _states(2)

    def fail_after_append(snapshot: GrowingSnapshot, version_dir: Path) -> None:
        del snapshot, version_dir
        raise RuntimeError("simulated publication crash")

    with pytest.raises(RuntimeError, match="simulated publication crash"):
        publish_growing_branch(
            train_uri,
            spec=spec,
            current=initial,
            fragments=fragments,
            welford=states,
            metadata_root=metadata_root,
            publish_metadata=fail_after_append,
        )

    recovered = publish_growing_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=fragments,
        welford=states,
        metadata_root=metadata_root,
    )

    assert recovered.row_count == 8
    assert lance.dataset(str(train_uri)).tags.get_version("growing-ready") == recovered.version


def test_publish_rejects_unknown_branch_advancement(tmp_path: Path) -> None:
    """A branch version without the pending identity fails closed.

    :param tmp_path: Isolated Lance and metadata roots.
    """
    spec, train_uri, metadata_root, initial = _baseline(tmp_path)
    fragments = _append_fragments(initial, 2)
    source = lance.dataset(initial.branch_uri).checkout_version(initial.version)
    transaction = lance.Transaction(
        read_version=source.version,
        operation=lance.LanceOperation.Append(list(fragments)),
        transaction_properties={"unexpected": "writer"},
    )
    lance.LanceDataset.commit(source, transaction)

    with pytest.raises(ValueError, match="advanced unexpectedly"):
        publish_growing_branch(
            train_uri,
            spec=spec,
            current=initial,
            fragments=fragments,
            welford=_states(2),
            metadata_root=metadata_root,
        )


def test_materialization_reconciles_crash_after_initial_local_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun activates an initial local commit without rewriting it.

    :param tmp_path: Isolated Lance and metadata roots.
    :param monkeypatch: Replaces activation only for the simulated crash.
    """
    _, train_uri, metadata_root, initial = _baseline(tmp_path)
    local_root = tmp_path / "local"
    real_activate = growing_lance._activate

    def fail_activation(*args: object, **kwargs: object) -> ActiveGrowingSnapshot:
        del args, kwargs
        raise RuntimeError("simulated activation crash")

    monkeypatch.setattr(growing_lance, "_activate", fail_activation)
    with pytest.raises(RuntimeError, match="simulated activation crash"):
        materialize_and_activate(
            train_uri,
            snapshot=initial,
            metadata_root=metadata_root,
            local_root=local_root,
            columns=DATASET_FIELD_NAMES,
        )
    committed_version = lance.dataset(str(local_root / "train.lance")).version
    monkeypatch.setattr(growing_lance, "_activate", real_activate)

    recovered = materialize_and_activate(
        train_uri,
        snapshot=initial,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )

    assert recovered.local_version == committed_version
    assert lance.dataset(recovered.dataset_path).count_rows() == initial.row_count


def test_materialization_without_active_rejects_different_local_identity(
    tmp_path: Path,
) -> None:
    """An unrecorded local dataset from another snapshot fails closed.

    :param tmp_path: Isolated Lance and metadata roots.
    """
    spec, train_uri, metadata_root, initial = _baseline(tmp_path)
    local_root = tmp_path / "local"
    first = materialize_and_activate(
        train_uri,
        snapshot=initial,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )
    published = publish_growing_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=_append_fragments(initial, 2),
        welford=_states(2),
        metadata_root=metadata_root,
    )
    (local_root / "active.json").unlink()

    with pytest.raises(ValueError, match="different remote snapshot"):
        materialize_and_activate(
            train_uri,
            snapshot=published,
            metadata_root=metadata_root,
            local_root=local_root,
            columns=DATASET_FIELD_NAMES,
        )

    assert lance.dataset(first.dataset_path).version == first.local_version


def test_materialization_reconciles_crash_after_local_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun activates a local append committed before an activation crash.

    :param tmp_path: Isolated Lance and metadata roots.
    :param monkeypatch: Replaces activation only for the simulated crash.
    """
    spec, train_uri, metadata_root, initial = _baseline(tmp_path)
    local_root = tmp_path / "local"
    materialize_and_activate(
        train_uri,
        snapshot=initial,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )
    published = publish_growing_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=_append_fragments(initial, 2),
        welford=_states(2),
        metadata_root=metadata_root,
    )
    real_activate = growing_lance._activate

    def fail_activation(*args: object, **kwargs: object) -> ActiveGrowingSnapshot:
        del args, kwargs
        raise RuntimeError("simulated activation crash")

    monkeypatch.setattr(growing_lance, "_activate", fail_activation)
    with pytest.raises(RuntimeError, match="simulated activation crash"):
        materialize_and_activate(
            train_uri,
            snapshot=published,
            metadata_root=metadata_root,
            local_root=local_root,
            columns=DATASET_FIELD_NAMES,
        )
    monkeypatch.setattr(growing_lance, "_activate", real_activate)

    recovered = materialize_and_activate(
        train_uri,
        snapshot=published,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )

    assert recovered.remote_version == published.version
    assert lance.dataset(recovered.dataset_path).count_rows() == 8


def test_stale_materializer_cannot_regress_active_record(tmp_path: Path) -> None:
    """Activating an older remote version after a newer one is a safe no-op.

    :param tmp_path: Isolated source and local roots.
    """
    spec, train_uri, metadata_root, initial = _baseline(tmp_path)
    published = publish_growing_branch(
        train_uri,
        spec=spec,
        current=initial,
        fragments=_append_fragments(initial, 2),
        welford=_states(2),
        metadata_root=metadata_root,
    )
    local_root = tmp_path / "local"
    newest = materialize_and_activate(
        train_uri,
        snapshot=published,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )

    stale = materialize_and_activate(
        train_uri,
        snapshot=initial,
        metadata_root=metadata_root,
        local_root=local_root,
        columns=DATASET_FIELD_NAMES,
    )

    assert stale == newest
    assert ActiveGrowingSnapshot.model_validate_json(
        (local_root / "active.json").read_bytes()
    ) == newest
