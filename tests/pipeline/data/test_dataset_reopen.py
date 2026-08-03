"""Behaviour tests for reopening a finalized dataset so it can be extended."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from synth_setter.pipeline.data.dataset_reopen import (
    main,
    ReopenPlan,
    plan_reopen,
    reopen_dataset,
    validate_reopenable,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec

# Small enough that shard ranges are readable inline: train 4, val 2, test 2.
_SAMPLES_PER_SHARD = 10
_SOURCE_SIZES = (40, 20, 20)


def _spec(
    valid_dataset_spec_kwargs: dict[str, Any], overrides: dict[str, Any] | None = None
) -> DatasetSpec:
    """Build a split-seeded spec at the small shard scale these tests reason about.

    :param valid_dataset_spec_kwargs: Shared spec kwargs from the pipeline conftest.
    :param overrides: Top-level spec fields replacing the defaults.
    :returns: The constructed spec.
    """
    kwargs = copy.deepcopy(valid_dataset_spec_kwargs)
    kwargs["train_val_test_sizes"] = list(_SOURCE_SIZES)
    kwargs["train_val_test_seeds"] = [42, 43, 44]
    kwargs["render"]["samples_per_shard"] = _SAMPLES_PER_SHARD
    kwargs.update(overrides or {})
    return DatasetSpec(**kwargs)


@pytest.fixture()
def source_spec(valid_dataset_spec_kwargs: dict[str, Any]) -> DatasetSpec:
    """Build a finalized 8-shard source spec: train 0..3, val 4..5, test 6..7.

    :param valid_dataset_spec_kwargs: Shared spec kwargs from the pipeline conftest.
    :returns: The source spec these tests extend.
    """
    return _spec(valid_dataset_spec_kwargs)


def test_validate_reopenable_legacy_base_seed_spec_raises(
    valid_dataset_spec_kwargs: dict[str, Any],
) -> None:
    """Reject a spec whose rows are seeded from base_seed plus the shard id.

    :param valid_dataset_spec_kwargs: Shared spec kwargs from the pipeline conftest.
    """
    legacy = _spec(valid_dataset_spec_kwargs, {"train_val_test_seeds": None})

    with pytest.raises(ValueError, match="train_val_test_seeds"):
        validate_reopenable(legacy, (80, 20, 20))


def test_validate_reopenable_shrinking_train_raises(source_spec: DatasetSpec) -> None:
    """Reject a target train size below the source's.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    with pytest.raises(ValueError, match="shrink"):
        validate_reopenable(source_spec, (20, 20, 20))


def test_validate_reopenable_changed_val_size_raises(source_spec: DatasetSpec) -> None:
    """Reject a reopen that would resize the held-out splits.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    with pytest.raises(ValueError, match="val/test"):
        validate_reopenable(source_spec, (80, 30, 20))


def test_validate_reopenable_train_size_not_multiple_of_shard_raises(
    source_spec: DatasetSpec,
) -> None:
    """Reject a train size that does not divide into whole shards.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    with pytest.raises(ValueError, match="multiple"):
        validate_reopenable(source_spec, (85, 20, 20))


def test_validate_reopenable_unchanged_sizes_is_allowed(source_spec: DatasetSpec) -> None:
    """Accept a reopen that grows nothing.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    validate_reopenable(source_spec, _SOURCE_SIZES)


def test_plan_reopen_partitions_shard_ids_across_the_old_train_boundary(
    source_spec: DatasetSpec,
) -> None:
    """Split the shard space at the boundary the old train size defines.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")

    assert plan.preserved_shard_ids == range(0, 4)
    assert plan.discarded_shard_ids == range(4, 8)
    assert plan.pending_shard_ids == range(4, 12)


def test_plan_reopen_dest_spec_recomputes_shard_count(source_spec: DatasetSpec) -> None:
    """Recompute the shard layout; ``model_copy`` would carry the cached count through.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")

    assert plan.dest_spec.num_shards == 12
    assert plan.dest_spec.split_shard_ranges == {
        "train": (0, 8),
        "val": (8, 10),
        "test": (10, 12),
    }


def test_plan_reopen_preserves_seed_position_of_every_preserved_shard(
    source_spec: DatasetSpec,
) -> None:
    """Keep preserved shards at their seed positions, or the skip-probe is unsound.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")

    source_positions = [
        (source_spec.render_for_shard(s).base_seed, source_spec.render_for_shard(s).sample_offset)
        for s in source_spec.shards[:4]
    ]
    dest_positions = [
        (
            plan.dest_spec.render_for_shard(s).base_seed,
            plan.dest_spec.render_for_shard(s).sample_offset,
        )
        for s in plan.dest_spec.shards[:4]
    ]

    assert source_positions == [(42, 0), (42, 10), (42, 20), (42, 30)]
    assert dest_positions == source_positions


def test_plan_reopen_first_pending_shard_continues_the_train_stream(
    source_spec: DatasetSpec,
) -> None:
    """Start the first pending shard where the source's last train shard ended.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")

    first_pending = plan.dest_spec.render_for_shard(plan.dest_spec.shards[4])

    assert (first_pending.base_seed, first_pending.sample_offset) == (42, 40)


def test_plan_reopen_dest_prefix_differs_from_source_prefix(source_spec: DatasetSpec) -> None:
    """Give the destination its own prefix so workers never stage into the source.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")

    assert plan.dest_spec.r2.prefix != source_spec.r2.prefix
    assert plan.dest_spec.run_id == "grown-run"


def test_plan_reopen_zero_size_val_and_test_discards_nothing(
    valid_dataset_spec_kwargs: dict[str, Any],
) -> None:
    """Discard no staging when the source has no held-out splits to renumber.

    :param valid_dataset_spec_kwargs: Shared spec kwargs from the pipeline conftest.
    """
    train_only = _spec(valid_dataset_spec_kwargs, {"train_val_test_sizes": [40, 0, 0]})

    plan = plan_reopen(train_only, (80, 0, 0), dest_run_id="grown-run")

    assert plan.discarded_shard_ids == range(4, 4)
    assert plan.pending_shard_ids == range(4, 8)


def _seed_finalized_root(remote: Path, spec: DatasetSpec) -> Path:
    """Materialize a minimal finalized source root under the fake R2 remote.

    :param remote: Fake R2 root the objects land under.
    :param spec: Spec whose prefix and shard ids shape the layout.
    :returns: The seeded source root path.
    """
    root = remote / spec.r2.bucket / spec.r2.prefix.rstrip("/")
    (root / "metadata" / "workers").mkdir(parents=True, exist_ok=True)
    (root / "input_spec.json").write_text(spec.model_dump_json())
    (root / "dataset.complete").write_text("")
    (root / "dataset.json").write_text(json.dumps({"schema_version": 1}))
    (root / "stats.npz").write_bytes(b"stats-bytes")
    for shard in spec.shards:
        staged = root / "metadata" / "workers" / "shards" / f"shard-{shard.shard_id:06d}"
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "w0-a0.valid").write_text("")
    claims = root / "metadata" / "shard-claims.lance"
    claims.mkdir(parents=True, exist_ok=True)
    (claims / "manifest").write_text("claims")
    return root


def test_reopen_dataset_writes_a_grown_spec_at_the_destination(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Write the grown spec, with its recomputed shard count, to the destination.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    written = DatasetSpec.model_validate_json((dest / "input_spec.json").read_text())
    assert written.train_val_test_sizes == (80, 20, 20)
    assert written.num_shards == 12


def test_reopen_dataset_clears_the_complete_marker_at_the_destination(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Leave the destination without a completion marker so finalize runs.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    assert not (dest / "dataset.complete").exists()


def test_reopen_dataset_keeps_preserved_shard_staging(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Keep staged attempts for shards below the old train boundary.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    staged = dest / "metadata" / "workers" / "shards"
    assert (staged / "shard-000003" / "w0-a0.valid").exists()


def test_reopen_dataset_discards_staging_at_or_above_the_old_train_boundary(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Discard renumbered staging that would make the skip-probe skip train work.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    staged = dest / "metadata" / "workers" / "shards"
    assert not (staged / "shard-000004").exists()
    assert not (staged / "shard-000007").exists()


def test_reopen_dataset_drops_the_claims_table(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Drop the claims table so it repopulates over the grown shard range.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    assert not (dest / "metadata" / "shard-claims.lance").exists()


def test_reopen_dataset_keeps_the_dataset_card(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Keep the card that pins which attempt won for each preserved shard.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    assert json.loads((dest / "dataset.json").read_text()) == {"schema_version": 1}


def test_reopen_dataset_leaves_the_source_root_untouched(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Leave every object in the extended source root exactly as it was.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    source_root = _seed_finalized_root(fake_r2_remote, source_spec)

    reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    assert (source_root / "dataset.complete").exists()
    assert (source_root / "metadata" / "workers" / "shards" / "shard-000007").exists()
    assert (source_root / "metadata" / "shard-claims.lance").exists()
    kept = DatasetSpec.model_validate_json((source_root / "input_spec.json").read_text())
    assert kept.train_val_test_sizes == _SOURCE_SIZES


def test_reopen_dataset_dry_run_writes_no_destination(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Write nothing at all when planning only.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(
        source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run", dry_run=True
    )

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    assert not dest.exists()


def test_reopen_dataset_incomplete_source_raises(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Refuse to extend a dataset that was never finalized.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    root = _seed_finalized_root(fake_r2_remote, source_spec)
    (root / "dataset.complete").unlink()

    with pytest.raises(FileNotFoundError, match="dataset.complete"):
        reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")


def test_main_without_apply_writes_nothing(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Plan only by default, because the copy is the size of the source.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    main(["--source", source_spec.r2.dataset_root_uri(), "--train-size", "80"])

    bucket_root = fake_r2_remote / source_spec.r2.bucket / source_spec.r2.prefix_root
    assert [p.name for p in bucket_root.glob("*/*")] == [source_spec.run_id]


def test_main_with_apply_reopens_at_the_requested_train_size(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Reopen at the requested train size when writing is opted into.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    main(
        [
            "--source",
            source_spec.r2.dataset_root_uri(),
            "--train-size",
            "80",
            "--dest-run-id",
            "grown-run",
            "--apply",
        ]
    )

    dest = (
        fake_r2_remote / source_spec.r2.bucket / source_spec.r2.prefix_root /
        source_spec.task_name / "grown-run"
    )
    written = DatasetSpec.model_validate_json((dest / "input_spec.json").read_text())
    assert written.train_val_test_sizes == (80, 20, 20)
    assert not (dest / "dataset.complete").exists()


def test_main_carries_source_val_and_test_sizes_through(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Carry the source's held-out split sizes through; only train is settable.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    main(
        [
            "--source",
            source_spec.r2.dataset_root_uri(),
            "--train-size",
            "80",
            "--dest-run-id",
            "grown-run",
            "--apply",
        ]
    )

    dest = (
        fake_r2_remote / source_spec.r2.bucket / source_spec.r2.prefix_root /
        source_spec.task_name / "grown-run"
    )
    written = DatasetSpec.model_validate_json((dest / "input_spec.json").read_text())
    assert written.train_val_test_sizes[1:] == source_spec.train_val_test_sizes[1:]


def test_reopen_plan_is_immutable(source_spec: DatasetSpec) -> None:
    """Reject mutation of a returned plan.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")

    with pytest.raises((AttributeError, TypeError)):
        plan.dest_root_uri = "r2://other/root/"  # type: ignore[misc]

    assert isinstance(plan, ReopenPlan)
