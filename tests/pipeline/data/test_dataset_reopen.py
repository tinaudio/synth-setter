"""Behaviour tests for reopening a finalized dataset so it can be extended."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.dataset_reopen import (
    ReopenPlan,
    main,
    plan_dataset_reopen,
    plan_reopen,
    reopen_dataset,
    validate_reopenable,
)
from synth_setter.pipeline.schemas.prefix import make_r2_prefix
from synth_setter.pipeline.schemas.spec import DatasetSpec

# Keep shard ranges readable in assertions.
_SAMPLES_PER_SHARD = 10
_SOURCE_SIZES = (40, 20, 20)


def _spec(
    valid_dataset_spec_kwargs: Mapping[str, object],
    overrides: Mapping[str, object] | None = None,
) -> DatasetSpec:
    """Build a split-seeded spec at the small shard scale these tests reason about.

    :param valid_dataset_spec_kwargs: Shared spec kwargs from the pipeline conftest.
    :param overrides: Top-level spec fields replacing the defaults.
    :returns: The constructed spec.
    """
    kwargs = copy.deepcopy(dict(valid_dataset_spec_kwargs))
    kwargs["train_val_test_sizes"] = list(_SOURCE_SIZES)
    kwargs["train_val_test_seeds"] = [42, 43, 44]
    render = cast(dict[str, object], kwargs["render"])
    render["samples_per_shard"] = _SAMPLES_PER_SHARD
    kwargs.update(overrides or {})
    return DatasetSpec.model_validate(kwargs)


@pytest.fixture()
def source_spec(valid_dataset_spec_kwargs: Mapping[str, object]) -> DatasetSpec:
    """Build a finalized 8-shard source spec: train 0..3, val 4..5, test 6..7.

    :param valid_dataset_spec_kwargs: Shared spec kwargs from the pipeline conftest.
    :returns: The source spec these tests extend.
    """
    return _spec(valid_dataset_spec_kwargs)


def test_validate_reopenable_legacy_base_seed_spec_raises(
    valid_dataset_spec_kwargs: Mapping[str, object],
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
    valid_dataset_spec_kwargs: Mapping[str, object],
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
    selected_attempts = []
    (root / "stats.npz").write_bytes(b"stats-bytes")
    (root / "config.yaml").write_text("stale: true\n")
    for shard in spec.shards:
        staged = root / "metadata" / "workers" / "shards" / f"shard-{shard.shard_id:06d}"
        staged.mkdir(parents=True, exist_ok=True)
        for suffix in (".fragment.json", ".shard-stats.npz", ".valid"):
            (staged / f"w0-a0{suffix}").write_text("")
        selected_attempts.append(
            {
                "shard_id": shard.shard_id,
                "attempt": "w0-a0",
                "valid_key": f"{spec.r2.prefix}metadata/workers/shards/"
                f"shard-{shard.shard_id:06d}/w0-a0.valid",
            }
        )
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": spec.run_id,
                "finalized_at": "2026-08-03T00:00:00+00:00",
                "selected_attempts": selected_attempts,
            }
        )
    )
    for split in ("train", "val", "test"):
        versions = root / f"{split}.lance" / "_versions"
        versions.mkdir(parents=True, exist_ok=True)
        (versions / "1.manifest").write_text("manifest")
        transactions = root / f"{split}.lance" / "_transactions"
        transactions.mkdir(parents=True, exist_ok=True)
        (transactions / "1.txn").write_text("transaction")
        data = root / f"{split}.lance" / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "frag.lance").write_text("fragment")
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


def test_reopen_dataset_copies_only_the_source_card_winner(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Preserve the finalized winner when a source shard has another valid attempt.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    source = _seed_finalized_root(fake_r2_remote, source_spec)
    staged = source / "metadata" / "workers" / "shards" / "shard-000000"
    for suffix in (".fragment.json", ".shard-stats.npz", ".valid"):
        (staged / f"w1-a1{suffix}").write_text("")

    plan = reopen_dataset(
        source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run"
    )

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    copied = dest / "metadata" / "workers" / "shards" / "shard-000000"
    assert sorted(path.name for path in copied.iterdir()) == [
        "w0-a0.fragment.json",
        "w0-a0.shard-stats.npz",
        "w0-a0.valid",
    ]


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


def test_reopen_dataset_drops_the_dataset_card(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Drop the card, since finalize rejects one carrying the source's run id.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    assert not (dest / "dataset.json").exists()


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

    plan = plan_dataset_reopen(
        source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run"
    )

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    assert not dest.exists()


def test_reopen_dataset_source_spec_for_another_root_raises(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Reject a copied source spec whose embedded root names another dataset.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    source = _seed_finalized_root(fake_r2_remote, source_spec)
    alias_root_uri = source_spec.r2.dataset_root_uri().replace(source_spec.run_id, "alias-run")
    alias = Path(str(source).replace(source_spec.run_id, "alias-run"))
    alias.mkdir(parents=True)
    shutil.copy2(source / "input_spec.json", alias / "input_spec.json")

    with pytest.raises(ValueError, match="requested source root"):
        reopen_dataset(alias_root_uri, (80, 20, 20), dest_run_id="grown-run")


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


def test_reopen_dataset_destination_state_without_identity_raises(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Reject copied destination state that has no verifiable reopen identity.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")
    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    foreign_attempt = dest / "metadata" / "workers" / "shards" / "shard-000004" / "foreign.valid"
    foreign_attempt.parent.mkdir(parents=True)
    foreign_attempt.write_text("")

    with pytest.raises(ValueError, match="reopen identity"):
        reopen_dataset(
            source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run"
        )

    assert foreign_attempt.exists()


def test_reopen_dataset_different_source_provenance_raises(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Reject a destination identity published for a different source spec.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)
    reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")
    payload = source_spec.model_dump(mode="json")
    payload["run_id"] = "other-source"
    r2_payload = cast(dict[str, object], payload["r2"])
    r2_payload["prefix"] = make_r2_prefix(
        source_spec.task_name, "other-source", source_spec.r2.prefix_root
    )
    other_source = DatasetSpec.model_validate(payload)
    _seed_finalized_root(fake_r2_remote, other_source)

    with pytest.raises(ValueError, match="reopen identity does not match"):
        reopen_dataset(
            other_source.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run"
        )


def test_reopen_dataset_different_destination_spec_raises(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Reject reuse of a destination identity for different grown sizes.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)
    reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    with pytest.raises(ValueError, match="reopen identity does not match"):
        reopen_dataset(
            source_spec.r2.dataset_root_uri(), (100, 20, 20), dest_run_id="grown-run"
        )


def test_reopen_dataset_exact_partial_resume_restores_copied_state(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Resume copied state only when its source and destination identity match exactly.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")
    _seed_finalized_root(fake_r2_remote, source_spec)
    reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")
    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    (dest / "input_spec.json").unlink()
    restored = dest / "metadata" / "workers" / "shards" / "shard-000003" / "w0-a0.valid"
    restored.unlink()

    reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    assert restored.exists()
    written = DatasetSpec.model_validate_json((dest / "input_spec.json").read_text())
    assert written == plan.dest_spec


def test_reopen_dataset_marker_deletion_failure_preserves_destination_state(
    source_spec: DatasetSpec, fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort before any mutation when the existing completion marker cannot be removed.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    :param monkeypatch: Pytest patcher used to inject a storage deletion failure.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)
    plan = reopen_dataset(
        source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run"
    )
    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    (dest / "dataset.complete").write_text("")
    (dest / "dataset.json").write_text("keep-card")
    before = {path.relative_to(dest): path.read_bytes() for path in dest.rglob("*") if path.is_file()}

    def fail_delete(_uri: str) -> None:
        raise subprocess.CalledProcessError(1, ["rclone", "deletefile"])

    monkeypatch.setattr("synth_setter.pipeline.r2_io.delete_object", fail_delete)
    with pytest.raises(subprocess.CalledProcessError):
        reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    after = {path.relative_to(dest): path.read_bytes() for path in dest.rglob("*") if path.is_file()}
    assert after == before


def test_reopen_dataset_required_cleanup_failure_aborts_before_spec_upload(
    source_spec: DatasetSpec, fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagate strict cleanup failure and leave the grown spec unpublished.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    :param monkeypatch: Pytest patcher used to inject a storage cleanup failure.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)
    plan = reopen_dataset(
        source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run"
    )
    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    (dest / "input_spec.json").unlink()
    blocked_prefix = dest / "val.lance"
    blocked_prefix.mkdir()
    blocked = blocked_prefix / "fragment.lance"
    blocked.write_text("must-remain")
    original_delete_prefix = r2_io.delete_prefix

    def fail_val_cleanup(uri: str) -> None:
        if uri == f'{plan.dest_spec.r2.split_lance_uri("val")}/':
            raise subprocess.CalledProcessError(1, ["rclone", "purge"])
        original_delete_prefix(uri)

    monkeypatch.setattr("synth_setter.pipeline.r2_io.delete_prefix", fail_val_cleanup)
    with pytest.raises(subprocess.CalledProcessError):
        reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    assert blocked.read_text() == "must-remain"
    assert not (dest / "input_spec.json").exists()


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


def test_reopen_dataset_copies_only_preserved_staging_and_train_fragment_data(
    source_spec: DatasetSpec, fake_r2_remote: Path
) -> None:
    """Copy no finalized manifests, held-out data, sidecars, claims, or stale config.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    :param fake_r2_remote: Fake R2 root that the real rclone binary resolves against.
    """
    _seed_finalized_root(fake_r2_remote, source_spec)

    plan = reopen_dataset(source_spec.r2.dataset_root_uri(), (80, 20, 20), dest_run_id="grown-run")

    dest = fake_r2_remote / source_spec.r2.bucket / plan.dest_spec.r2.prefix.rstrip("/")
    copied_files = {path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file()}
    assert copied_files == {
        "input_spec.json",
        "metadata/reopen.json",
        "metadata/workers/shards/shard-000000/w0-a0.fragment.json",
        "metadata/workers/shards/shard-000000/w0-a0.shard-stats.npz",
        "metadata/workers/shards/shard-000000/w0-a0.valid",
        "metadata/workers/shards/shard-000001/w0-a0.fragment.json",
        "metadata/workers/shards/shard-000001/w0-a0.shard-stats.npz",
        "metadata/workers/shards/shard-000001/w0-a0.valid",
        "metadata/workers/shards/shard-000002/w0-a0.fragment.json",
        "metadata/workers/shards/shard-000002/w0-a0.shard-stats.npz",
        "metadata/workers/shards/shard-000002/w0-a0.valid",
        "metadata/workers/shards/shard-000003/w0-a0.fragment.json",
        "metadata/workers/shards/shard-000003/w0-a0.shard-stats.npz",
        "metadata/workers/shards/shard-000003/w0-a0.valid",
        "train.lance/data/frag.lance",
    }


def test_reopen_plan_is_immutable(source_spec: DatasetSpec) -> None:
    """Reject mutation of a returned plan.

    :param source_spec: Finalized 8-shard source spec built by the module fixture.
    """
    plan = plan_reopen(source_spec, (80, 20, 20), dest_run_id="grown-run")

    with pytest.raises((AttributeError, TypeError)):
        plan.dest_root_uri = "r2://other/root/"  # type: ignore[misc]

    assert isinstance(plan, ReopenPlan)
