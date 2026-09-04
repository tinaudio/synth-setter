"""Strict growing snapshot adoption and checkpoint tests."""

from __future__ import annotations

import hashlib
import lightning
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lance
import numpy as np
import pytest

from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.data.growing_lance import ActiveGrowingSnapshot, GrowingSnapshot
from tests.helpers.lance_fixtures import write_mel_stats, write_seeded_lance_shard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline(tmp_path: Path) -> Path:
    root = tmp_path / "baseline"
    root.mkdir()
    write_seeded_lance_shard(root / "train.lance", num_rows=16, seed=1)
    write_seeded_lance_shard(root / "val.lance", num_rows=6, seed=2)
    write_seeded_lance_shard(root / "test.lance", num_rows=6, seed=3)
    write_mel_stats(root)
    return root


def _active_snapshot(tmp_path: Path, baseline: Path) -> tuple[Path, ActiveGrowingSnapshot]:
    root = tmp_path / "growing"
    dataset_path = root / "train.lance"
    shutil.copytree(baseline / "train.lance", dataset_path)
    local = lance.dataset(dataset_path)
    local_transaction = local.read_transaction(local.version)
    assert local_transaction is not None
    version_root = root / "versions/7"
    version_root.mkdir(parents=True)
    write_mel_stats(version_root, mean=4.0, std=2.0)
    np.savez(
        version_root / "welford.npz",
        count=np.int64(16),
        mean=np.zeros((1,), dtype=np.float32),
        m2=np.ones((1,), dtype=np.float32),
    )
    stats_sha = _sha(version_root / "stats.npz")
    welford_sha = _sha(version_root / "welford.npz")
    remote = GrowingSnapshot(
        branch="growing",
        branch_uri="s3://example/train.lance/tree/growing",
        version=7,
        baseline_version=1,
        baseline_transaction="baseline-transaction",
        transaction="remote-transaction",
        baseline_train_shards=1,
        max_train_shards=500,
        num_extra_shards=10,
        high_watermark=7,
        dataset_spec_fingerprint="spec-fingerprint",
        row_count=local.count_rows(),
        fragment_count=len(local.get_fragments()),
        schema_fingerprint="remote-schema",
        stats_sha256=stats_sha,
        welford_sha256=welford_sha,
    )
    (version_root / "snapshot.json").write_text(remote.model_dump_json())
    active = ActiveGrowingSnapshot(
        branch=remote.branch,
        remote_version=remote.version,
        remote_transaction=remote.transaction,
        local_version=local.version,
        local_transaction=local_transaction.uuid,
        dataset_path=str(dataset_path),
        version_stats_path=str(version_root),
        dataset_spec_fingerprint=remote.dataset_spec_fingerprint,
        row_count=remote.row_count,
        fragment_count=remote.fragment_count,
        schema_fingerprint=hashlib.sha256(local.schema.serialize().to_pybytes()).hexdigest(),
        stats_sha256=stats_sha,
        welford_sha256=welford_sha,
        high_watermark=remote.high_watermark,
    )
    active_path = root / "active.json"
    active_path.write_text(active.model_dump_json())
    return active_path, active


def _newer_identity(active_path: Path, active: ActiveGrowingSnapshot) -> ActiveGrowingSnapshot:
    old_root = Path(active.version_stats_path)
    version_root = old_root.parent / "8"
    shutil.copytree(old_root, version_root)
    remote = GrowingSnapshot.model_validate_json((version_root / "snapshot.json").read_bytes())
    remote = remote.model_copy(
        update={"version": 8, "transaction": "remote-transaction-8", "high_watermark": 8}
    )
    (version_root / "snapshot.json").write_text(remote.model_dump_json())
    newer = active.model_copy(
        update={
            "remote_version": 8,
            "remote_transaction": remote.transaction,
            "version_stats_path": str(version_root),
            "high_watermark": 8,
        }
    )
    active_path.write_text(newer.model_dump_json())
    return newer


def _module(baseline: Path, active_path: Path) -> LanceVSTDataModule:
    return LanceVSTDataModule(
        dataset_root=baseline,
        growing_active_record=active_path,
        batch_size=4,
        num_workers=0,
        val_num_workers=0,
        persistent_workers=False,
        param_spec_name=ParamSpecName("surge_xt"),
    )


def test_checkpoint_persists_full_strict_active_snapshot(tmp_path: Path) -> None:
    """Checkpoint state retains every remote and local identity field.

    :param tmp_path: Isolated baseline and growing roots.
    """
    baseline = _baseline(tmp_path)
    active_path, active = _active_snapshot(tmp_path, baseline)
    module = _module(baseline, active_path)
    module.setup("fit")

    state = module.state_dict()

    assert ActiveGrowingSnapshot.model_validate(state["growing_active_snapshot"]) == active
    assert state["growing_history"] == (7,)


def test_checkpoint_resume_rejects_corrupt_version_statistics(tmp_path: Path) -> None:
    """A changed version-bound artifact fails before loader construction.

    :param tmp_path: Isolated baseline and growing roots.
    """
    baseline = _baseline(tmp_path)
    active_path, _ = _active_snapshot(tmp_path, baseline)
    source = _module(baseline, active_path)
    source.setup("fit")
    state = source.state_dict()
    (active_path.parent / "versions/7/stats.npz").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="stats.npz"):
        _module(baseline, active_path).load_state_dict(state)


def test_checkpoint_resume_restores_recorded_snapshot_before_newer_active(
    tmp_path: Path,
) -> None:
    """Resume builds the checkpoint version before adopting a newer active record.

    :param tmp_path: Isolated baseline and growing roots.
    """
    baseline = _baseline(tmp_path)
    active_path, active = _active_snapshot(tmp_path, baseline)
    source = _module(baseline, active_path)
    source.setup("fit")
    state = source.state_dict()
    newer = _newer_identity(active_path, active)
    resumed = _module(baseline, active_path)

    resumed.load_state_dict(state)
    resumed.setup("fit")

    assert resumed.state_dict()["growing_history"] == (7,)
    resumed.train_dataloader()
    assert resumed.state_dict()["growing_history"] == (7,)
    resumed.train_dataloader()
    assert resumed.state_dict()["growing_history"] == (7, newer.remote_version)


def test_ddp_candidate_disagreement_retains_current_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All ranks participate and one missing identity rejects DDP adoption.

    :param tmp_path: Isolated baseline and growing roots.
    :param monkeypatch: Simulates two initialized distributed ranks.
    """
    baseline = _baseline(tmp_path)
    active_path, _ = _active_snapshot(tmp_path, baseline)
    module = _module(baseline, active_path)
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 2)
    monkeypatch.setattr("torch.distributed.broadcast_object_list", lambda values, src: None)

    def disagree(values: list[bool | None], ready: bool) -> None:
        values[:] = [ready, False]

    monkeypatch.setattr("torch.distributed.all_gather_object", disagree)

    assert module._read_active_train() is None


def test_test_stage_uses_frozen_baseline_statistics(tmp_path: Path) -> None:
    """Standalone test setup ignores growing cumulative train statistics.

    :param tmp_path: Isolated baseline and growing roots.
    """
    baseline = _baseline(tmp_path)
    active_path, _ = _active_snapshot(tmp_path, baseline)
    with_growth = _module(baseline, active_path)
    baseline_only = _module(baseline, baseline / "missing-active.json")
    with_growth.setup("test")
    baseline_only.setup("test")

    grown_batch = next(iter(with_growth.test_dataloader()))
    baseline_batch = next(iter(baseline_only.test_dataloader()))

    assert np.array_equal(grown_batch["mel"].numpy(), baseline_batch["mel"].numpy())


def test_growing_record_without_dataloader_reload_warns_snapshots_never_adopt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A trainer that never reloads loaders can never adopt a newer snapshot.

    :param tmp_path: Isolated baseline and growing roots.
    :param caplog: Captured datamodule log records.
    """
    baseline = _baseline(tmp_path)
    active_path, _ = _active_snapshot(tmp_path, baseline)
    module = _module(baseline, active_path)
    module.trainer = cast(
        "lightning.Trainer", SimpleNamespace(reload_dataloaders_every_n_epochs=0)
    )

    with caplog.at_level(logging.WARNING, logger="synth_setter.data.lance_datamodule"):
        module.setup("fit")

    assert any("reload_dataloaders_every_n_epochs" in record.message for record in caplog.records)
