"""Unit + offline-e2e tests for the train-side model W&B artifact wiring.

``build_model_artifact`` is asserted on a real ``wandb.Artifact`` (no run, no
network): name, type, the R2 reference, and metadata are all observable on the
returned object, so these tests exercise the real construction rather than a
mock of it. ``_upload_best_checkpoint`` drives the real ``rclone`` binary
against a local-backed ``r2:`` remote (the ``fake_r2_remote`` pattern) so the
checkpoint upload is exercised end-to-end, not asserted on a mock.
``_log_model_artifact`` is driven against a ``WandbLogger`` subclass stub to pin
the WandbLogger-only / best-effort contract.

``test_train_logs_model_artifact_to_offline_wandb_run`` closes the gap those
unit tests leave open: it drives the real ``train(cfg)`` entrypoint with a real
``WandbLogger(offline=True)`` and decodes the offline ``run-*.wandb`` binary, so
dropping or mis-gating the ``train()``-end ``_log_model_artifact`` call (which
every cfg-level train test no-ops past with ``logger=None``) fails here.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest
import torch
import wandb
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import (
    _checkpoint_metadata,
    _checkpoint_path_written_by_run,
    _derive_checkpoint_uri,
    _log_model_artifact,
    _upload_best_checkpoint,
    build_model_artifact,
    train,
)
from synth_setter.pipeline import r2_io
from tests.helpers.wandb_offline import read_run_binary

_CKPT_URI = "r2://models/model-flow-simple/best.ckpt"
_CKPT_S3_REF = "s3://models/model-flow-simple/best.ckpt"
_LAUNCH_UUID = "7ac31b3ff42c4f13a21997adb4a74e86"
_SECOND_LAUNCH_UUID = "f52af7e63eaa41048599080186b92b5d"
_TRAINING_RUN_ID = "flow-simple-20260908T170724945Z"

# `cfg_train` composes no Hydra experiment, so `resolve_run_config_id` falls back
# to `task_name` ("train") — the config_id the e2e artifact name is built from.
_E2E_ARTIFACT_NAME = "model-train"


def _cfg(
    task_name: str = "flow-simple",
    upload_checkpoints_uri: str | None = None,
    bucket: str = "intermediate-data",
) -> DictConfig:
    """Build a minimal train cfg carrying ``task_name``, the ``r2`` group, and the upload override.

    No experiment is composed, so ``resolve_run_config_id`` falls back to
    ``task_name`` — the config_id the artifact name and derived URI are built from.

    :param task_name: Drives the resolved config_id and thus the artifact name.
    :param upload_checkpoints_uri: ``r2://`` upload-target override, or ``None``
        to auto-derive from ``r2.bucket`` + config_id.
    :param bucket: The ``r2.bucket`` the derived checkpoint URI is rooted at.
    :returns: A DictConfig with ``task_name``, ``r2.bucket``, and
        ``training.upload_checkpoints_uri``.
    """
    return cast(
        DictConfig,
        OmegaConf.create(
            {
                "task_name": task_name,
                "r2": {"bucket": bucket},
                "training": {"upload_checkpoints_uri": upload_checkpoints_uri},
            }
        ),
    )


class _RecordingWandbLogger(WandbLogger):
    """``WandbLogger`` subclass capturing the logged artifact without ``wandb.init``.

    Bypasses the base ``__init__`` (which calls ``wandb.init``) and exposes
    ``self`` as ``experiment`` so ``logger.experiment.log_artifact`` records
    into ``self.logged``.
    """

    def __init__(self) -> None:
        self.logged: list[Any] = []

    @property
    def experiment(self) -> Any:  # type: ignore[override]
        """Return self so ``experiment.log_artifact`` records the artifact."""
        return self

    def log_artifact(self, artifact: Any) -> None:  # type: ignore[override]
        """Record the artifact W&B would have logged.

        :param artifact: The ``wandb.Artifact`` ``_log_model_artifact`` passes.
        """
        self.logged.append(artifact)


def test_derive_checkpoint_uri_default_uses_config_run_and_launch_ids() -> None:
    """A null override derives a launch-scoped checkpoint URI."""
    uri = _derive_checkpoint_uri(_cfg(task_name="flow-simple"), _TRAINING_RUN_ID, _LAUNCH_UUID)
    assert uri == (
        "r2://intermediate-data/checkpoints/flow-simple/"
        "flow-simple-20260908T170724945Z/7ac31b3ff42c4f13a21997adb4a74e86/model.ckpt"
    )


def test_derive_checkpoint_uri_override_is_used_verbatim() -> None:
    """A set ``upload_checkpoints_uri`` overrides the derived path verbatim."""
    uri = _derive_checkpoint_uri(
        _cfg(upload_checkpoints_uri=_CKPT_URI), _TRAINING_RUN_ID, _LAUNCH_UUID
    )
    assert uri == _CKPT_URI


def test_build_model_artifact_name_is_model_prefixed_config_id() -> None:
    """The artifact name is ``model-{config_id}`` per storage-provenance-spec §4."""
    artifact = build_model_artifact(_cfg(task_name="flow-simple"))
    assert artifact.name == "model-flow-simple"


def test_build_model_artifact_type_is_model() -> None:
    """The artifact type is ``model`` per storage-provenance-spec §4."""
    artifact = build_model_artifact(_cfg())
    assert artifact.type == "model"


def test_build_model_artifact_metadata_carries_git_sha() -> None:
    """Metadata records git_sha per storage-provenance-spec §6."""
    artifact = build_model_artifact(_cfg())
    assert set(artifact.metadata) == {"git_sha"}
    assert isinstance(artifact.metadata["git_sha"], str)


def test_build_model_artifact_with_ckpt_uri_adds_s3_reference() -> None:
    """An uploaded ``r2://`` ckpt URI is referenced as an ``s3://`` URI."""
    artifact = build_model_artifact(_cfg(), ckpt_uri=_CKPT_URI)
    refs = {entry.ref for entry in artifact.manifest.entries.values()}
    assert refs == {_CKPT_S3_REF}


def test_build_model_artifact_without_ckpt_uri_adds_no_reference() -> None:
    """No ckpt URI (upload skipped) logs a lineage-only artifact with no reference."""
    artifact = build_model_artifact(_cfg())
    assert artifact.manifest.entries == {}


def test_build_model_artifact_merges_checkpoint_metadata_alongside_git_sha() -> None:
    """Checkpoint metadata joins ``git_sha`` so the 0-byte reference is identifiable (#2424)."""
    artifact = build_model_artifact(_cfg(), _CKPT_URI, {"epoch": 7, "ckpt_bytes": 336})
    assert artifact.metadata == {
        "git_sha": artifact.metadata["git_sha"],
        "epoch": 7,
        "ckpt_bytes": 336,
    }


def _fake_trainer(
    epoch: int = 3,
    global_step: int = 4200,
    monitor: str | None = "val/loss",
    best_model_score: Any = None,
) -> Any:
    """Build a finished-trainer stand-in exposing what ``_checkpoint_metadata`` reads.

    :param epoch: Value returned as ``trainer.current_epoch``.
    :param global_step: Value returned as ``trainer.global_step``.
    :param monitor: The checkpoint callback's monitored metric name, or ``None``.
    :param best_model_score: The checkpoint callback's best score (a tensor in real runs).
    :returns: A namespace with ``current_epoch``, ``global_step``, and ``checkpoint_callback``.
    """
    return SimpleNamespace(
        current_epoch=epoch,
        global_step=global_step,
        checkpoint_callback=SimpleNamespace(monitor=monitor, best_model_score=best_model_score),
    )


def _write_checkpoint(path: Path, epoch: int, global_step: int) -> Path:
    """Save a Lightning-shaped checkpoint carrying its own training counters.

    :param path: File the checkpoint is written to.
    :param epoch: Value stored as the checkpoint's ``epoch``.
    :param global_step: Value stored as the checkpoint's ``global_step``.
    :returns: The written path.
    """
    torch.save(
        {"epoch": epoch, "global_step": global_step, "state_dict": {"w": torch.zeros(2)}}, path
    )
    return path


def _fake_checkpoint_callback(best_model_path: str, dirpath: str | None) -> Any:
    """Build a ModelCheckpoint stand-in exposing the best path and its write directory.

    :param best_model_path: Value returned as ``best_model_path``.
    :param dirpath: Directory the callback writes into, or ``None`` when unset.
    :returns: A namespace with ``best_model_path`` and ``dirpath``.
    """
    return SimpleNamespace(best_model_path=best_model_path, dirpath=dirpath)


def test_checkpoint_path_written_by_run_keeps_a_path_under_the_callback_dir(
    tmp_path: Path,
) -> None:
    """A checkpoint this run wrote into its own callback directory is uploadable.

    :param tmp_path: Stands in for the run's checkpoint directory.
    """
    best = tmp_path / "checkpoints" / "epoch_003.ckpt"
    trainer = SimpleNamespace(
        checkpoint_callback=_fake_checkpoint_callback(str(best), str(tmp_path / "checkpoints"))
    )

    assert _checkpoint_path_written_by_run(cast(Any, trainer)) == str(best)


def test_checkpoint_path_written_by_run_drops_a_resumed_foreign_best(tmp_path: Path) -> None:
    """A best path restored from resumed callback state is not this run's to upload (#3260).

    :param tmp_path: Holds both the previous run's directory and this run's.
    """
    inherited = tmp_path / "final500" / "checkpoints" / "step_000500.ckpt"
    trainer = SimpleNamespace(
        checkpoint_callback=_fake_checkpoint_callback(
            str(inherited), str(tmp_path / "wandb5000" / "checkpoints")
        )
    )

    assert _checkpoint_path_written_by_run(cast(Any, trainer)) == ""


def test_checkpoint_path_written_by_run_warns_which_path_it_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Dropping an inherited best path is announced with the path, not silent.

    :param tmp_path: Holds both the previous run's directory and this run's.
    :param caplog: Captures the warning naming the dropped checkpoint.
    """
    inherited = tmp_path / "final500" / "checkpoints" / "step_000500.ckpt"
    trainer = SimpleNamespace(
        checkpoint_callback=_fake_checkpoint_callback(
            str(inherited), str(tmp_path / "wandb5000" / "checkpoints")
        )
    )

    with caplog.at_level(logging.WARNING):
        _checkpoint_path_written_by_run(cast(Any, trainer))

    assert "step_000500.ckpt" in caplog.text


def test_checkpoint_path_written_by_run_without_a_dirpath_keeps_the_path(tmp_path: Path) -> None:
    """With no write directory to compare against, the callback's own answer stands.

    :param tmp_path: Supplies the checkpoint path the callback reports.
    """
    best = tmp_path / "epoch_003.ckpt"
    trainer = SimpleNamespace(checkpoint_callback=_fake_checkpoint_callback(str(best), None))

    assert _checkpoint_path_written_by_run(cast(Any, trainer)) == str(best)


def test_checkpoint_path_written_by_run_without_a_checkpoint_callback_is_empty() -> None:
    """A run with checkpointing disabled entirely reports no path to upload."""
    trainer = SimpleNamespace(checkpoint_callback=None)

    assert _checkpoint_path_written_by_run(cast(Any, trainer)) == ""


def test_checkpoint_metadata_records_uri_epoch_step_and_size(tmp_path: Path) -> None:
    """The referenced checkpoint's URI, position in training, and byte size are recorded.

    :param tmp_path: Holds the local checkpoint whose counters and size are read.
    """
    ckpt = _write_checkpoint(tmp_path / "model.ckpt", epoch=3, global_step=4200)

    metadata = _checkpoint_metadata(_fake_trainer(), str(ckpt), _CKPT_URI)

    assert metadata["ckpt_uri"] == _CKPT_URI
    assert metadata["epoch"] == 3
    assert metadata["global_step"] == 4200
    assert metadata["ckpt_bytes"] == ckpt.stat().st_size


def test_checkpoint_metadata_reports_the_uploaded_file_not_the_trainer(tmp_path: Path) -> None:
    """A resumed run's stale best checkpoint is described by its own counters (#3260).

    :param tmp_path: Holds the step-500 checkpoint a step-5000 trainer uploads.
    """
    ckpt = _write_checkpoint(tmp_path / "step_000500.ckpt", epoch=8, global_step=500)

    metadata = _checkpoint_metadata(
        _fake_trainer(epoch=80, global_step=5000), str(ckpt), _CKPT_URI
    )

    assert metadata["epoch"] == 8
    assert metadata["global_step"] == 500


def test_checkpoint_metadata_stale_best_records_the_trainer_step_separately(
    tmp_path: Path,
) -> None:
    """The run that uploaded an older checkpoint stays visible beside its counters.

    :param tmp_path: Holds the step-500 checkpoint a step-5000 trainer uploads.
    """
    ckpt = _write_checkpoint(tmp_path / "step_000500.ckpt", epoch=8, global_step=500)

    metadata = _checkpoint_metadata(
        _fake_trainer(epoch=80, global_step=5000), str(ckpt), _CKPT_URI
    )

    assert metadata["trainer_global_step"] == 5000


def test_checkpoint_metadata_stale_best_warns_with_both_steps(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Uploading another run's checkpoint is announced in the run log, not just in W&B.

    :param tmp_path: Holds the step-500 checkpoint a step-5000 trainer uploads.
    :param caplog: Captures the warning naming both steps.
    """
    ckpt = _write_checkpoint(tmp_path / "step_000500.ckpt", epoch=8, global_step=500)

    with caplog.at_level(logging.WARNING):
        _checkpoint_metadata(_fake_trainer(epoch=80, global_step=5000), str(ckpt), _CKPT_URI)

    assert "500" in caplog.text and "5000" in caplog.text


def test_checkpoint_metadata_current_checkpoint_omits_the_trainer_step(tmp_path: Path) -> None:
    """A checkpoint written by this run carries no redundant trainer counter.

    :param tmp_path: Holds a checkpoint saved at the trainer's own step.
    """
    ckpt = _write_checkpoint(tmp_path / "model.ckpt", epoch=3, global_step=4200)

    metadata = _checkpoint_metadata(_fake_trainer(), str(ckpt), _CKPT_URI)

    assert "trainer_global_step" not in metadata


def test_checkpoint_metadata_unreadable_checkpoint_omits_counters(tmp_path: Path) -> None:
    """Bytes that are not a checkpoint yield no checkpoint step, only the uploading run's.

    :param tmp_path: Holds a file that torch cannot load.
    """
    corrupt = tmp_path / "model.ckpt"
    corrupt.write_bytes(b"x" * 17)

    metadata = _checkpoint_metadata(_fake_trainer(), str(corrupt), _CKPT_URI)

    assert "epoch" not in metadata
    assert "global_step" not in metadata
    assert metadata["trainer_global_step"] == 4200
    assert metadata["ckpt_bytes"] == 17


def test_checkpoint_metadata_records_monitored_metric_as_float(tmp_path: Path) -> None:
    """The monitored metric and its tensor score are recorded as a JSON-encodable float.

    :param tmp_path: Holds the local checkpoint whose size is read.
    """
    ckpt = _write_checkpoint(tmp_path / "model.ckpt", epoch=3, global_step=4200)

    metadata = _checkpoint_metadata(
        _fake_trainer(best_model_score=torch.tensor(0.327)), str(ckpt), _CKPT_URI
    )

    assert metadata["monitor"] == "val/loss"
    assert metadata["monitor_score"] == pytest.approx(0.327)


def test_checkpoint_metadata_without_score_omits_the_key(tmp_path: Path) -> None:
    """An unscored checkpoint omits ``monitor_score`` rather than recording ``None``.

    :param tmp_path: Holds the local checkpoint whose size is read.
    """
    ckpt = _write_checkpoint(tmp_path / "model.ckpt", epoch=3, global_step=4200)

    metadata = _checkpoint_metadata(_fake_trainer(monitor=None), str(ckpt), _CKPT_URI)

    assert "monitor" not in metadata
    assert "monitor_score" not in metadata


def test_checkpoint_metadata_unreadable_path_omits_size_without_raising(tmp_path: Path) -> None:
    """A vanished local checkpoint still yields metadata — size is dropped, not fatal.

    :param tmp_path: Supplies a path with no checkpoint written under it.
    """
    metadata = _checkpoint_metadata(_fake_trainer(), str(tmp_path / "gone.ckpt"), _CKPT_URI)

    assert "ckpt_bytes" not in metadata
    assert metadata["ckpt_uri"] == _CKPT_URI


def test_log_model_artifact_forwards_checkpoint_metadata() -> None:
    """``_log_model_artifact`` passes checkpoint metadata through to the logged artifact."""
    logger = _RecordingWandbLogger()

    _log_model_artifact([logger], _cfg(), _CKPT_URI, {"epoch": 9})

    assert logger.logged[0].metadata["epoch"] == 9


def test_upload_best_checkpoint_same_run_launches_keep_distinct_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two launches of one run upload readable checkpoints without overwriting.

    Drives the real ``rclone`` binary against a local-backed ``r2:`` remote, so
    the assertions read the materialized objects instead of mocked calls.

    :param tmp_path: Backs the ``r2:`` remote; the uploaded objects land under it.
    :param monkeypatch: Points rclone at the local fs and forces R2 reachable.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)
    first_ckpt = tmp_path / "epoch=3.ckpt"
    second_ckpt = tmp_path / "epoch=4.ckpt"
    first_ckpt.write_bytes(b"first weights")
    second_ckpt.write_bytes(b"second weights")

    first_uri = _upload_best_checkpoint(
        _cfg(task_name="flow-simple"), str(first_ckpt), _TRAINING_RUN_ID, _LAUNCH_UUID
    )
    second_uri = _upload_best_checkpoint(
        _cfg(task_name="flow-simple"),
        str(second_ckpt),
        _TRAINING_RUN_ID,
        _SECOND_LAUNCH_UUID,
    )

    assert first_uri != second_uri
    first_object = (
        tmp_path
        / "intermediate-data"
        / "checkpoints"
        / "flow-simple"
        / "flow-simple-20260908T170724945Z"
        / _LAUNCH_UUID
        / "model.ckpt"
    )
    second_object = first_object.parents[1] / _SECOND_LAUNCH_UUID / "model.ckpt"
    assert first_object.read_bytes() == b"first weights"
    assert second_object.read_bytes() == b"second weights"


def test_upload_best_checkpoint_unreachable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """When R2 is unavailable (local CPU / CI), no upload happens and None is returned.

    :param monkeypatch: Makes ``ensure_r2_env_loaded`` raise, simulating absent R2 creds.
    """

    def _unavailable(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("R2 credentials missing from process env")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _unavailable)
    assert (
        _upload_best_checkpoint(
            _cfg(), "/run/checkpoints/epoch=3.ckpt", _TRAINING_RUN_ID, _LAUNCH_UUID
        )
        is None
    )


def test_upload_best_checkpoint_empty_path_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``best_model_path`` (no checkpoint written) yields a lineage-only None.

    :param monkeypatch: Stubs ``ensure_r2_env_loaded`` so only the empty path gates.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)
    assert _upload_best_checkpoint(_cfg(), "", _TRAINING_RUN_ID, _LAUNCH_UUID) is None


def test_upload_best_checkpoint_upload_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An rclone upload failure degrades to lineage-only (None) instead of aborting the run.

    :param monkeypatch: Stubs R2 env-load as available and makes ``upload_to_uri`` raise.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)

    def _boom(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("rclone boom")

    monkeypatch.setattr(r2_io, "upload_to_uri", _boom)
    assert (
        _upload_best_checkpoint(
            _cfg(task_name="flow-simple"),
            "/x/epoch=3.ckpt",
            _TRAINING_RUN_ID,
            _LAUNCH_UUID,
        )
        is None
    )


def test_log_model_artifact_logs_to_wandb_logger() -> None:
    """A WandbLogger receives a ``model``-typed artifact with the expected name."""
    logger = _RecordingWandbLogger()
    _log_model_artifact([logger], _cfg(task_name="flow-simple"), None)
    assert len(logger.logged) == 1
    assert logger.logged[0].name == "model-flow-simple"
    assert logger.logged[0].type == "model"


def test_log_model_artifact_forwards_ckpt_uri_reference() -> None:
    """A ckpt URI passed through is referenced as ``s3://`` on the logged artifact."""
    logger = _RecordingWandbLogger()
    _log_model_artifact([logger], _cfg(task_name="flow-simple"), _CKPT_URI)
    refs = {entry.ref for entry in logger.logged[0].manifest.entries.values()}
    assert refs == {_CKPT_S3_REF}


def test_log_model_artifact_no_wandb_logger_is_noop() -> None:
    """With no WandbLogger present, logging is a no-op (does not raise)."""

    class _PlainLogger:
        pass

    _log_model_artifact([cast(Logger, _PlainLogger())], _cfg(), None)


def test_log_model_artifact_empty_loggers_is_noop() -> None:
    """An empty logger list is a no-op (the wandb-free default path)."""
    _log_model_artifact([], _cfg(), None)


def test_log_model_artifact_swallows_wandb_failure() -> None:
    """A wandb ``log_artifact`` failure warns and is swallowed, never aborting training."""

    class _FailingWandbLogger(_RecordingWandbLogger):
        def log_artifact(self, artifact: Any) -> NoReturn:  # type: ignore[override]
            raise RuntimeError("wandb boom")

    _log_model_artifact([_FailingWandbLogger()], _cfg(), None)


def _attach_offline_wandb_logger(cfg: DictConfig, save_dir: Path) -> None:
    """Swap ``cfg.logger`` for a real offline ``WandbLogger`` group rooted at ``save_dir``.

    The shared ``cfg_train`` fixture pins ``logger=None``, which makes the
    ``train()``-end artifact path a no-op; replacing it with an ``offline=True``
    WandbLogger is what forces the real ``_log_model_artifact`` call to run.

    :param cfg: Train cfg, mutated in place to carry ``logger.wandb``.
    :param save_dir: Directory the offline run's ``wandb/`` tree is written under.
    """
    with open_dict(cfg):
        cfg.logger = {
            "wandb": {
                "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                "offline": True,
                "save_dir": str(save_dir),
                "id": None,
                "job_type": "",
                "project": "train-model-artifact-test-project",
            }
        }


@pytest.mark.slow
def test_train_logs_model_artifact_to_offline_wandb_run(
    cfg_train: DictConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``train(cfg)`` end-to-end logs the ``model-{config_id}`` artifact to a real offline run.

    Drives the real entrypoint (1-step ``fast_dev_run`` CPU train/test) with a
    real ``WandbLogger(offline=True)`` swapped in for the fixture's ``logger=None``,
    then decodes the offline ``run-*.wandb`` binary the live client wrote. No wandb
    internals are mocked: the artifact name, ``model`` type, and ``git_sha`` metadata
    are read back from the bytes — so dropping or mis-gating the ``train()``-end
    ``_log_model_artifact`` call (which the cfg-level train tests no-op past) fails
    here. ``fast_dev_run`` writes no checkpoint and CI has no R2, so the artifact is
    lineage-only; the ``s3://`` reference path is pinned by the
    ``_upload_best_checkpoint`` / ``build_model_artifact`` unit tests above.

    :param cfg_train: Tiny CPU TorchSynth train config; no external plugin.
    :param tmp_path: Hosts the offline run dir and the model checkpoints.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()

    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
    _attach_offline_wandb_logger(cfg_train, tmp_path)

    train(cfg_train)
    assert wandb.run is None, "train() did not close the wandb run on return"

    offline_dirs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(offline_dirs) == 1, (
        f"expected one offline-run dir under {tmp_path / 'wandb'}, found {offline_dirs}"
    )
    binary_files = glob.glob(str(offline_dirs[0] / "run-*.wandb"))
    assert len(binary_files) == 1, (
        f"expected one .wandb binary in {offline_dirs[0]}, found {binary_files}"
    )

    payload = read_run_binary(
        Path(binary_files[0]),
        until=lambda data: _E2E_ARTIFACT_NAME.encode() in data,
    )
    assert _E2E_ARTIFACT_NAME.encode() in payload, (
        f"model artifact {_E2E_ARTIFACT_NAME!r} not recorded in offline run binary"
    )
    assert b"model" in payload, "artifact type 'model' not recorded"
    assert b"git_sha" in payload, "artifact metadata 'git_sha' not recorded in offline run binary"


@pytest.mark.slow
def test_checkpoint_metadata_describes_a_trainer_written_checkpoint(cfg_train: DictConfig) -> None:
    """A real two-step run's own checkpoint is described by the counters it stores.

    Drives the real ``train(cfg)`` entrypoint so the metadata is read back from
    bytes Lightning wrote, not from a hand-assembled payload.

    :param cfg_train: Tiny CPU TorchSynth train config; no external plugin.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = False
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.max_steps = 2
        cfg_train.trainer.limit_train_batches = 2
        cfg_train.trainer.limit_val_batches = 2
        cfg_train.trainer.val_check_interval = 2
        cfg_train.test = False

    _, object_dict = train(cfg_train)
    trainer = cast(Any, object_dict["trainer"])
    best_model_path = trainer.checkpoint_callback.best_model_path
    assert best_model_path, "the run wrote no checkpoint to describe"

    metadata = _checkpoint_metadata(trainer, best_model_path, _CKPT_URI)

    assert metadata["global_step"] == 2
    assert "trainer_global_step" not in metadata


@pytest.mark.slow
@pytest.mark.integration_r2
def test_train_uploaded_checkpoint_is_launch_scoped_and_described_by_artifact(
    cfg_train_lance: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real train run uploads and describes its checkpoint under its run ID.

    :param cfg_train_lance: CPU-cheap Lance train cfg, run for two steps so a checkpoint exists.
    :param tmp_path: Hosts the dataset, offline run directory, and training outputs.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` environment.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()

    bucket = "intermediate-data"

    with open_dict(cfg_train_lance):
        cfg_train_lance.trainer.fast_dev_run = False
        cfg_train_lance.trainer.max_epochs = 1
        cfg_train_lance.trainer.max_steps = 2
        cfg_train_lance.trainer.limit_train_batches = 2
        cfg_train_lance.trainer.limit_val_batches = 2
        cfg_train_lance.trainer.val_check_interval = 2
        cfg_train_lance.trainer.check_val_every_n_epoch = 1
        cfg_train_lance.test = False
        cfg_train_lance.r2.bucket = bucket
        cfg_train_lance.training.upload_checkpoints_uri = None
        # vst_ffn logs val/param_mse, not the group default's val/loss.
        cfg_train_lance.callbacks.model_checkpoint.monitor = "val/param_mse"
    _attach_offline_wandb_logger(cfg_train_lance, tmp_path)

    prefix = ""
    try:
        train(cfg_train_lance)

        training_run_id = str(cfg_train_lance.logger.wandb.id)
        prefix = f"checkpoints/train/{training_run_id}/"
        offline_dirs = list((tmp_path / "wandb").glob("offline-run-*"))
        binary = next(iter(offline_dirs[0].glob("run-*.wandb")))
        payload = read_run_binary(binary, until=lambda data: b'"ckpt_bytes"' in data)
        start = payload.find(b'{"git_sha"')
        metadata = json.loads(payload[start : payload.find(b"}", start) + 1])

        ckpt_uri = metadata["ckpt_uri"]
        assert re.fullmatch(rf"r2://{bucket}/{prefix}[0-9a-f]{{32}}/model\.ckpt", ckpt_uri)
        ckpt_bytes = r2_io.object_size(ckpt_uri)
        assert ckpt_bytes is not None and ckpt_bytes > 0
        assert metadata["ckpt_bytes"] == ckpt_bytes
        assert metadata["global_step"] == 2
        assert metadata["monitor"] == "val/param_mse"
        assert isinstance(metadata["monitor_score"], float)
    finally:
        if prefix:
            r2_io.purge_prefix(bucket, prefix)
