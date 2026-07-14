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
import os
import shutil
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
import wandb
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import (
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


def test_derive_checkpoint_uri_default_uses_bucket_and_config_id() -> None:
    """A null override derives ``r2://{bucket}/checkpoints/{config_id}/model.ckpt``."""
    uri = _derive_checkpoint_uri(_cfg(task_name="flow-simple"))
    assert uri == "r2://intermediate-data/checkpoints/flow-simple/model.ckpt"


def test_derive_checkpoint_uri_override_is_used_verbatim() -> None:
    """A set ``upload_checkpoints_uri`` overrides the derived path verbatim."""
    uri = _derive_checkpoint_uri(_cfg(upload_checkpoints_uri=_CKPT_URI))
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


def test_upload_best_checkpoint_reachable_uploads_to_derived_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When R2 is reachable, the best ckpt uploads to the derived URI (renamed model.ckpt).

    Drives the real ``rclone`` binary against a local-backed ``r2:`` remote, so
    the assertion is the object materializing on disk — not a mocked call.

    :param tmp_path: Backs the ``r2:`` remote; the uploaded object lands under it.
    :param monkeypatch: Points rclone at the local fs and forces R2 reachable.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")
    monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "local")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)
    ckpt = tmp_path / "epoch=3.ckpt"
    ckpt.write_bytes(b"weights")

    uri = _upload_best_checkpoint(_cfg(task_name="flow-simple"), str(ckpt))

    assert uri == "r2://intermediate-data/checkpoints/flow-simple/model.ckpt"
    landed = tmp_path / "intermediate-data" / "checkpoints" / "flow-simple" / "model.ckpt"
    assert landed.read_bytes() == b"weights"


def test_upload_best_checkpoint_unreachable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """When R2 is unavailable (local CPU / CI), no upload happens and None is returned.

    :param monkeypatch: Makes ``ensure_r2_env_loaded`` raise, simulating absent R2 creds.
    """

    def _unavailable(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("R2 credentials missing from process env")

    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _unavailable)
    assert _upload_best_checkpoint(_cfg(), "/run/checkpoints/epoch=3.ckpt") is None


def test_upload_best_checkpoint_empty_path_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``best_model_path`` (no checkpoint written) yields a lineage-only None.

    :param monkeypatch: Stubs ``ensure_r2_env_loaded`` so only the empty path gates.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *a, **k: None)
    assert _upload_best_checkpoint(_cfg(), "") is None


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
    assert _upload_best_checkpoint(_cfg(task_name="flow-simple"), "/x/epoch=3.ckpt") is None


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

    :param cfg_train: Tiny CPU train cfg (``datamodule=ksin``, ``model=ffn``); no VST.
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
