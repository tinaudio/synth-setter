"""Unit + offline-e2e tests for the train-side model W&B artifact wiring.

``build_model_artifact`` is asserted on a real ``wandb.Artifact`` (no run, no
network): name, type, the opt-in R2 reference, and metadata are all observable
on the returned object, so these tests exercise the real construction rather
than a mock of it. ``_log_model_artifact`` is driven against a ``WandbLogger``
subclass stub to pin the WandbLogger-only / best-effort contract.

``test_train_logs_model_artifact_to_offline_wandb_run`` closes the gap those
unit tests leave open: it drives the real ``train(cfg)`` entrypoint with a real
``WandbLogger(offline=True)`` and decodes the offline ``run-*.wandb`` binary, so
dropping or mis-gating the ``train()``-end ``_log_model_artifact`` call (which
every cfg-level train test no-ops past with ``logger=None``) fails here.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
import wandb
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import _log_model_artifact, build_model_artifact, train
from tests.helpers.wandb_offline import read_run_binary

_CKPT_URI = "r2://models/model-flow-simple/best.ckpt"

# `cfg_train` composes no Hydra experiment, so `resolve_run_config_id` falls back
# to `task_name` ("train") — the config_id the e2e artifact name is built from.
_E2E_ARTIFACT_NAME = "model-train"
_E2E_UPLOAD_URI = "r2://models/model-train/best.ckpt"
_E2E_S3_REF = "s3://models/model-train/best.ckpt"


def _cfg(task_name: str = "flow-simple", upload_checkpoints_uri: str | None = None) -> DictConfig:
    """Build a minimal train cfg carrying ``task_name`` and the opt-in upload URI.

    No experiment is composed, so ``resolve_run_config_id`` falls back to
    ``task_name`` — the config_id the artifact name is built from.

    :param task_name: Drives the resolved config_id and thus the artifact name.
    :param upload_checkpoints_uri: ``r2://`` ckpt prefix, or ``None`` for the
        lineage-only (no-reference) default.
    :returns: A DictConfig with ``task_name`` and ``training.upload_checkpoints_uri``.
    """
    return cast(
        DictConfig,
        OmegaConf.create(
            {
                "task_name": task_name,
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


def test_build_model_artifact_with_uri_adds_s3_reference() -> None:
    """A configured ``r2://`` ckpt URI is referenced as an ``s3://`` URI."""
    artifact = build_model_artifact(_cfg(upload_checkpoints_uri=_CKPT_URI))
    refs = {entry.ref for entry in artifact.manifest.entries.values()}
    assert refs == {"s3://models/model-flow-simple/best.ckpt"}


def test_build_model_artifact_without_uri_adds_no_reference() -> None:
    """A null ckpt URI logs a lineage-only artifact with no R2 reference (#92 pending)."""
    artifact = build_model_artifact(_cfg(upload_checkpoints_uri=None))
    assert artifact.manifest.entries == {}


def test_log_model_artifact_logs_to_wandb_logger() -> None:
    """A WandbLogger receives a ``model``-typed artifact with the expected name."""
    logger = _RecordingWandbLogger()
    _log_model_artifact([logger], _cfg(task_name="flow-simple"))
    assert len(logger.logged) == 1
    assert logger.logged[0].name == "model-flow-simple"
    assert logger.logged[0].type == "model"


def test_log_model_artifact_no_wandb_logger_is_noop() -> None:
    """With no WandbLogger present, logging is a no-op (does not raise)."""

    class _PlainLogger:
        pass

    _log_model_artifact([cast(Logger, _PlainLogger())], _cfg())


def test_log_model_artifact_empty_loggers_is_noop() -> None:
    """An empty logger list is a no-op (the wandb-free default path)."""
    _log_model_artifact([], _cfg())


def test_log_model_artifact_swallows_wandb_failure() -> None:
    """A wandb ``log_artifact`` failure warns and is swallowed, never aborting training."""

    class _FailingWandbLogger(_RecordingWandbLogger):
        def log_artifact(self, artifact: Any) -> NoReturn:  # type: ignore[override]
            raise RuntimeError("wandb boom")

    _log_model_artifact([_FailingWandbLogger()], _cfg())


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
@pytest.mark.parametrize("upload_checkpoints_uri", [None, _E2E_UPLOAD_URI])
def test_train_logs_model_artifact_to_offline_wandb_run(
    cfg_train: DictConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upload_checkpoints_uri: str | None,
) -> None:
    """``train(cfg)`` end-to-end logs the ``model-{config_id}`` artifact to a real offline run.

    Drives the real entrypoint (1-step ``fast_dev_run`` CPU train/test) with a
    real ``WandbLogger(offline=True)`` swapped in for the fixture's ``logger=None``,
    then decodes the offline ``run-*.wandb`` binary the live client wrote. No wandb
    internals are mocked: the artifact name, ``model`` type, and ``git_sha`` metadata
    are read back from the bytes — so dropping or mis-gating the ``train()``-end
    ``_log_model_artifact`` call (which the cfg-level train tests no-op past) fails
    here. The ``upload_checkpoints_uri`` leg additionally pins the opt-in ``s3://``
    reference into the recorded artifact record.

    :param cfg_train: Tiny CPU train cfg (``datamodule=ksin``, ``model=ffn``); no VST.
    :param tmp_path: Hosts the offline run dir and the model checkpoints.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env.
    :param upload_checkpoints_uri: ``None`` for the lineage-only leg, or an ``r2://``
        prefix whose ``s3://`` rewrite must land in the artifact record.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()

    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.training.upload_checkpoints_uri = upload_checkpoints_uri
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

    needs_ref = upload_checkpoints_uri is not None
    payload = read_run_binary(
        Path(binary_files[0]),
        until=lambda data: (
            _E2E_ARTIFACT_NAME.encode() in data and (not needs_ref or _E2E_S3_REF.encode() in data)
        ),
    )
    assert _E2E_ARTIFACT_NAME.encode() in payload, (
        f"model artifact {_E2E_ARTIFACT_NAME!r} not recorded in offline run binary"
    )
    assert b"model" in payload, "artifact type 'model' not recorded"
    assert b"git_sha" in payload, "artifact metadata 'git_sha' not recorded in offline run binary"
    if needs_ref:
        assert _E2E_S3_REF.encode() in payload, (
            f"opt-in checkpoint reference {_E2E_S3_REF!r} not recorded on the artifact"
        )
