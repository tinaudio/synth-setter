"""Unit tests for ``train.build_model_artifact`` and ``train._log_model_artifact``.

``build_model_artifact`` is asserted on a real ``wandb.Artifact`` (no run, no
network): name, type, the opt-in R2 reference, and metadata are all observable
on the returned object, so these tests exercise the real construction rather
than a mock of it. ``_log_model_artifact`` is driven against a ``WandbLogger``
subclass stub to pin the WandbLogger-only / best-effort contract.
"""

from __future__ import annotations

from typing import Any, NoReturn, cast

from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.train import _log_model_artifact, build_model_artifact

_CKPT_URI = "r2://models/model-flow-simple/best.ckpt"


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
