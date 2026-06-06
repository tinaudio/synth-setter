"""Unit tests for eval's ``eval-results`` W&B artifact builder and logger.

Asserts on a real ``wandb.Artifact`` object (no run, no network): name,
type, R2 reference, and metadata are observable on the returned artifact,
so these tests exercise the real construction. The best-effort logging
helper is covered with fakes for the no-op and swallow-on-error paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.eval import (
    _eval_summary_metrics,
    _log_eval_results_artifact,
    build_eval_results_artifact,
)

_CONFIG_ID = "nsynth-v1"
_UPLOAD_URI = "r2://eval-artifacts/eval-nsynth-v1-20260606T000000000Z"
_GIT_SHA = "0" * 40


def _cfg(task_name: str = _CONFIG_ID) -> DictConfig:
    """Build the minimal cfg slice the artifact path reads.

    :param task_name: Resolves the config_id when no Hydra experiment is
        selected (the test path), so the artifact name is deterministic.
    :returns: A :class:`DictConfig` carrying only ``task_name``.
    """
    return OmegaConf.create({"task_name": task_name})  # type: ignore[no-any-return]


def test_build_eval_results_artifact_name_is_eval_prefixed_config_id() -> None:
    """The artifact name is ``eval-{config_id}`` per storage-provenance-spec §4."""
    artifact = build_eval_results_artifact(_cfg(), _UPLOAD_URI, {}, _GIT_SHA)
    assert artifact.name == f"eval-{_CONFIG_ID}"


def test_build_eval_results_artifact_type_is_eval_results() -> None:
    """The artifact type is ``eval-results`` per storage-provenance-spec §4."""
    artifact = build_eval_results_artifact(_cfg(), _UPLOAD_URI, {}, _GIT_SHA)
    assert artifact.type == "eval-results"


def test_build_eval_results_artifact_references_upload_prefix_as_s3() -> None:
    """The R2 output prefix is referenced as an ``s3://`` URI per storage-provenance-spec §4."""
    artifact = build_eval_results_artifact(_cfg(), _UPLOAD_URI, {}, _GIT_SHA)
    refs = {entry.ref for entry in artifact.manifest.entries.values()}
    assert refs == {"s3://eval-artifacts/eval-nsynth-v1-20260606T000000000Z"}


def test_build_eval_results_artifact_metadata_carries_summary_metrics_and_git_sha() -> None:
    """Metadata round-trips the scalar summary metrics plus ``git_sha`` per §6."""
    artifact = build_eval_results_artifact(_cfg(), _UPLOAD_URI, {"test/param_mse": 0.25}, _GIT_SHA)
    assert artifact.metadata == {"test/param_mse": 0.25, "git_sha": _GIT_SHA}


def test_eval_summary_metrics_coerces_scalar_tensors_to_floats() -> None:
    """A 0-d tensor metric is coerced to a native float for JSON-safe metadata."""
    summary = _eval_summary_metrics({"test/param_mse": torch.tensor(0.5)})
    assert summary == {"test/param_mse": 0.5}
    assert isinstance(summary["test/param_mse"], float)


def test_eval_summary_metrics_drops_non_scalar_values() -> None:
    """Non-scalar values (vectors) are dropped so metadata stays small and scalar."""
    summary = _eval_summary_metrics({"vec": torch.tensor([1.0, 2.0]), "scalar": 3.0})
    assert summary == {"scalar": 3.0}


def _wandb_logger_mock() -> MagicMock:
    """Return a ``WandbLogger`` mock whose ``.experiment.log_artifact`` is observable.

    :returns: A ``MagicMock(spec=WandbLogger)`` so ``isinstance`` narrows it.
    """
    return MagicMock(spec=WandbLogger)


def test_log_eval_results_artifact_logs_artifact_to_wandb_logger() -> None:
    """A ``WandbLogger`` with a set upload URI receives the built artifact."""
    logger = _wandb_logger_mock()
    _log_eval_results_artifact([logger], _cfg(), _UPLOAD_URI, {"test/param_mse": 0.0}, _GIT_SHA)
    logger.experiment.log_artifact.assert_called_once()
    artifact = logger.experiment.log_artifact.call_args.args[0]
    assert artifact.name == f"eval-{_CONFIG_ID}"
    assert artifact.type == "eval-results"


def test_log_eval_results_artifact_noop_when_no_wandb_logger() -> None:
    """A logger list with no ``WandbLogger`` logs nothing."""
    # A non-WandbLogger entry is skipped; the only assertion is that it does not raise.
    _log_eval_results_artifact([MagicMock(spec=Logger)], _cfg(), _UPLOAD_URI, {}, _GIT_SHA)


def test_log_eval_results_artifact_noop_when_upload_uri_unset() -> None:
    """A null upload URI is a no-op even with a ``WandbLogger`` present."""
    logger = _wandb_logger_mock()
    _log_eval_results_artifact([logger], _cfg(), None, {}, _GIT_SHA)
    logger.experiment.log_artifact.assert_not_called()


def test_log_eval_results_artifact_swallows_logging_error() -> None:
    """A failing ``log_artifact`` is warned and swallowed so eval still completes."""
    logger = _wandb_logger_mock()
    logger.experiment.log_artifact.side_effect = RuntimeError("wandb down")
    # Must not raise despite the logging backend failing.
    _log_eval_results_artifact([logger], _cfg(), _UPLOAD_URI, {}, _GIT_SHA)
