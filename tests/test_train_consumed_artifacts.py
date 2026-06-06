"""Unit tests for ``train._consumed_artifact_refs`` and the lineage call seam.

The pure helper maps the opt-in ``consumed_*`` cfg fields to the
``(name, alias)`` lineage edges training feeds to ``use_input_artifacts``
(``storage-provenance-spec.md`` §5). The seam tests below drive the real
``train(cfg)`` with its heavy collaborators stubbed and pin that the entrypoint
actually calls ``use_input_artifacts`` with those edges, gated on
``train``/``test`` — coverage the isolated helper tests cannot give. Both kept
out of the canonical ``test_train.py`` per
``tests/_meta/test_entrypoint_test_modules.py``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.train import _consumed_artifact_refs, train


def _seam_cfg(
    output_dir: Path,
    *,
    dataset_id: str | None,
    train_flag: bool,
    test_flag: bool,
    alias: str = "latest",
) -> DictConfig:
    """Build a minimal train cfg that drives ``train(cfg)`` under stubbed collaborators.

    Only the keys ``train`` reads are populated; every instantiated object is a
    mock, so no real datamodule/model/trainer is built.

    :param output_dir: ``paths.output_dir`` — read only by ``task_wrapper``'s
        finally-block log line, never written to.
    :param dataset_id: ``consumed_dataset_config_id`` value (``None`` = unset).
    :param train_flag: ``cfg.train`` — gates ``trainer.fit`` and the lineage edge.
    :param test_flag: ``cfg.test`` — gates ``trainer.test`` and the lineage edge.
    :param alias: ``consumed_artifact_alias`` flowing into the edge.
    :returns: A ``DictConfig`` accepted by ``train``.
    """
    return OmegaConf.create(
        {
            "seed": None,
            "datamodule": {"_target_": "stub.Datamodule"},
            "model": {"_target_": "stub.Model"},
            "trainer": {"_target_": "stub.Trainer"},
            "callbacks": None,
            "logger": {"wandb": {"_target_": "stub.WandbLogger"}},
            "watch_gradients": False,
            "train": train_flag,
            "test": test_flag,
            "ckpt_path": None,
            "consumed_dataset_config_id": dataset_id,
            "consumed_artifact_alias": alias,
            "paths": {"output_dir": str(output_dir)},
        }
    )


@contextlib.contextmanager
def _stub_train_collaborators(logger_sentinel: object) -> Iterator[MagicMock]:
    """Patch ``train``'s heavy collaborators and yield the ``use_input_artifacts`` spy.

    ``instantiate_loggers`` returns ``logger_sentinel`` so the test can assert the
    exact object handed to ``use_input_artifacts``; the trainer is a mock whose
    ``fit``/``test`` are inert, and the hyperparameter/provenance writers are
    no-ops so nothing touches wandb or disk.

    :param logger_sentinel: Object ``instantiate_loggers`` is stubbed to return.
    :yields MagicMock: The patched ``use_input_artifacts`` mock for call assertions.
    """
    # The trainer is the third ``instantiate`` call; ``train`` merges its
    # ``callback_metrics`` into a dict, so back the attribute with a real dict.
    instantiated = MagicMock()
    instantiated.callback_metrics = {}
    instantiated.checkpoint_callback.best_model_path = "ckpt.ckpt"
    with (
        patch("synth_setter.cli.train.use_input_artifacts") as spy,
        patch("synth_setter.cli.train.hydra.utils.instantiate", return_value=instantiated),
        patch("synth_setter.cli.train.instantiate_callbacks", return_value=[]),
        patch("synth_setter.cli.train.instantiate_loggers", return_value=logger_sentinel),
        patch("synth_setter.cli.train.log_hyperparameters"),
        patch("synth_setter.cli.train.log_wandb_provenance"),
        patch("synth_setter.cli.train.pin_wandb_run_id"),
        patch("synth_setter.cli.train.make_wandb_run_id", return_value="rid"),
        patch("synth_setter.cli.train.resolve_run_config_id", return_value="cid"),
    ):
        yield spy


def test_train_calls_use_input_artifacts_with_dataset_edge_when_id_set(tmp_path: Path) -> None:
    """``train`` hands the logger and the ``data-{id}`` edge to ``use_input_artifacts``.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir``.
    """
    logger_sentinel = MagicMock(name="loggers")
    cfg = _seam_cfg(tmp_path, dataset_id="diva-v1", train_flag=True, test_flag=False)

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_called_once_with(logger_sentinel, [("data-diva-v1", "latest")])


def test_train_calls_use_input_artifacts_with_empty_edges_when_id_unset(tmp_path: Path) -> None:
    """An unset dataset id drives the no-edge path: the spy is called with ``[]``.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir``.
    """
    logger_sentinel = MagicMock(name="loggers")
    cfg = _seam_cfg(tmp_path, dataset_id=None, train_flag=True, test_flag=False)

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_called_once_with(logger_sentinel, [])


def test_train_records_lineage_when_only_test_is_true(tmp_path: Path) -> None:
    """A test-only run (``train=False, test=True``) still records the dataset edge.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir``.
    """
    logger_sentinel = MagicMock(name="loggers")
    cfg = _seam_cfg(tmp_path, dataset_id="diva-v1", train_flag=False, test_flag=True)

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_called_once_with(logger_sentinel, [("data-diva-v1", "latest")])


def test_train_skips_lineage_when_train_and_test_both_false(tmp_path: Path) -> None:
    """Neither ``train`` nor ``test`` set means the lineage gate stays shut.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir``.
    """
    logger_sentinel = MagicMock(name="loggers")
    cfg = _seam_cfg(tmp_path, dataset_id="diva-v1", train_flag=False, test_flag=False)

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_not_called()


def test_consumed_artifact_refs_dataset_id_set_returns_data_edge() -> None:
    """A set dataset config_id yields one ``data-{id}`` edge at the alias."""
    cfg = OmegaConf.create(
        {"consumed_dataset_config_id": "diva-v1", "consumed_artifact_alias": "latest"}
    )

    assert _consumed_artifact_refs(cfg) == [("data-diva-v1", "latest")]


def test_consumed_artifact_refs_dataset_id_null_returns_empty() -> None:
    """A null dataset config_id yields no edges — the opt-out no-op path."""
    cfg = OmegaConf.create(
        {"consumed_dataset_config_id": None, "consumed_artifact_alias": "latest"}
    )

    assert _consumed_artifact_refs(cfg) == []


def test_consumed_artifact_refs_alias_override_is_honored() -> None:
    """A non-default alias flows into the edge verbatim."""
    cfg = OmegaConf.create(
        {"consumed_dataset_config_id": "diva-v1", "consumed_artifact_alias": "v3"}
    )

    assert _consumed_artifact_refs(cfg) == [("data-diva-v1", "v3")]
