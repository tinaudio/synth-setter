"""Unit tests for ``eval._consumed_artifact_refs`` and the lineage call seam.

The pure helper maps the opt-in ``consumed_*`` cfg fields to the
``(name, alias)`` lineage edges eval feeds to ``use_input_artifacts`` — both
the model and the dataset it consumes (``storage-provenance-spec.md`` §5). The
seam test below drives the real ``evaluate(cfg)`` with its heavy collaborators
stubbed and pins that the entrypoint calls ``use_input_artifacts`` once with the
model edge ordered before the dataset edge — coverage the isolated helper tests
cannot give. Both kept out of the canonical ``test_eval.py`` per
``tests/_meta/test_entrypoint_test_modules.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from synth_setter.cli.eval import _consumed_artifact_refs, evaluate


def test_evaluate_calls_use_input_artifacts_with_model_edge_before_data_edge(
    tmp_path: Path,
) -> None:
    """``evaluate`` hands the logger and the model-then-data edges to ``use_input_artifacts``.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir`` (read only by
        ``task_wrapper``'s finally-block log line).
    """
    logger_sentinel = MagicMock(name="loggers")
    cfg = OmegaConf.create(
        {
            "datamodule": {"_target_": "stub.Datamodule"},
            "model": {"_target_": "stub.Model"},
            "trainer": {"_target_": "stub.Trainer"},
            "callbacks": None,
            "logger": {"wandb": {"_target_": "stub.WandbLogger"}},
            "ckpt_path": None,
            "mode": "test",
            "consumed_train_config_id": "flow-simple",
            "consumed_dataset_config_id": "diva-v1",
            "consumed_artifact_alias": "latest",
            "paths": {"output_dir": str(tmp_path)},
        }
    )
    # ``evaluate`` builds ``dict(trainer.callback_metrics)``, so back it with a real dict.
    instantiated = MagicMock()
    instantiated.callback_metrics = {}

    with (
        patch("synth_setter.cli.eval.use_input_artifacts") as spy,
        patch("synth_setter.cli.eval.hydra.utils.instantiate", return_value=instantiated),
        patch("synth_setter.cli.eval.instantiate_callbacks", return_value=[]),
        patch("synth_setter.cli.eval.instantiate_loggers", return_value=logger_sentinel),
        patch("synth_setter.cli.eval.log_hyperparameters"),
        patch("synth_setter.cli.eval.log_wandb_provenance"),
        patch("synth_setter.cli.eval.pin_wandb_run_id"),
        patch("synth_setter.cli.eval.make_wandb_run_id", return_value="rid"),
        patch("synth_setter.cli.eval.resolve_run_config_id", return_value="cid"),
    ):
        evaluate(cfg)

    spy.assert_called_once_with(
        logger_sentinel,
        [("model-flow-simple", "latest"), ("data-diva-v1", "latest")],
    )


def test_consumed_artifact_refs_both_ids_set_returns_model_then_data_edges() -> None:
    """Set train + dataset ids yield the model edge first, then the dataset edge."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": "flow-simple",
            "consumed_dataset_config_id": "diva-v1",
            "consumed_artifact_alias": "latest",
        }
    )

    assert _consumed_artifact_refs(cfg) == [
        ("model-flow-simple", "latest"),
        ("data-diva-v1", "latest"),
    ]


def test_consumed_artifact_refs_only_model_id_set_returns_model_edge_only() -> None:
    """A set train id with a null dataset id yields the model edge alone."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": "flow-simple",
            "consumed_dataset_config_id": None,
            "consumed_artifact_alias": "latest",
        }
    )

    assert _consumed_artifact_refs(cfg) == [("model-flow-simple", "latest")]


def test_consumed_artifact_refs_only_dataset_id_set_returns_data_edge_only() -> None:
    """A set dataset id with a null train id yields the dataset edge alone."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": None,
            "consumed_dataset_config_id": "diva-v1",
            "consumed_artifact_alias": "latest",
        }
    )

    assert _consumed_artifact_refs(cfg) == [("data-diva-v1", "latest")]


def test_consumed_artifact_refs_both_ids_null_returns_empty() -> None:
    """Both ids null yields no edges — the opt-out no-op path."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": None,
            "consumed_dataset_config_id": None,
            "consumed_artifact_alias": "latest",
        }
    )

    assert _consumed_artifact_refs(cfg) == []
