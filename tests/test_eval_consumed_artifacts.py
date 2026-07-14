"""Unit tests for ``eval._consumed_artifact_refs`` and the lineage call seam.

The pure helper maps the model config ID and local dataset provenance to
``(name, alias)`` lineage edges eval feeds to ``use_input_artifacts`` — both
the model and the dataset it consumes (``storage-provenance-spec.md`` §5). The
seam test below drives the real ``evaluate(cfg)`` with its heavy collaborators
stubbed and pins that the entrypoint calls ``use_input_artifacts`` once with the
model edge ordered before the dataset edge — coverage the isolated helper tests
cannot give. Both kept out of the canonical ``test_eval.py`` per
``tests/_meta/test_entrypoint_test_modules.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from synth_setter.cli.eval import _consumed_artifact_refs, evaluate
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import write_spec_to_path


def test_evaluate_calls_use_input_artifacts_with_model_edge_before_data_edge(
    tmp_path: Path, dataset_spec_factory: Callable[..., DatasetSpec]
) -> None:
    """``evaluate`` hands the logger and the model-then-data edges to ``use_input_artifacts``.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir`` (read only by
        ``task_wrapper``'s finally-block log line).
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    logger_sentinel = MagicMock(name="loggers")
    write_spec_to_path(
        dataset_spec_factory(
            task_name="diva-v1",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        tmp_path / "input_spec.json",
    )
    cfg = OmegaConf.create(
        {
            "datamodule": {"_target_": "stub.Datamodule", "dataset_root": str(tmp_path)},
            "model": {"_target_": "stub.Model"},
            "trainer": {"_target_": "stub.Trainer"},
            "callbacks": None,
            "logger": {"wandb": {"_target_": "stub.WandbLogger"}},
            "ckpt_path": None,
            "mode": "test",
            "consumed_train_config_id": "flow-simple",
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


def test_consumed_artifact_refs_model_and_dataset_present_returns_model_then_data_edges(
    tmp_path: Path, dataset_spec_factory: Callable[..., DatasetSpec]
) -> None:
    """A model ID plus discovered dataset preserves model-before-data ordering.

    :param tmp_path: Local dataset root containing the frozen input spec.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": "flow-simple",
            "datamodule": {"dataset_root": str(tmp_path)},
        }
    )

    write_spec_to_path(
        dataset_spec_factory(
            task_name="diva-v1",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        tmp_path / "input_spec.json",
    )
    assert _consumed_artifact_refs(cfg) == [
        ("model-flow-simple", "latest"),
        ("data-diva-v1", "latest"),
    ]


def test_consumed_artifact_refs_missing_dataset_provenance_returns_model_edge_only() -> None:
    """A model ID without a discoverable local dataset keeps its model edge."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": "flow-simple",
            "datamodule": {"dataset_root": "/datasets/surge"},
        }
    )

    assert _consumed_artifact_refs(cfg) == [("model-flow-simple", "latest")]


def test_consumed_artifact_refs_model_id_absent_returns_dataset_edge_only(
    tmp_path: Path, dataset_spec_factory: Callable[..., DatasetSpec]
) -> None:
    """A discovered local dataset does not require a model config ID.

    :param tmp_path: Local dataset root containing the frozen input spec.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": None,
            "datamodule": {"dataset_root": str(tmp_path)},
        }
    )

    write_spec_to_path(
        dataset_spec_factory(
            task_name="diva-v1",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        tmp_path / "input_spec.json",
    )
    assert _consumed_artifact_refs(cfg) == [("data-diva-v1", "latest")]


def test_consumed_artifact_refs_without_model_or_dataset_returns_empty() -> None:
    """An eval without model or local dataset provenance creates no input links."""
    cfg = OmegaConf.create(
        {
            "consumed_train_config_id": None,
            "datamodule": {},
        }
    )

    assert _consumed_artifact_refs(cfg) == []
