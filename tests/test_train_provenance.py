"""Provenance-wiring tests for the ``synth-setter-train`` entrypoint.

Pins the storage-provenance-spec.md run-id, job_type, and W&B-provenance
invariants at the ``train`` entrypoint seam: the run id is pinned in the
``{config_id}-{timestamp}`` convention with
``job_type=training``, and ``log_wandb_provenance`` is invoked once a logger
exists. Heavy collaborators (datamodule / model / trainer instantiation,
hyperparameter logging) are stubbed at their seams so the test isolates the
wiring rather than running a real fit. Sibling to ``test_train.py`` per the
``tests/_meta`` entrypoint-only rule.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.train import train

# ``make_wandb_run_id`` output: ``<config_id>-YYYYMMDDTHHMMSSsssZ``.
_RUN_ID_PATTERN = r"{config_id}-\d{{8}}T\d{{9}}Z"


def _leak_foreign_experiment_into_hydra_singleton() -> None:
    """Populate the process-global ``HydraConfig`` with a foreign experiment choice.

    Mimics a sibling test that composed an experiment and left the ``HydraConfig``
    singleton set, so ``resolve_run_config_id`` would read
    ``runtime.choices.experiment`` instead of falling back to ``task_name`` — the
    leak that flaked the provenance run-id tests under xdist (#1518, #1523).
    """
    hydra_conf = OmegaConf.structured(HydraConf)
    OmegaConf.update(hydra_conf, "runtime.choices.experiment", "surge/some_other_exp")
    HydraConfig.instance().set_config(OmegaConf.create({"hydra": hydra_conf}))


def _wandb_logger_cfg() -> DictConfig:
    """Build a minimal training cfg carrying a wandb logger group and a known task_name.

    :returns: Cfg with ``logger.wandb.{id,job_type}`` unset and ``train``/``test``
        disabled so the entrypoint reaches the provenance wiring without fitting.
    """
    return OmegaConf.create(
        {
            "task_name": "flow_simple",
            "paths": {"output_dir": "."},
            "logger": {"wandb": {"id": None, "job_type": ""}},
            "datamodule": {"_target_": "stub.datamodule"},
            "model": {"_target_": "stub.model"},
            "trainer": {"_target_": "stub.trainer"},
            "train": False,
            "test": False,
        }
    )


@pytest.fixture
def stubbed_train_entrypoint() -> Iterator[MagicMock]:
    """Stub ``train``'s heavy collaborators and spy on ``log_wandb_provenance``.

    Hydra instantiation returns mocks (the trainer mock carries a real-dict
    ``callback_metrics`` so the entrypoint's metric merge is a no-op), callbacks
    resolve to ``[]``, ``instantiate_loggers`` returns a truthy sentinel so the
    ``if logger:`` block runs, and ``log_hyperparameters`` is a no-op.

    :yields: The ``log_wandb_provenance`` spy for call-count assertions.
    :ytype: MagicMock
    """
    trainer = MagicMock()
    trainer.callback_metrics = {}
    with (
        patch("synth_setter.cli.train.hydra.utils.instantiate", return_value=trainer),
        patch("synth_setter.cli.train.instantiate_callbacks", return_value=[]),
        patch("synth_setter.cli.train.instantiate_loggers", return_value=[object()]),
        patch("synth_setter.cli.train.log_hyperparameters"),
        patch("synth_setter.cli.train.log_wandb_provenance") as provenance_spy,
    ):
        yield provenance_spy


class TestTrainProvenanceWiring:
    """The train entrypoint pins the run identity and stamps provenance."""

    def test_pins_run_id_in_config_id_timestamp_convention(
        self, stubbed_train_entrypoint: MagicMock
    ) -> None:
        """``cfg.logger.wandb.id`` is pinned as ``{task_name}-{timestamp}``.

        :param stubbed_train_entrypoint: Collaborator-stub fixture.
        """
        cfg = _wandb_logger_cfg()

        train(cfg)

        assert re.fullmatch(_RUN_ID_PATTERN.format(config_id="flow_simple"), cfg.logger.wandb.id)

    def test_pins_run_id_to_task_name_after_leaked_experiment_is_reset(
        self, stubbed_train_entrypoint: MagicMock
    ) -> None:
        """Clearing a leaked ``HydraConfig`` restores the ``task_name`` run-id fallback.

        Pins the contract that the autouse ``_reset_hydra_config_singleton`` fixture
        (``tests/conftest.py``) relies on: after a foreign
        ``runtime.choices.experiment`` is cleared from the process-global singleton,
        the pinned run id resolves to ``{task_name}-{timestamp}`` rather than the
        leaked experiment basename — the xdist flake fixed in #1518, #1523. The test
        clears the singleton itself so it is deterministic regardless of ordering.

        :param stubbed_train_entrypoint: Collaborator-stub fixture.
        """
        _leak_foreign_experiment_into_hydra_singleton()
        HydraConfig.instance().cfg = None
        cfg = _wandb_logger_cfg()

        train(cfg)

        assert re.fullmatch(_RUN_ID_PATTERN.format(config_id="flow_simple"), cfg.logger.wandb.id)

    def test_pins_training_job_type(self, stubbed_train_entrypoint: MagicMock) -> None:
        """``cfg.logger.wandb.job_type`` is pinned to ``training``.

        :param stubbed_train_entrypoint: Collaborator-stub fixture.
        """
        cfg = _wandb_logger_cfg()

        train(cfg)

        assert cfg.logger.wandb.job_type == "training"

    def test_stamps_wandb_provenance_once(self, stubbed_train_entrypoint: MagicMock) -> None:
        """``log_wandb_provenance`` is invoked exactly once when a logger exists.

        :param stubbed_train_entrypoint: Collaborator-stub fixture exposing the spy.
        """
        train(_wandb_logger_cfg())

        stubbed_train_entrypoint.assert_called_once_with()
