"""Provenance-wiring tests for the ``synth-setter-eval`` entrypoint.

Pins the storage-provenance-spec.md run-id, job_type, and W&B-provenance
invariants at the ``evaluate`` entrypoint seam: the run id is pinned in the
``{config_id}-{timestamp}`` convention with ``job_type=evaluation``, and
``log_wandb_provenance`` is invoked
once a logger exists. Heavy collaborators (datamodule / model / trainer
instantiation, hyperparameter logging, test loop) are stubbed at their seams so
the test isolates the wiring rather than running a real evaluation. Sibling to
``test_eval.py`` per the ``tests/_meta`` entrypoint-only rule.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli.eval import evaluate

# ``make_wandb_run_id`` output: ``<config_id>-YYYYMMDDTHHMMSSsssZ``.
_RUN_ID_PATTERN = r"{config_id}-\d{{8}}T\d{{9}}Z"


def _wandb_logger_cfg() -> DictConfig:
    """Build a minimal eval cfg carrying a wandb logger group and a known task_name.

    :returns: Cfg with ``logger.wandb.{id,job_type}`` unset and ``mode=test`` so
        the entrypoint reaches the provenance wiring with a no-op test loop.
    """
    return OmegaConf.create(
        {
            "task_name": "flow_simple",
            "paths": {"output_dir": "."},
            "logger": {"wandb": {"id": None, "job_type": ""}},
            "datamodule": {"_target_": "stub.datamodule"},
            "model": {"_target_": "stub.model"},
            "trainer": {"_target_": "stub.trainer"},
            "ckpt_path": None,
            "mode": "test",
        }
    )


@pytest.fixture
def stubbed_eval_entrypoint() -> Iterator[MagicMock]:
    """Stub ``evaluate``'s heavy collaborators and spy on ``log_wandb_provenance``.

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
        patch("synth_setter.cli.eval.hydra.utils.instantiate", return_value=trainer),
        patch("synth_setter.cli.eval.instantiate_callbacks", return_value=[]),
        patch("synth_setter.cli.eval.instantiate_loggers", return_value=[object()]),
        patch("synth_setter.cli.eval.log_hyperparameters"),
        patch("synth_setter.cli.eval.log_wandb_provenance") as provenance_spy,
    ):
        yield provenance_spy


class TestEvalProvenanceWiring:
    """The eval entrypoint pins the run identity and stamps provenance."""

    def test_pins_run_id_in_config_id_timestamp_convention(
        self, stubbed_eval_entrypoint: MagicMock
    ) -> None:
        """``cfg.logger.wandb.id`` is pinned as ``{task_name}-{timestamp}``.

        :param stubbed_eval_entrypoint: Collaborator-stub fixture.
        """
        cfg = _wandb_logger_cfg()

        evaluate(cfg)

        assert re.fullmatch(_RUN_ID_PATTERN.format(config_id="flow_simple"), cfg.logger.wandb.id)

    def test_pins_evaluation_job_type(self, stubbed_eval_entrypoint: MagicMock) -> None:
        """``cfg.logger.wandb.job_type`` is pinned to ``evaluation``.

        :param stubbed_eval_entrypoint: Collaborator-stub fixture.
        """
        cfg = _wandb_logger_cfg()

        evaluate(cfg)

        assert cfg.logger.wandb.job_type == "evaluation"

    def test_stamps_wandb_provenance_once(self, stubbed_eval_entrypoint: MagicMock) -> None:
        """``log_wandb_provenance`` is invoked exactly once when a logger exists.

        :param stubbed_eval_entrypoint: Collaborator-stub fixture exposing the spy.
        """
        evaluate(_wandb_logger_cfg())

        stubbed_eval_entrypoint.assert_called_once_with()
