"""Tests for ``src.utils.callbacks._log_figure`` logger dispatch.

Uses ``MagicMock(spec=...)`` so ``isinstance`` checks against the real Lightning
logger classes pass, without instantiating any backends (no W&B auth prompt,
no TensorBoard file writes).
"""

from unittest.mock import MagicMock

from matplotlib.figure import Figure
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from src.utils.callbacks import _log_figure


def _make_trainer(
    loggers: list[object],
    global_step: int = 42,
    is_global_zero: bool = True,
) -> MagicMock:
    """Build a stand-in ``Trainer`` exposing ``loggers``, ``global_step``, and rank."""
    trainer = MagicMock()
    trainer.loggers = loggers
    trainer.global_step = global_step
    trainer.is_global_zero = is_global_zero
    return trainer


def test_log_figure_routes_to_wandb_logger_when_only_wandb_logger_present():
    """A lone WandbLogger receives one ``log_image`` call with the figure and step."""
    wandb_logger = MagicMock(spec=WandbLogger)
    trainer = _make_trainer([wandb_logger], global_step=42)
    fig = MagicMock(spec=Figure)

    _log_figure(trainer, "plot", fig)

    wandb_logger.log_image.assert_called_once_with(key="plot", images=[fig], step=42)


def test_log_figure_routes_to_tensorboard_logger_when_only_tensorboard_logger_present():
    """A lone TensorBoardLogger receives one ``experiment.add_figure`` call."""
    tb_logger = MagicMock(spec=TensorBoardLogger)
    trainer = _make_trainer([tb_logger], global_step=7)
    fig = MagicMock(spec=Figure)

    _log_figure(trainer, "pos_enc_similarity", fig)

    tb_logger.experiment.add_figure.assert_called_once_with(
        "pos_enc_similarity", fig, global_step=7
    )


def test_log_figure_dispatches_to_both_when_both_loggers_present():
    """When both loggers are attached, each receives exactly one call."""
    wandb_logger = MagicMock(spec=WandbLogger)
    tb_logger = MagicMock(spec=TensorBoardLogger)
    trainer = _make_trainer([wandb_logger, tb_logger], global_step=3)
    fig = MagicMock(spec=Figure)

    _log_figure(trainer, "assignment", fig)

    wandb_logger.log_image.assert_called_once_with(key="assignment", images=[fig], step=3)
    tb_logger.experiment.add_figure.assert_called_once_with("assignment", fig, global_step=3)


def test_log_figure_is_noop_when_no_image_capable_loggers_present():
    """CSV-only setup (the default after #612) stays silent — no calls, no errors."""
    csv_logger = MagicMock(spec=CSVLogger)
    trainer = _make_trainer([csv_logger], global_step=5)
    fig = MagicMock(spec=Figure)

    _log_figure(trainer, "plot", fig)

    # CSVLogger has no image API; ensure we didn't accidentally invoke anything.
    assert not csv_logger.method_calls


def test_log_figure_is_noop_on_non_zero_rank():
    """Under DDP, only rank 0 should emit — SummaryWriter is not rank-safe."""
    wandb_logger = MagicMock(spec=WandbLogger)
    tb_logger = MagicMock(spec=TensorBoardLogger)
    trainer = _make_trainer([wandb_logger, tb_logger], is_global_zero=False)
    fig = MagicMock(spec=Figure)

    _log_figure(trainer, "plot", fig)

    wandb_logger.log_image.assert_not_called()
    tb_logger.experiment.add_figure.assert_not_called()
