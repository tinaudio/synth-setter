"""Tests for ``synth_setter.utils.callbacks._log_figure`` logger dispatch.

Exercises the real ``_log_figure`` routing against lightweight logger
stand-ins that subclass the production ``WandbLogger`` / ``TensorBoardLogger``
(so the ``isinstance`` dispatch fires) but record calls instead of touching any
backend — no W&B auth prompt, no TensorBoard file writes. Only the leaf logger
backends are faked; the production routing/rank-gating/argument wiring runs for
real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from matplotlib.figure import Figure

from synth_setter.utils.callbacks import _log_figure


class _RecordingWandbLogger(WandbLogger):
    """``WandbLogger`` that records ``log_image`` calls without a W&B backend.

    Subclasses the production class so ``_log_figure``'s ``isinstance`` branch
    selects it, but bypasses ``__init__`` so no run is started.
    """

    def __init__(self) -> None:
        self.image_calls: list[dict[str, object]] = []

    def log_image(self, key: str, images: list[object], step: int) -> None:
        """Record the keyword-routed image payload the callback dispatches.

        :param key: Log key the callback routed the figure under.
        :param images: Single-element list holding the dispatched figure.
        :param step: Global step the callback tagged the image with.
        """
        self.image_calls.append({"key": key, "images": images, "step": step})


class _RecordingTensorBoardExperiment:
    """Stand-in for ``TensorBoardLogger.experiment`` recording ``add_figure``."""

    def __init__(self) -> None:
        self.figure_calls: list[dict[str, object]] = []

    def add_figure(self, tag: str, figure: object, global_step: int) -> None:
        """Record the positional/keyword payload the callback dispatches.

        :param tag: TensorBoard tag the callback routed the figure under.
        :param figure: The dispatched matplotlib figure.
        :param global_step: Global step the callback tagged the figure with.
        """
        self.figure_calls.append({"tag": tag, "figure": figure, "global_step": global_step})


class _RecordingTensorBoardLogger(TensorBoardLogger):
    """``TensorBoardLogger`` exposing a recording ``experiment``, no file writes."""

    def __init__(self) -> None:
        self._recording_experiment = _RecordingTensorBoardExperiment()

    @property
    def experiment(self) -> _RecordingTensorBoardExperiment:  # type: ignore[override]
        """Return the recorder in place of the real ``SummaryWriter``."""
        return self._recording_experiment


class _RecordingCSVLogger(CSVLogger):
    """``CSVLogger`` stand-in; has no image API, so ``_log_figure`` must skip it.

    Any attribute access the callback makes would raise ``AttributeError`` (no
    ``log_image`` / ``experiment.add_figure``), proving the no-op path is taken.
    """

    def __init__(self) -> None:
        pass


@dataclass
class _FakeTrainer:
    """Minimal ``Trainer`` surface ``_log_figure`` reads: loggers, step, and rank.

    .. attribute :: loggers

       Loggers ``_log_figure`` iterates over for image dispatch.

    .. attribute :: global_step

       Step value the callback stamps onto each emitted figure.

    .. attribute :: is_global_zero

       Rank-0 gate; ``False`` makes ``_log_figure`` a no-op.
    """

    loggers: list[object]
    global_step: int = 42
    is_global_zero: bool = True


def _trainer(
    loggers: list[object], *, global_step: int = 42, is_global_zero: bool = True
) -> Trainer:
    """Build a ``_FakeTrainer`` cast to ``Trainer`` for ``_log_figure``'s signature.

    :param loggers: Loggers attached to the fake trainer.
    :param global_step: Step value the callback stamps onto figures.
    :param is_global_zero: Rank-0 gate; ``False`` makes ``_log_figure`` a no-op.
    :returns: The fake narrowed to ``Trainer`` for the call site's type checker.
    """
    return cast("Trainer", _FakeTrainer(loggers, global_step, is_global_zero))


def test_log_figure_routes_to_wandb_logger_when_only_wandb_logger_present():
    """A lone WandbLogger receives one ``log_image`` call with the figure and step."""
    wandb_logger = _RecordingWandbLogger()
    trainer = _trainer([wandb_logger], global_step=42)
    fig = Figure()

    _log_figure(trainer, "plot", fig)

    assert wandb_logger.image_calls == [{"key": "plot", "images": [fig], "step": 42}]


def test_log_figure_routes_to_tensorboard_logger_when_only_tensorboard_logger_present():
    """A lone TensorBoardLogger receives one ``experiment.add_figure`` call."""
    tb_logger = _RecordingTensorBoardLogger()
    trainer = _trainer([tb_logger], global_step=7)
    fig = Figure()

    _log_figure(trainer, "pos_enc_similarity", fig)

    assert tb_logger.experiment.figure_calls == [
        {"tag": "pos_enc_similarity", "figure": fig, "global_step": 7}
    ]


def test_log_figure_dispatches_to_both_when_both_loggers_present():
    """When both loggers are attached, each receives exactly one call."""
    wandb_logger = _RecordingWandbLogger()
    tb_logger = _RecordingTensorBoardLogger()
    trainer = _trainer([wandb_logger, tb_logger], global_step=3)
    fig = Figure()

    _log_figure(trainer, "assignment", fig)

    assert wandb_logger.image_calls == [{"key": "assignment", "images": [fig], "step": 3}]
    assert tb_logger.experiment.figure_calls == [
        {"tag": "assignment", "figure": fig, "global_step": 3}
    ]


def test_log_figure_is_noop_when_no_image_capable_loggers_present():
    """CSV-only setup (the default after #612) stays silent — no calls, no errors."""
    csv_logger = _RecordingCSVLogger()
    trainer = _trainer([csv_logger], global_step=5)
    fig = Figure()

    # A non-skip would touch ``log_image`` / ``experiment`` on the CSV stand-in
    # and raise ``AttributeError``; reaching this assertion proves the no-op path.
    _log_figure(trainer, "plot", fig)


def test_log_figure_is_noop_on_non_zero_rank():
    """Under DDP, only rank 0 should emit — SummaryWriter is not rank-safe."""
    wandb_logger = _RecordingWandbLogger()
    tb_logger = _RecordingTensorBoardLogger()
    trainer = _trainer([wandb_logger, tb_logger], is_global_zero=False)
    fig = Figure()

    _log_figure(trainer, "plot", fig)

    assert wandb_logger.image_calls == []
    assert tb_logger.experiment.figure_calls == []
