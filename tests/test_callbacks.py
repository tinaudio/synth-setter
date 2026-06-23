"""Tests for ``synth_setter.utils.callbacks._log_figure`` logger dispatch.

Exercises the real ``_log_figure`` routing against lightweight logger
stand-ins that subclass the production ``WandbLogger`` / ``TensorBoardLogger``
(so the ``isinstance`` dispatch fires) but record calls instead of touching any
backend — no W&B auth prompt, no TensorBoard file writes. Only the leaf logger
backends are faked; the production routing/rank-gating/argument wiring runs for
real.

Integration tests (marked ``slow``) spin up a real Lightning ``Trainer`` with a
shrunk ``KSinFlowMatchingModule`` and verify that ``PlotLossPerTimestep`` writes
artifacts to the logger's offline storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_module
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from matplotlib.figure import Figure
from omegaconf import open_dict

from synth_setter.utils.callbacks import PlotLossPerTimestep, _log_figure


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


# ---------------------------------------------------------------------------
# Integration test — real trainer, offline W&B
# ---------------------------------------------------------------------------


def _build_ksin_module_and_datamodule():
    """Instantiate a tiny ``KSinFlowMatchingModule`` + ``KSinDataModule`` via Hydra.

    Uses the same ``datamodule=ksin, model=ffn`` overrides as the ``cfg_train``
    fixture so the encoder/vector-field shapes are consistent with the data.

    :returns: ``(pl_module, datamodule)`` ready for ``Trainer.fit``.
    """
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["datamodule=ksin", "model=flow", "trainer=cpu"],
        )

    with open_dict(cfg):
        cfg.datamodule.train_val_test_sizes = [2, 2, 2]
        cfg.datamodule.batch_size = 2
        cfg.datamodule.num_workers = 0
        cfg.datamodule.break_symmetry = True
        cfg.model.compile = False

    import hydra

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    pl_module = hydra.utils.instantiate(cfg.model)
    return pl_module, datamodule


@pytest.mark.slow
def test_plot_loss_per_timestep_logs_image_key_to_wandb_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PlotLossPerTimestep`` writes a ``plot`` image into the offline W&B run directory.

    Runs a full Lightning ``Trainer`` loop (``fast_dev_run=True``) with a real but
    tiny ``KSinFlowMatchingModule`` and a ``WandbLogger`` in offline mode.  After the
    run, asserts that:

    - W&B created at least one PNG file whose name starts with ``plot_``, proving
      that ``_log_figure`` reached the ``WandbLogger`` branch and ``log_image`` was
      called with the correct key.
    - The PNG file is non-empty (the figure was actually rendered and encoded).

    :param tmp_path: Pytest-provided temporary directory for the offline W&B run.
    :param monkeypatch: Used to set ``WANDB_MODE=offline`` for the duration of the test.
    """
    monkeypatch.setenv("WANDB_MODE", "offline")
    # Prevent W&B from touching the home-directory default wandb/ folder.
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    pl_module, datamodule = _build_ksin_module_and_datamodule()

    wandb_logger = WandbLogger(
        project="synth-setter-test",
        save_dir=str(tmp_path),
        offline=True,
    )

    callback = PlotLossPerTimestep(num_timesteps=3)

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[callback],
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    trainer.fit(pl_module, datamodule=datamodule)

    # W&B offline runs write images under:
    #   <save_dir>/wandb/offline-run-<timestamp>-<id>/files/media/images/
    # The filename is ``<key>_<idx>_<hash>.png`` — the key prefix is the
    # assertion target, not the full name (which includes a content hash).
    images_dir = next((tmp_path / "wandb").glob("offline-run-*/files/media/images"))
    plot_pngs = list(images_dir.glob("plot_*.png"))

    assert plot_pngs, (
        f"No 'plot_*.png' found in {images_dir}; "
        "expected PlotLossPerTimestep to log one image via WandbLogger.log_image"
    )
    assert all(p.stat().st_size > 0 for p in plot_pngs), (
        "At least one 'plot_*.png' is empty — the figure was not rendered"
    )
