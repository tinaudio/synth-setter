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
from pathlib import Path
from typing import cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from matplotlib.figure import Figure
from torchsynth.signal import Signal

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.utils.callbacks import LogPerParamMSE, PredictionWriter, _log_figure


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


def test_log_per_param_mse_without_param_spec_raises_type_error() -> None:
    """Per-parameter metric labels require callers to select a ParamSpec."""
    with pytest.raises(TypeError, match="param_spec"):
        LogPerParamMSE()  # type: ignore[call-arg]


def test_prediction_writer_serializes_real_torchsynth_signals_as_plain_tensors(
    tmp_path: Path,
) -> None:
    """Prediction, audio, and parameter artifacts load safely as exact base tensors.

    :param tmp_path: Pytest-provided directory for callback artifacts.
    """
    prediction = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3).as_subclass(Signal).requires_grad_()
    )
    audio = torch.arange(8, dtype=torch.float32).reshape(2, 4).as_subclass(Signal).requires_grad_()
    params = (
        torch.arange(10, dtype=torch.float32).reshape(2, 5).as_subclass(Signal).requires_grad_()
    )
    writer = PredictionWriter(tmp_path, write_interval="batch")

    writer.write_on_batch_end(
        cast("Trainer", None),
        cast("LightningModule", None),
        (prediction, {"audio": audio, "params": params}),
        None,
        None,
        0,
        0,
    )

    loaded_prediction = torch.load(tmp_path / "pred-0.pt", weights_only=True)
    loaded_audio = torch.load(tmp_path / "target-audio-0.pt", weights_only=True)
    loaded_params = torch.load(tmp_path / "target-params-0.pt", weights_only=True)
    assert type(loaded_prediction) is torch.Tensor
    assert type(loaded_audio) is torch.Tensor
    assert type(loaded_params) is torch.Tensor
    assert not loaded_prediction.requires_grad
    assert not loaded_audio.requires_grad
    assert not loaded_params.requires_grad
    assert (
        loaded_prediction.device.type
        == loaded_audio.device.type
        == loaded_params.device.type
        == "cpu"
    )
    assert torch.equal(loaded_prediction, prediction)
    assert torch.equal(loaded_audio, audio)
    assert torch.equal(loaded_params, params)


def test_prediction_writer_epoch_serializes_torchsynth_signals_as_plain_tensors(
    tmp_path: Path,
) -> None:
    """Epoch artifacts from Signal values reload through the weights-only boundary.

    :param tmp_path: Pytest-provided directory for callback artifacts.
    """
    prediction = torch.arange(6, dtype=torch.float32).reshape(2, 3).as_subclass(Signal)
    audio = torch.arange(8, dtype=torch.float32).reshape(2, 4).as_subclass(Signal)
    params = torch.arange(10, dtype=torch.float32).reshape(2, 5).as_subclass(Signal)
    writer = PredictionWriter(tmp_path, write_interval="epoch")

    writer.write_on_epoch_end(
        cast("Trainer", None),
        cast("LightningModule", None),
        (prediction, {"audio": audio, "params": params}),
        [],
    )

    loaded_prediction = torch.load(tmp_path / "predictions.pt", weights_only=True)
    loaded_audio = torch.load(tmp_path / "target-audio.pt", weights_only=True)
    loaded_params = torch.load(tmp_path / "target-params.pt", weights_only=True)
    assert type(loaded_prediction) is torch.Tensor
    assert type(loaded_audio) is torch.Tensor
    assert type(loaded_params) is torch.Tensor
    assert torch.equal(loaded_prediction, prediction)
    assert torch.equal(loaded_audio, audio)
    assert torch.equal(loaded_params, params)


class _RecordingModule:
    """Stand-in for the LightningModule, capturing the metric dict the callback emits."""

    def __init__(self) -> None:
        self.logged: dict[str, float] = {}

    def log_dict(self, metrics: dict[str, float]) -> None:
        """Record the per-parameter metrics the callback dispatches.

        :param metrics: Metric name to value mapping emitted by the callback.
        """
        self.logged.update(metrics)


def _run_validation_epoch(param_spec: str, per_param_mse: torch.Tensor) -> dict[str, float]:
    """Drive one real validation epoch of ``LogPerParamMSE`` and return what it logged.

    :param param_spec: Registered ParamSpec name selecting the metric labels.
    :param per_param_mse: One per-encoded-column MSE vector, as the modules emit it.
    :returns: The metric mapping the callback passed to ``log_dict``.
    """
    callback = LogPerParamMSE(param_spec)
    module = _RecordingModule()
    trainer = cast("Trainer", None)
    pl_module = cast("LightningModule", module)
    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(trainer, pl_module, {"per_param_mse": per_param_mse}, None, 0)
    callback.on_validation_epoch_end(trainer, pl_module)
    return module.logged


@pytest.mark.parametrize("param_spec", ["surge_4", "surge_simple", "surge_xt", "obxf"])
def test_log_per_param_mse_emits_one_metric_per_parameter_name(param_spec: str) -> None:
    """Every ParamSpec parameter gets a metric, regardless of how many columns it spans.

    ``names`` is one entry per ``Parameter`` while the module's ``per_param_mse``
    is one entry per encoded column; the two never match (``note_start_and_end``
    alone spans two columns), so positional zipping silently drops the tail.

    :param param_spec: Registered ParamSpec name under test.
    """
    spec = param_specs[param_spec]
    per_param_mse = torch.arange(spec.encoded_width, dtype=torch.float32)

    logged = _run_validation_epoch(param_spec, per_param_mse)

    assert sorted(logged) == sorted(f"per_param_mse/{name}" for name in spec.names)


def test_log_per_param_mse_emits_optional_best_swap_metrics() -> None:
    """Best-swap vectors use a separate per-parameter namespace when present."""
    spec = param_specs["surge_4"]
    callback = LogPerParamMSE("surge_4")
    module = _RecordingModule()
    trainer = cast("Trainer", None)
    pl_module = cast("LightningModule", module)
    outputs = {
        "per_param_mse": torch.zeros(spec.encoded_width),
        "per_param_mse_best_swap": torch.arange(spec.encoded_width, dtype=torch.float32),
    }

    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(trainer, pl_module, outputs, None, 0)
    callback.on_validation_epoch_end(trainer, pl_module)

    assert module.logged["per_param_mse_best_swap/note_start_and_end"] == pytest.approx(5.5)


def test_log_per_param_mse_averages_a_multi_column_parameter_over_its_span() -> None:
    """A parameter spanning several columns reports the mean of those columns.

    ``note_start_and_end`` occupies the final two columns of every spec, so under
    positional zipping it would take the value of an earlier column instead.
    """
    spec = param_specs["surge_4"]
    # Columns 0..6; note_start_and_end owns the last two, whose mean is 5.5.
    per_param_mse = torch.arange(spec.encoded_width, dtype=torch.float32)

    logged = _run_validation_epoch("surge_4", per_param_mse)

    assert logged["per_param_mse/note_start_and_end"] == pytest.approx(5.5)


def test_log_per_param_mse_labels_parameters_after_a_onehot_correctly() -> None:
    """Labels stay aligned past a multi-column parameter rather than shifting by one.

    ``surge_xt`` has 32 onehot parameters across 300 columns but only 164 names,
    so positional zipping misnames every parameter after the first onehot.
    """
    spec = param_specs["surge_xt"]
    spans = dict((param.name, sl) for param, sl in spec.encoded_slices())
    per_param_mse = torch.arange(spec.encoded_width, dtype=torch.float32)

    logged = _run_validation_epoch("surge_xt", per_param_mse)

    pitch_span = spans["pitch"]
    assert logged["per_param_mse/pitch"] == pytest.approx(float(pitch_span.start))


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
