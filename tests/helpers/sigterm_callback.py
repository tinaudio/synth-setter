"""Lightning boundary probes for SIGTERM subprocess tests."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

_EXCEPTION_MARKER_ENV = "SYNTH_SETTER_TEST_EXCEPTION_MARKER"
_LOGGER_MARKER_ENV = "SYNTH_SETTER_TEST_LOGGER_MARKER"
_READY_FIFO_ENV = "SYNTH_SETTER_TEST_READY_FIFO"


class SignalLifecycleCallback(Callback):
    """Signal when fit starts and record Lightning's interruption hook."""

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Notify the parent process after Lightning installs its signal handlers.

        :param trainer: Active Lightning trainer.
        :param pl_module: Module entering the fit loop.
        """
        with Path(os.environ[_READY_FIFO_ENV]).open("wb", buffering=0) as ready_fifo:
            ready_fifo.write(b"1")

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        """Record that Lightning dispatched its graceful interruption hook.

        :param trainer: Interrupted Lightning trainer.
        :param pl_module: Module interrupted during fit.
        :param exception: Exception raised after SIGTERM.
        """
        Path(os.environ[_EXCEPTION_MARKER_ENV]).write_text(type(exception).__name__)


class SignalFinalizeLogger(Logger):
    """Record the status Lightning passes to logger finalization."""

    def __init__(self, **_kwargs: object) -> None:
        r"""Accept the CSV logger fields forwarded by Hydra.

        :param \*\*_kwargs: Unused CSV logger configuration.
        """
        super().__init__()

    @property
    def name(self) -> str:
        """Return the test logger name.

        :returns: Stable logger name.
        """
        return "sigterm"

    @property
    def version(self) -> str:
        """Return the test logger version.

        :returns: Stable logger version.
        """
        return "test"

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Accept hyperparameters without external I/O.

        :param params: Training hyperparameters.
        """

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Accept metrics without external I/O.

        :param metrics: Metrics emitted by Lightning.
        :param step: Optional trainer step.
        """

    def finalize(self, status: str) -> None:
        """Record Lightning's terminal logger status.

        :param status: Lightning completion status.
        """
        Path(os.environ[_LOGGER_MARKER_ENV]).write_text(status)
