"""Lightning callback recording parameter targets consumed during training."""

from typing import ClassVar

import torch
from lightning import Callback, LightningModule, Trainer


class ParamCaptureCallback(Callback):
    """Retain detached parameter targets from every completed training batch.

    .. attribute :: captured

        Parameter batches captured from the current test run.
    """

    captured: ClassVar[list[torch.Tensor]] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: dict[str, torch.Tensor | None],
        batch_idx: int,
    ) -> None:
        """Record the parameter targets consumed by the training step.

        :param trainer: Unused; Lightning hook signature.
        :param pl_module: Unused; Lightning hook signature.
        :param outputs: Unused; Lightning hook signature.
        :param batch: Training batch whose parameter targets are captured.
        :param batch_idx: Unused; Lightning hook signature.
        """
        params = batch["params"]
        assert params is not None
        ParamCaptureCallback.captured.append(params.detach().clone().cpu())
