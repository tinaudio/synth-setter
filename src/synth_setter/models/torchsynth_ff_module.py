"""Lightning module for feed-forward TorchSynth parameter prediction."""

from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule

from synth_setter.metrics import ChamferDistance, LinearAssignmentDistance, LogSpectralDistance
from synth_setter.models.components.loss import ChamferLoss, MSESortLoss


class TorchSynthFeedForwardModule(LightningModule):
    """Regress TorchSynth parameters from online-rendered audio."""

    def __init__(
        self,
        net: torch.nn.Module,
        loss_fn: str,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None,
        compile: bool,
        params_per_token: int = 1,
    ):
        """Configure the TorchSynth regression model and audio-domain metrics.

        :param net: Network mapping rendered audio to normalized parameters.
        :param loss_fn: Parameter loss name: ``mse``, ``chamfer``, or ``mse_sort``.
        :param optimizer: Optimizer factory receiving the module parameters.
        :param scheduler: Optional scheduler factory receiving the optimizer.
        :param compile: Compile ``net`` in place before fitting.
        :param params_per_token: Parameter grouping width for permutation metrics.
        :raises NotImplementedError: ``loss_fn`` is unsupported.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self._compile = compile
        self._optimizer_factory = optimizer
        self._scheduler_factory = scheduler

        if loss_fn == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss_fn == "chamfer":
            self.criterion = ChamferLoss(params_per_token)
        elif loss_fn == "mse_sort":
            self.criterion = MSESortLoss(params_per_token)
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss_fn}")

        self.val_lsd = LogSpectralDistance()
        self.val_chamfer = ChamferDistance(params_per_token)

        self.test_lsd = LogSpectralDistance()
        self.test_chamfer = ChamferDistance(params_per_token)
        self.test_lad = LinearAssignmentDistance(params_per_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict normalized parameters from rendered audio.

        :param x: Batched audio.
        :returns: Batched parameter predictions.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Discard metric state accumulated by Lightning's sanity validation."""
        self.val_lsd.reset()
        self.val_chamfer.reset()

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        """Compute loss and predictions for one collated TorchSynth batch.

        :param batch: Audio, target params, optional noise, and renderer callable.
        :returns: Loss, predictions, targets, and input audio.
        """
        x, y, *_ = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y, x

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Log and return one training loss.

        :param batch: Collated TorchSynth training batch.
        :param batch_idx: Lightning batch index.
        :returns: Scalar loss used for backpropagation.
        """
        loss, *_ = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Accumulate validation loss and audio-domain metrics.

        :param batch: Collated TorchSynth validation batch.
        :param batch_idx: Lightning batch index.
        """
        loss, preds, targets, inputs = self.model_step(batch)

        *_, synth_fn = batch
        self.val_lsd(preds, inputs, synth_fn)
        self.val_chamfer(preds, targets)

        self.log("val/lsd", self.val_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/chamfer", self.val_chamfer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Accumulate test loss and parameter/audio metrics.

        :param batch: Collated TorchSynth test batch.
        :param batch_idx: Lightning batch index.
        """
        loss, preds, targets, inputs = self.model_step(batch)

        *_, synth_fn = batch
        self.test_lsd(preds, inputs, synth_fn)
        self.test_chamfer(preds, targets)
        self.test_lad(preds, targets)

        param_mse = (preds - targets).square().mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/chamfer",
            self.test_chamfer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/lad", self.test_lad, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Compile the network only for the fit stage.

        :param stage: Lightning stage name.
        """
        if self._compile and stage == "fit":
            self.net.compile()

    def configure_optimizers(self) -> dict[str, Any]:
        """Instantiate the configured optimizer and optional scheduler.

        :returns: Lightning optimizer configuration.
        """
        optimizer = self._optimizer_factory(params=self.parameters())

        if self._scheduler_factory is not None:
            scheduler = self._scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
