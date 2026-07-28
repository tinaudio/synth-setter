"""Lightning module for feed-forward TorchSynth parameter prediction."""

from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule

from synth_setter.metrics import LogSpectralDistance

Renderer = Callable[[torch.Tensor], torch.Tensor]
TorchSynthBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, Renderer]
ModelStepOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class TorchSynthFeedForwardModule(LightningModule):
    """Regress TorchSynth parameters from online-rendered audio."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None,
        compile: bool,
    ) -> None:
        """Configure the TorchSynth regression model and audio-domain metric.

        :param net: Network mapping rendered audio to normalized parameters.
        :param optimizer: Optimizer factory receiving the module parameters.
        :param scheduler: Optional scheduler factory receiving the optimizer.
        :param compile: Compile ``net`` in place before fitting.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self._compile = compile
        self._optimizer_factory = optimizer
        self._scheduler_factory = scheduler
        self.criterion = torch.nn.MSELoss()
        self.val_lsd = LogSpectralDistance()
        self.test_lsd = LogSpectralDistance()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict normalized parameters from rendered audio.

        :param x: Batched audio with shape ``(batch, signal_length)``.
        :returns: Float parameter predictions with shape ``(batch, num_params)``.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Discard metric state accumulated by Lightning's sanity validation."""
        self.val_lsd.reset()

    def model_step(self, batch: TorchSynthBatch) -> ModelStepOutput:
        """Compute loss and predictions for one collated TorchSynth batch.

        :param batch: Audio, normalized params, sampling noise, and renderer.
        :returns: Scalar loss, predictions, targets, and input audio.
        """
        inputs, targets, _noise, _renderer = batch
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        return loss, predictions, targets, inputs

    def training_step(self, batch: TorchSynthBatch, _batch_idx: int) -> torch.Tensor:
        """Log and return one training loss.

        :param batch: Collated TorchSynth training batch.
        :param _batch_idx: Unused Lightning batch index.
        :returns: Scalar loss used for backpropagation.
        """
        loss, *_ = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: TorchSynthBatch, _batch_idx: int) -> None:
        """Accumulate validation loss and audio-domain distance.

        :param batch: Collated TorchSynth validation batch.
        :param _batch_idx: Unused Lightning batch index.
        """
        loss, predictions, _targets, inputs = self.model_step(batch)
        renderer = batch[-1]
        self.val_lsd(predictions, inputs, renderer)

        self.log("val/lsd", self.val_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: TorchSynthBatch, _batch_idx: int) -> None:
        """Accumulate test loss and parameter/audio distances.

        :param batch: Collated TorchSynth test batch.
        :param _batch_idx: Unused Lightning batch index.
        """
        loss, predictions, targets, inputs = self.model_step(batch)
        renderer = batch[-1]
        self.test_lsd(predictions, inputs, renderer)

        param_mse = (predictions - targets).square().mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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
