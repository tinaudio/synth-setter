from typing import Any

import torch
from lightning import LightningModule

from src.metrics import ChamferDistance, LinearAssignmentDistance, LogSpectralDistance
from src.models.components.loss import ChamferLoss


class KSinFeedForwardModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss_fn: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        params_per_token: int = 3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

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
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_lsd.reset()
        self.val_chamfer.reset()

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y, *_ = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y, x

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets, inputs = self.model_step(batch)

        *_, synth_fn = batch
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets, inputs = self.model_step(batch)

        # update and log metrics
        *_, synth_fn = batch
        self.val_lsd(preds, inputs, synth_fn)
        self.val_chamfer(preds, targets)

        self.log("val/lsd", self.val_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/chamfer", self.val_chamfer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
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

    def on_test_epoch_end(self) -> None:
        # TODO: implement metrics
        # self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        # etc...
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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


if __name__ == "__main__":
    _ = KSinFeedForwardModule(None, None, None, None, None)
