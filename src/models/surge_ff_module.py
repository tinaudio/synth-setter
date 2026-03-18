from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm


class SurgeFeedForwardModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        warmup_steps: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(self, batch: dict[str, torch.Tensor]):
        target_params = batch["params"]
        mel_spec = batch["mel_spec"]

        pred_params = self.net(mel_spec)
        loss = torch.nn.functional.mse_loss(pred_params, target_params)
        return loss, pred_params, target_params, mel_spec

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, *_ = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets, *_ = self.model_step(batch)
        per_param_mse = (preds - targets).square().mean(dim=0)
        param_mse = per_param_mse.mean()
        self.log("val/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"param_mse": param_mse, "per_param_mse": per_param_mse}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets, *_ = self.model_step(batch)
        per_param_mse = (preds - targets).square().mean(dim=0)
        param_mse = per_param_mse.mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"param_mse": param_mse, "per_param_mse": per_param_mse}

    def on_test_epoch_end(self) -> None:
        pass

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        mel_spec = batch["mel_spec"]
        preds = self.net(mel_spec)
        return (
            preds,
            batch,
        )

    def setup(self, stage: str) -> None:
        if not self.hparams.compile:
            return

        self.net = torch.compile(self.net)

    def on_before_optimizer_step(self, optimizer) -> None:
        norms = grad_norm(self.net, 2.0)
        norms = {f"net/{k}": v for k, v in norms.items()}
        self.log_dict(norms, on_step=True, on_epoch=False)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1e-10, 1.0, self.hparams.warmup_steps
            )
        else:
            warmup_scheduler = None

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
        else:
            scheduler = None

        if warmup_scheduler is not None and scheduler is None:
            scheduler = warmup_scheduler
        elif warmup_scheduler is not None and scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.hparams.warmup_steps],
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/chamfer",
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SurgeFlowMatchingModule(None, None, None, None)
