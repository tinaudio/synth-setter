from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from src.models.components.vae import compute_flowvae_loss


class SurgeFlowVAEModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        warmup_steps: int = 15_000,
        beta_max: float = 0.2,
        beta_start: float = 0.1,
        beta_warmup_steps: int = 60_000,
        param_spec: str = "surge_xt",
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

        vae_out = self.net(mel_spec)
        losses = compute_flowvae_loss(vae_out, mel_spec, target_params, self.hparams.param_spec)

        return losses, mel_spec, target_params, vae_out

    def get_beta(self) -> float:
        step = self.global_step
        if step > self.hparams.beta_warmup_steps:
            return self.hparams.beta_max

        return self.hparams.beta_start + (self.global_step / self.hparams.beta_warmup_steps) * (
            self.hparams.beta_max - self.hparams.beta_start
        )

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        losses, *_, vae_out = self.model_step(batch)
        x_hat = vae_out.x_hat

        self.log("train/param_mean", x_hat.mean(), on_step=True, on_epoch=True)
        self.log("train/param_std", x_hat.std(), on_step=True, on_epoch=True)

        beta = self.get_beta()
        loss = losses["reconstruction_loss"] + beta * losses["latent_loss"] + losses["param_loss"]

        losses_to_log = {f"train/{k}": v for k, v in losses.items()}
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(losses_to_log, on_step=True, on_epoch=True)
        self.log("train/beta", beta, on_step=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        losses, *_, vae_out = self.model_step(batch)
        x_hat = vae_out.x_hat

        self.log("val/param_mean", x_hat.mean(), on_step=False, on_epoch=True)
        self.log("val/param_std", x_hat.std(), on_step=False, on_epoch=True)

        losses = {f"val/{k}": v for k, v in losses.items()}
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        losses, *_ = self.model_step(batch)
        losses = {f"test/{k}": v for k, v in losses.items()}
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        mel_spec = batch["mel_spec"]
        out = self.net(mel_spec)

        return (
            out.x_hat,
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
                optimizer, 0.1, 1.0, self.hparams.warmup_steps
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
    pass
