"""Lightning module for feed-forward VST parameter prediction."""

from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from synth_setter.conditioning import (
    Conditioning,
    conditioning_batch_key,
    resolve_embedding_conditioning,
)


class VSTFeedForwardModule(LightningModule):
    """Feed-forward LightningModule that regresses VST parameters from audio features."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        warmup_steps: int = 0,
        conditioning: Conditioning = "mel",
        encoder: torch.nn.Module | None = None,
        encoder_num_heads: int | None = None,
        encoder_output_dim: int | None = None,
    ):
        """Wire the regression net and persist the optimizer/scheduler hyperparameters.

        :param net: Feature extractor mapping ``mel`` to predicted parameters.
        :param optimizer: ``functools.partial``-style optimizer factory (Hydra
            ``_partial_: true``); invoked in :meth:`configure_optimizers`.
        :param scheduler: ``functools.partial``-style scheduler factory or ``None``.
        :param compile: Whether to ``torch.compile`` the net during fit setup.
        :param warmup_steps: If positive, wrap the scheduler with a linear warmup.
        :param conditioning: Legacy mel/m2l mode or a fixed-shape embedding spec.
        :param encoder: Profile-selected embedding encoder. When configured, it replaces
            the legacy mel network for cached conditioning.
        :param encoder_num_heads: Model-owned attention head count for sequence encoders.
        :param encoder_output_dim: Configured cached-encoder output width.
        :raises ValueError: If cached conditioning has no encoder.
        """
        super().__init__()

        self._embedding_conditioning = resolve_embedding_conditioning(conditioning)
        self._conditioning_key = conditioning_batch_key(conditioning)
        if self._embedding_conditioning is not None:
            if encoder is None:
                raise ValueError("cached conditioning requires an encoder")
            net = encoder

        self.save_hyperparameters(logger=False)

        self.net = net

    def on_train_start(self):
        pass

    def _get_conditioning_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch[self._conditioning_key]

    def model_step(self, batch: dict[str, torch.Tensor]):
        target_params = batch["params"]
        conditioning = self._get_conditioning_from_batch(batch)

        pred_params = self.net(conditioning)
        if self._embedding_conditioning is not None and pred_params.shape != target_params.shape:
            raise ValueError(
                f"cached conditioning encoder output shape {tuple(pred_params.shape)} "
                f"does not match target shape {tuple(target_params.shape)}"
            )
        loss = torch.nn.functional.mse_loss(pred_params, target_params)
        return loss, pred_params, target_params, conditioning

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss, *_ = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss, preds, targets, *_ = self.model_step(batch)
        per_param_mse = (preds - targets).square().mean(dim=0)
        param_mse = per_param_mse.mean()
        self.log("val/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"param_mse": param_mse, "per_param_mse": per_param_mse, "preds": preds}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss, preds, targets, *_ = self.model_step(batch)
        per_param_mse = (preds - targets).square().mean(dim=0)
        param_mse = per_param_mse.mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"param_mse": param_mse, "per_param_mse": per_param_mse}

    def on_test_epoch_end(self) -> None:
        pass

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        preds = self.net(conditioning)
        return (
            preds,
            batch,
        )

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net.compile()

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
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


# Deprecated alias: archived W&B run configs and external job scripts resolve the
# old ``_target_`` path.
SurgeFeedForwardModule = VSTFeedForwardModule
