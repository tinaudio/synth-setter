from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm


def call_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    conditioning: torch.Tensor,
    cfg_strength: float,
):
    y_c = f(x, t, conditioning)
    y_u = f(x, t, None)

    return (1 - cfg_strength) * y_u + cfg_strength * y_c


def rk4_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    conditioning: torch.Tensor,
    cfg_strength: float,
):
    f = partial(call_with_cfg, f, conditioning=conditioning, cfg_strength=cfg_strength)
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class SurgeFlowMatchingModule(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        vector_field: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        conditioning: Literal["mel", "m2l"] = "mel",
        warmup_steps: int = 5000,
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 4.0,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 4.0,
        compile: bool = False,
        num_params: int = 90,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.vector_field = vector_field

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def _sample_time(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.rand(n, 1, device=device)

    def _weight_time(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def _basic_sample(self, params: torch.Tensor, oversample: float = 1.0):
        if oversample == 1.0:
            x0 = torch.randn_like(params)
        elif oversample < 1.0:
            raise ValueError(f"oversample must be >= 1.0, got {oversample}")
        else:
            n = int(oversample * params.shape[0])
            x0 = torch.randn(n, *params.shape[1:], device=params.device)
        x1 = params

        return x0, x1

    def _rectified_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        x_t = x0 * (1 - t) * (1 - self.hparams.rectified_sigma_min) + x1 * t

        return x_t

    def _sample_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        x_t = self._rectified_probability_path(x0, x1, t)
        return x_t

    def _rectified_vector_field(self, x0: torch.Tensor, x1: torch.Tensor):
        return x1 - x0

    def _evaluate_target_field(
        self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        target = self._rectified_vector_field(x0, x1)
        return target

    def _get_conditioning_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.hparams.conditioning == "mel":
            return batch["mel_spec"]
        elif self.hparams.conditioning == "m2l":
            return batch["m2l"]
        else:
            raise ValueError(f"Unknown conditioning {self.hparams.conditioning}")

    def _train_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        conditioning = self._get_conditioning_from_batch(batch)
        params = batch["params"]
        noise = batch["noise"]

        # Get conditioning vector
        conditioning = self.encoder(conditioning)
        z = self.vector_field.apply_dropout(conditioning, self.hparams.cfg_dropout_rate)

        with torch.no_grad():
            # Sample time-steps
            t = self._sample_time(params.shape[0], params.device)
            w = self._weight_time(t)

            x0 = noise
            x1 = params

            # we sample a point along the trajectory
            x_t = self._sample_probability_path(x0, x1, t)
            target = self._evaluate_target_field(x0, x1, x_t, t)

        prediction = self.vector_field(x_t, t, z)

        # compute and weight loss
        loss = (prediction - target).square().mean(dim=-1)
        loss = loss * w
        loss = loss.mean()

        penalty = None
        if hasattr(self.vector_field, "penalty"):
            penalty = self.vector_field.penalty()

        return loss, penalty

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, penalty = self._train_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if penalty is not None:
            self.log("train/penalty", penalty, on_step=True, on_epoch=True, prog_bar=True)

        return loss + penalty

    def on_train_epoch_end(self) -> None:
        pass

    def _warp_time(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _sample(
        self,
        conditioning: torch.Tensor | None,
        noise: torch.Tensor,
        steps: int,
        cfg_strength: float,
    ):
        if conditioning is not None:
            conditioning = self.encoder(conditioning)

        t = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt = 1.0 / steps

        sample = noise

        for _ in range(steps):
            warped_t = self._warp_time(t)
            warped_t_plus_dt = self._warp_time(t + dt)
            warped_dt = warped_t_plus_dt - warped_t

            sample = rk4_with_cfg(
                self.vector_field,
                sample,
                warped_t,
                warped_dt,
                conditioning,
                cfg_strength,
            )
            t = t + dt

        return sample

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.validation_sample_steps,
            self.hparams.validation_cfg_strength,
        )

        per_param_mse = (pred_params - batch["params"]).square().mean(dim=0)
        param_mse = per_param_mse.mean()
        self.log("val/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"param_mse": param_mse, "per_param_mse": per_param_mse}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.test_sample_steps,
            self.hparams.test_cfg_strength,
        )

        param_mse = (pred_params - batch["params"]).square().mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        return param_mse

    def on_test_epoch_end(self) -> None:
        pass

    def predict_step(self, batch: dict[str, Any], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        return (
            self._sample(
                conditioning,
                torch.randn(
                    conditioning.shape[0],
                    self.hparams.num_params,
                    device=conditioning.device,
                ),
                self.hparams.test_sample_steps,
                self.hparams.test_cfg_strength,
            ),
            batch,
        )

    def setup(self, stage: str) -> None:
        if not self.hparams.compile:
            return

        self.vector_field = torch.compile(self.vector_field)
        self.encoder = torch.compile(self.encoder)

    def on_before_optimizer_step(self, optimizer) -> None:
        vf_norms = grad_norm(self.vector_field, 2.0)
        encoder_norms = grad_norm(self.encoder, 2.0)

        vf_norms = {f"vector_field/{k}": v for k, v in vf_norms.items()}
        encoder_norms = {f"encoder/{k}": v for k, v in encoder_norms.items()}

        self.log_dict(vf_norms, on_step=True, on_epoch=False)
        self.log_dict(encoder_norms, on_step=True, on_epoch=False)

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
