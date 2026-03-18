import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import ot as pot
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from scipy.optimize import linear_sum_assignment

from src.metrics import (
    ChamferDistance,
    LinearAssignmentDistance,
    LogSpectralDistance,
)
from src.utils.math import divmod


def late_curve(x, a):
    if a == 0.0:
        return x
    return (1 - torch.exp(-a * x)) / (1 - math.exp(-a))


def cosine_curve(x):
    return 0.5 + 0.5 * torch.cos(torch.pi * (1 + x))


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


class KSinFlowMatchingModule(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        vector_field: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cfg_dropout_rate: float = 0.1,
        p_time: Literal["uniform", "bias_later", "lognormal", "beta"] = "uniform",
        w_time: Literal["none", "flatten", "reverse"] = "none",
        coupling: Literal["uniform", "ot", "eot", "kabsch", "procrustes"] = "none",
        oversample_ot: float = 1.0,
        probability_path: Literal["rectified", "cfm", "fm"] = "rectified",
        cfm_sigma: float = 1e-5,
        fm_sigma: float = 1e-5,
        rectified_sigma_min: float = 0.0,
        sample_schedule: Literal["linear", "cosine", "late"] = "uniform",
        late_sample_schedule_curve: float = 2.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 4.0,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 4.0,
        sinkhorn_reg: float = 0.05,
        sinkhorn_thresh: float = 1e-6,
        ot_replace: bool = True,
        freeze_for_first_n_steps: int = 0,
        compile: bool = False,
        params_per_token: int = 3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.vector_field = vector_field

        self.val_lsd = LogSpectralDistance()
        self.val_chamfer = ChamferDistance(params_per_token)
        # self.val_lad = LinearAssignmentDistance()

        self.test_lsd = LogSpectralDistance()
        self.test_chamfer = ChamferDistance(params_per_token)
        self.test_lad = LinearAssignmentDistance(params_per_token)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.vector_field(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_lsd.reset()
        self.val_chamfer.reset()
        # self.val_lad.reset()

    def _sample_time(self, n: int, device: torch.device) -> torch.Tensor:
        if self.hparams.p_time == "uniform":
            return torch.rand(n, 1, device=device)
        elif self.hparams.p_time == "bias_later":
            t = torch.rand(n, 1, device=device)
            return late_curve(t, 1.0)
        elif self.hparams.p_time == "lognormal":
            return torch.randn(n, 1, device=device).sigmoid()
        elif self.hparams.p_time == "beta":
            dist = torch.distributions.Beta(2.5, 1.0)
            return dist.sample((n, 1)).to(device)
        elif self.hparams.p_time == "extreme_beta":
            dist = torch.distributions.Beta(10, 1.5)
            return dist.sample((n, 1)).to(device)

    def _weight_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.hparams.w_time == "none":
            return torch.ones_like(t)
        elif self.hparams.w_time == "flatten":
            half_snr = torch.log(t / (1 - t))
            weighting = torch.exp(-half_snr) + 1
            inv_weighting = weighting.pow(-2)
            return inv_weighting
        elif self.hparams.w_time == "reverse":
            half_snr = torch.log(t / (1 - t))
            weighting = torch.exp(-half_snr) + 1
            inv_weighting = weighting.pow(-4)
            return inv_weighting

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

    def _eot_sample(self, params: torch.Tensor, z: torch.Tensor):
        x0, x1 = self._basic_sample(params)

        batch_size = z.shape[0]
        a = pot.unif(x0.shape[0], type_as=x0)
        b = pot.unif(x1.shape[0], type_as=x1)
        costs = torch.cdist(x0, x1).square()

        # cost should be invariant to dimension
        costs = costs / x0.shape[-1]

        ot_map = pot.sinkhorn(
            a,
            b,
            costs,
            self.hparams.sinkhorn_reg,
            method="sinkhorn",
            numItermax=1000,
            stopThr=self.hparams.sinkhorn_thresh,
        )
        # # ot_map = pot.emd(a, b, costs, numThreads=4)
        pi = ot_map.flatten()
        samples = torch.multinomial(pi, batch_size, replacement=self.hparams.ot_replace)

        i, j = divmod(samples, batch_size)

        x0 = x0[i]
        x1 = x1[j]
        z = z[j]

        return x0, x1, z

    @torch.no_grad
    def _ot_sample(self, params: torch.Tensor, z: torch.Tensor):
        x0, x1 = self._basic_sample(params, oversample=self.hparams.oversample_ot)
        costs = torch.cdist(x0, x1)
        costs = costs.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(costs)
        row_ind = torch.from_numpy(row_ind).cuda()
        col_ind = torch.from_numpy(col_ind).cuda()

        x0 = x0[row_ind]
        x1 = x1[col_ind]
        z = z[col_ind]

        return x0, x1, z

    @torch.no_grad
    def _kabsch_sample(self, params: torch.Tensor, z: torch.Tensor):
        x0, x1 = self._basic_sample(params)
        H = x0.transpose(-1, -2) @ x1
        U, _, V = torch.svd(H)
        d = torch.linalg.det(U) * torch.linalg.det(V)
        S = torch.eye(U.shape[-1], device=U.device)
        S[..., -1, -1] = d
        R = U @ S @ V.transpose(-1, -2)

        x0 = x0 @ R.transpose(-1, -2)

        return x0, x1, z

    @torch.no_grad
    def _procrustes_sample(self, params: torch.Tensor, z: torch.Tensor):
        x0, x1 = self._basic_sample(params)
        H = x0.transpose(-1, -2) @ x1
        U, _, V = torch.svd(H)
        R = U @ V.transpose(-1, -2)
        x0 = x0 @ R.transpose(-1, -2)

        return x0, x1, z

    @torch.no_grad
    def _row_procrustes_sample(self, params: torch.Tensor, z: torch.Tensor):
        x0, x1 = self._basic_sample(params)
        H = x0 @ x1.transpose(-1, -2)
        U, _, V = torch.svd(H)
        R = U @ V.transpose(-1, -2)
        x0 = R @ x0

        return x0, x1, z

    def _sample_x0_and_x1(self, params: torch.Tensor, z: torch.Tensor):
        """Applies coupling according to the schemes in:
        https://proceedings.mlr.press/v202/pooladian23a/pooladian23a.pdf#page=5.59
        """
        if self.hparams.coupling == "uniform":
            x0, x1 = self._basic_sample(params)
            return x0, x1, z
        elif self.hparams.coupling == "ot":
            return self._ot_sample(params, z)
        elif self.hparams.coupling == "eot":
            return self._eot_sample(params, z)
        elif self.hparams.coupling == "kabsch":
            return self._kabsch_sample(params, z)
        elif self.hparams.coupling == "procrustes":
            return self._procrustes_sample(params, z)
        elif self.hparams.coupling == "row_procrustes":
            return self._row_procrustes_sample(params, z)
        else:
            raise NotImplementedError(f"Unknown coupling {self.hparams.coupling}")

    def _rectified_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        x_t = x0 * (1 - t) * (1 - self.hparams.rectified_sigma_min) + x1 * t

        return x_t

    def _cfm_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        mu_t = x0 * (1 - t) + x1 * t
        sigma_t = self.hparams.cfm_sigma * torch.randn_like(mu_t)
        x_t = mu_t + sigma_t

        return x_t

    def _fm_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        mu_t = x1 * t
        sigma_t = t * self.hparams.fm_sigma - t + 1
        sigma_t = sigma_t.sqrt() * torch.randn_like(mu_t)
        x_t = mu_t + sigma_t

        return x_t

    def _sample_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        if self.hparams.probability_path == "rectified":
            x_t = self._rectified_probability_path(x0, x1, t)
        elif self.hparams.probability_path == "cfm":
            x_t = self._cfm_probability_path(x0, x1, t)
        elif self.hparams.probability_path == "fm":
            x_t = self._fm_probability_path(x0, x1, t)
        else:
            raise NotImplementedError(f"Unknown probability path {self.hparams.probability_path}")

        return x_t

    def _rectified_vector_field(self, x0: torch.Tensor, x1: torch.Tensor):
        return x1 - x0

    def _cfm_vector_field(self, x0: torch.Tensor, x1: torch.Tensor):
        return x1 - x0

    def _fm_vector_field(self, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        numer = x1 - (1 - self.hparams.fm_sigma) * x_t
        denom = 1 - (1 - self.hparams.fm_sigma) * t

        return numer / denom

    def _evaluate_target_field(
        self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        if self.hparams.probability_path == "rectified":
            target = self._rectified_vector_field(x0, x1)
        elif self.hparams.probability_path == "cfm":
            target = self._cfm_vector_field(x0, x1)
        elif self.hparams.probability_path == "fm":
            target = self._fm_vector_field(x1, x_t, t)
        else:
            raise NotImplementedError(f"Unknown probability path {self.hparams.probability_path}")

        return target

    def _train_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        signal, params, noise, _ = batch

        # Get conditioning vector
        conditioning = self.encoder(signal)
        z = self.vector_field.apply_dropout(conditioning, self.hparams.cfg_dropout_rate)

        with torch.no_grad():
            # Sample time-steps
            t = self._sample_time(signal.shape[0], signal.device)
            w = self._weight_time(t)

            # x0, x1, z = self._sample_x0_and_x1(params, z)
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
        if self.global_step < self.hparams.freeze_for_first_n_steps:
            # freeze vector_field and encoder, leaving only projection active
            for param in self.vector_field.parameters():
                param.requires_grad = False

            for param in self.vector_field.projection.parameters():
                param.requires_grad = True

            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # unfreeze vector_field and encoder
            for param in self.vector_field.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = True

        loss, penalty = self._train_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if penalty is not None:
            self.log("train/penalty", penalty, on_step=True, on_epoch=True, prog_bar=True)

        return loss + penalty

    def on_train_epoch_end(self) -> None:
        pass

    def _warp_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.hparams.sample_schedule == "linear":
            return t
        elif self.hparams.sample_schedule == "cosine":
            return cosine_curve(t)
        elif self.hparams.sample_schedule == "late":
            return late_curve(t, self.hparams.late_sample_schedule_curve)
        else:
            raise NotImplementedError

    def _sample(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        steps: int,
        cfg_strength: float,
    ):
        x, y, sample, _ = batch

        # sample = torch.randn_like(y)
        conditioning = self.encoder(x)
        t = torch.zeros(sample.shape[0], 1, device=sample.device)
        dt = 1.0 / steps

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

        return sample, y, x

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds, targets, inputs = self._sample(
            batch,
            self.hparams.validation_sample_steps,
            self.hparams.validation_cfg_strength,
        )

        *_, synth_fn = batch
        # update and log metrics
        self.val_lsd(preds, inputs, synth_fn)
        self.val_chamfer(preds, targets)
        # self.val_lad(preds, targets)

        self.log("val/lsd", self.val_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/chamfer", self.val_chamfer, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/lad", self.val_lad, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds, targets, inputs = self._sample(
            batch, self.hparams.test_sample_steps, self.hparams.test_cfg_strength
        )

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
        self.log("test/lad", self.test_lad, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        # TODO: implement metrics
        # self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        # etc...
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.vector_field = torch.compile(self.vector_field)
            self.encoder = torch.compile(self.encoder)

    def on_before_optimizer_step(self, optimizer) -> None:
        encoder_norms = grad_norm(self.encoder, 2.0)
        vf_norms = grad_norm(self.vector_field, 2.0)

        encoder_norms = {f"encoder/{k}": v for k, v in encoder_norms.items()}
        vf_norms = {f"vector_field/{k}": v for k, v in vf_norms.items()}

        self.log_dict(encoder_norms, on_step=True, on_epoch=True)
        self.log_dict(vf_norms, on_step=True, on_epoch=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/chamfer",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = KSinFlowMatchingModule(None, None, None, None)
