"""Lightning module for flow-matching VST parameter prediction."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from synth_setter.conditioning import (
    Conditioning,
    SketchControls,
    resolve_embedding_conditioning,
    resolve_sketch_controls,
    select_conditioning,
)
from synth_setter.metrics import BestSwapParamMSE, best_swap_per_param_mse
from synth_setter.models.components.sketch_tokens import SketchControlTokens


def call_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    conditioning: torch.Tensor,
    cfg_strength: float,
    ctrl_tokens: torch.Tensor | None = None,
):
    # The unconditional branch drops the control tokens along with the
    # conditioning; only ctrl-aware fields ever receive the keyword.
    y_c = f(x, t, conditioning) if ctrl_tokens is None else f(x, t, conditioning, ctrl_tokens)
    y_u = f(x, t, None)

    return (1 - cfg_strength) * y_u + cfg_strength * y_c


def rk4_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    conditioning: torch.Tensor,
    cfg_strength: float,
    ctrl_tokens: torch.Tensor | None = None,
):
    f = partial(
        call_with_cfg,
        f,
        conditioning=conditioning,
        cfg_strength=cfg_strength,
        ctrl_tokens=ctrl_tokens,
    )
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class VSTFlowMatchingModule(LightningModule):
    """Flow-matching LightningModule for VST parameter prediction (CFG + RK4 sampling)."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        vector_field: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # Keyword-only: a stale positional caller would silently train at a bogus width.
        *,
        num_params: int,
        conditioning: Conditioning = "mel",
        sketch_controls: SketchControls = None,
        sketch_dropout_rate: float = 0.2,
        sketch_all_dropout_rate: float = 0.2,
        encoder_num_heads: int | None = None,
        encoder_output_dim: int | None = None,
        warmup_steps: int = 5000,
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 4.0,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 4.0,
        compile: bool = False,
    ):
        """Wire the encoder/vector-field and persist the flow-matching hyperparameters.

        :param encoder: Encoder over legacy mel or a fixed-shape embedding.
        :param vector_field: Network predicting the flow velocity field.
        :param optimizer: ``functools.partial``-style optimizer factory (Hydra
            ``_partial_: true``); invoked in :meth:`configure_optimizers`.
        :param scheduler: ``functools.partial``-style scheduler factory or ``None``.
        :param num_params: Parameter-vector width the field operates on.
        :param conditioning: Legacy mel/m2l mode or a fixed-shape embedding spec.
        :param sketch_controls: Optional sketch-control spec enabling concat
            control-token injection into the vector field (#2612).
        :param sketch_dropout_rate: Independent per-control CFG drop probability.
        :param sketch_all_dropout_rate: Probability of additionally dropping
            every control at once (Sketch2Sound-style joint dropout).
        :param encoder_num_heads: Model-owned attention head count for sequence encoders.
        :param encoder_output_dim: Configured encoder width consumed by the vector field.
        :param warmup_steps: If positive, wrap the scheduler with a linear warmup.
        :param cfg_dropout_rate: Probability of dropping conditioning during training (CFG).
        :param rectified_sigma_min: Minimum noise scale for the rectified probability path.
        :param validation_sample_steps: RK4 integration steps used at validation.
        :param validation_cfg_strength: Classifier-free-guidance strength at validation.
        :param test_sample_steps: RK4 integration steps used at test.
        :param test_cfg_strength: Classifier-free-guidance strength at test.
        :param compile: Whether to compile the encoder and vector field during fit setup.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.vector_field = vector_field
        self._embedding_conditioning = resolve_embedding_conditioning(conditioning)
        self._sketch_controls = resolve_sketch_controls(sketch_controls)
        self.sketch_tokens = (
            SketchControlTokens(
                d_model=vector_field.d_model,
                num_ctrl_tokens=self._sketch_controls.num_ctrl_tokens,
            )
            if self._sketch_controls is not None
            else None
        )

        self.val_param_mse_best_swap = BestSwapParamMSE()
        self.test_param_mse_best_swap = BestSwapParamMSE()

    def on_train_start(self):
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
        return select_conditioning(batch, self._embedding_conditioning)

    def _sketch_drop_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Draw the per-control CFG drop mask for one training step.

        :param batch_size: Rows in the current batch.
        :param device: Device the mask is drawn on.
        :returns: ``(batch_size, 3)`` boolean mask; ``True`` drops a control.
        """
        drop = torch.rand(batch_size, 3, device=device) < self.hparams.sketch_dropout_rate
        drop_all = torch.rand(batch_size, 1, device=device) < self.hparams.sketch_all_dropout_rate
        return drop | drop_all

    def _sketch_tokens_from_batch(
        self, batch: dict[str, torch.Tensor], *, training: bool
    ) -> torch.Tensor | None:
        """Tokenize the batch's sketch controls when a spec is configured.

        :param batch: Model batch; must carry ``sketch_ctrl`` when configured.
        :param training: Whether to draw the CFG drop mask; inference keeps all.
        :returns: Control tokens, or ``None`` without a configured spec.
        """
        if self.sketch_tokens is None:
            return None
        controls = batch["sketch_ctrl"]
        drop_mask = (
            self._sketch_drop_mask(controls.shape[0], controls.device)
            if training
            else torch.zeros(controls.shape[0], 3, dtype=torch.bool, device=controls.device)
        )
        return self.sketch_tokens(controls, drop_mask)

    def _train_step(self, batch: dict[str, torch.Tensor]):
        conditioning = self._get_conditioning_from_batch(batch)
        params = batch["params"]
        noise = batch["noise"]

        conditioning = self.encoder(conditioning)
        z = self.vector_field.apply_dropout(conditioning, self.hparams.cfg_dropout_rate)
        ctrl_tokens = self._sketch_tokens_from_batch(batch, training=True)

        with torch.no_grad():
            t = self._sample_time(params.shape[0], params.device)
            w = self._weight_time(t)

            x0 = noise
            x1 = params

            x_t = self._sample_probability_path(x0, x1, t)
            target = self._evaluate_target_field(x0, x1, x_t, t)

        if ctrl_tokens is None:
            prediction = self.vector_field(x_t, t, z)
        else:
            prediction = self.vector_field(x_t, t, z, ctrl_tokens)

        loss = (prediction - target).square().mean(dim=-1)
        loss = loss * w
        loss = loss.mean()

        penalty = None
        if hasattr(self.vector_field, "penalty"):
            penalty = self.vector_field.penalty()

        return loss, penalty

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
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
        ctrl_tokens: torch.Tensor | None = None,
    ):
        if conditioning is not None:
            conditioning = self.encoder(conditioning)
        else:
            # Unconditional sampling drops the sketch controls with everything else.
            ctrl_tokens = None

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
                ctrl_tokens,
            )
            t = t + dt

        return sample

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.validation_sample_steps,
            self.hparams.validation_cfg_strength,
            self._sketch_tokens_from_batch(batch, training=False),
        )

        per_param_mse = (pred_params - batch["params"]).square().mean(dim=0)
        per_param_mse_best_swap = best_swap_per_param_mse(pred_params, batch["params"])
        param_mse = per_param_mse.mean()
        self.log("val/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        self.val_param_mse_best_swap.update(pred_params, batch["params"])
        self.log(
            "val/param_mse_best_swap",
            self.val_param_mse_best_swap,
            on_step=False,
            on_epoch=True,
        )

        return {
            "param_mse": param_mse,
            "per_param_mse": per_param_mse,
            "per_param_mse_best_swap": per_param_mse_best_swap,
            "preds": pred_params,
        }

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.test_sample_steps,
            self.hparams.test_cfg_strength,
            self._sketch_tokens_from_batch(batch, training=False),
        )

        param_mse = (pred_params - batch["params"]).square().mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        self.test_param_mse_best_swap.update(pred_params, batch["params"])
        self.log(
            "test/param_mse_best_swap",
            self.test_param_mse_best_swap,
            on_step=False,
            on_epoch=True,
        )

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
                self._sketch_tokens_from_batch(batch, training=False),
            ),
            batch,
        )

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.vector_field.compile()
            self.encoder.compile()

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
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


# Deprecated alias: archived W&B run configs and external job scripts resolve the
# old ``_target_`` path.
SurgeFlowMatchingModule = VSTFlowMatchingModule
