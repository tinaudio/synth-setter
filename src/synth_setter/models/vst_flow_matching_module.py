"""Lightning module for flow-matching VST parameter prediction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Shaped, jaxtyped
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from synth_setter.conditioning import (
    Conditioning,
    SketchControls,
    conditioning_batch_key,
    resolve_sketch_controls,
)
from synth_setter.metrics import BestSwapParamMSE, best_swap_per_param_mse
from synth_setter.models.components.sketch_tokens import CONTROL_GROUPS, SketchControlTokens

_BATCH_SHAPE = "batch"
_BATCH_TIME_SHAPE = "batch 1"

if TYPE_CHECKING:
    from synth_setter.models.components.audio_feedback import (
        AudioFeedbackLoss,
        GradientBalance,
    )


@dataclass(frozen=True)
class TrainStepOutputs:
    """Loss terms produced by one training step.

    .. attribute :: loss

       Flow-matching loss; the only term every configuration produces.

    .. attribute :: audio_term

       Weighted audio-feedback loss, or ``None`` without an attached audio loss.

    .. attribute :: penalty

       Vector-field regularization penalty, or ``None`` for fields that define none.

    .. attribute :: grad_balance

       Gradient diagnostics for the audio term, or ``None`` off the probe cadence.

    .. attribute :: t

       Flow time per row, shaped ``(batch, 1)``.
    """

    loss: torch.Tensor
    audio_term: torch.Tensor | None
    penalty: torch.Tensor | None
    grad_balance: GradientBalance | None
    t: torch.Tensor


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
        audio_loss: AudioFeedbackLoss | None = None,
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
    ) -> None:
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
        :param audio_loss: Optional audio-feedback term on the rendered one-step
            estimate; requires an uncompiled, single-device, drop-last run (#2585).
        :param encoder_num_heads: Model-owned attention head count for sequence encoders.
        :param encoder_output_dim: Configured encoder width consumed by the vector field.
        :param warmup_steps: If positive, wrap the scheduler with a linear warmup.
        :param cfg_dropout_rate: Probability of dropping conditioning during
            training (CFG); with sketch controls configured, these rows also
            drop every control so the trained state matches the CFG
            unconditional branch.
        :param rectified_sigma_min: Minimum noise scale for the rectified probability path.
        :param validation_sample_steps: RK4 integration steps used at validation.
        :param validation_cfg_strength: Classifier-free-guidance strength at validation.
        :param test_sample_steps: RK4 integration steps used at test.
        :param test_cfg_strength: Classifier-free-guidance strength at test.
        :param compile: Whether to compile the encoder and vector field during fit setup.
        :raises ValueError: ``audio_loss`` is combined with a nonzero
            ``rectified_sigma_min`` or with ``compile=True`` (#2585).
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.vector_field = vector_field
        self._sketch_controls = resolve_sketch_controls(sketch_controls)
        self.sketch_tokens = (
            SketchControlTokens(
                d_model=vector_field.d_model,
                num_ctrl_tokens=self._sketch_controls.num_ctrl_tokens,
            )
            if self._sketch_controls is not None
            else None
        )
        self.audio_loss = audio_loss
        if audio_loss is not None and rectified_sigma_min != 0.0:
            # theta_hat = x_t + (1 - t) * prediction is the exact one-step estimate only
            # on the sigma-free path; any other sigma silently biases the rendered params.
            raise ValueError(
                f"audio feedback requires rectified_sigma_min=0, got {rectified_sigma_min}"
            )
        if audio_loss is not None and compile:
            from synth_setter.models.components.audio_feedback import (
                validate_audio_feedback_runtime,
            )

            # Only `compiled` is known here; world_size is re-checked against the real
            # trainer in on_train_start. Must fail before setup() compiles (#2585).
            validate_audio_feedback_runtime(compiled=True, world_size=1)
        self._conditioning_key = conditioning_batch_key(conditioning)

        self.val_param_mse_best_swap = BestSwapParamMSE()
        self.test_param_mse_best_swap = BestSwapParamMSE()

    def on_train_start(self) -> None:
        if self.audio_loss is None:
            return

        from synth_setter.models.components.audio_feedback import (
            validate_audio_feedback_runtime,
        )

        validate_audio_feedback_runtime(
            compiled=self.hparams.compile,
            world_size=self.trainer.world_size,
        )

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
        return batch[self._conditioning_key]

    @jaxtyped(typechecker=beartype)
    def _sketch_drop_mask(
        self,
        batch_size: int,
        device: torch.device,
        force_drop: Bool[torch.Tensor, "batch 1"] | None = None,
    ) -> Bool[torch.Tensor, f"batch {len(CONTROL_GROUPS)}"]:
        """Draw the per-control CFG drop mask for one training step.

        :param batch_size: Rows in the current batch.
        :param device: Device the mask is drawn on.
        :param force_drop: Rows whose controls are all dropped regardless of the
            configured rates (audio-CFG coupling), or ``None``.
        :returns: ``(batch_size, len(CONTROL_GROUPS))`` boolean mask; ``True`` drops a control.
        """
        num_groups = len(CONTROL_GROUPS)
        drop = torch.rand(batch_size, num_groups, device=device) < self.hparams.sketch_dropout_rate
        drop_all = torch.rand(batch_size, 1, device=device) < self.hparams.sketch_all_dropout_rate
        if force_drop is not None:
            drop_all = drop_all | force_drop
        return drop | drop_all

    @jaxtyped(typechecker=beartype)
    def _sketch_tokens_from_batch(
        self,
        batch: dict[str, Shaped[torch.Tensor, ...] | None],
        *,
        training: bool,
        force_drop: Bool[torch.Tensor, "batch 1"] | None = None,
    ) -> Float[torch.Tensor, "batch tokens d_model"] | None:
        """Tokenize the batch's sketch controls when a spec is configured.

        :param batch: Model batch; must carry ``sketch_ctrl`` when configured.
        :param training: Whether to draw the CFG drop mask; inference keeps all.
        :param force_drop: Training rows whose controls are all dropped
            (audio-CFG coupling), or ``None``.
        :returns: Control tokens, or ``None`` without a configured spec.
        """
        if self.sketch_tokens is None:
            return None
        controls = batch["sketch_ctrl"]
        drop_mask = (
            self._sketch_drop_mask(controls.shape[0], controls.device, force_drop=force_drop)
            if training
            else torch.zeros(
                controls.shape[0], len(CONTROL_GROUPS), dtype=torch.bool, device=controls.device
            )
        )
        return self.sketch_tokens(controls, drop_mask)

    @jaxtyped(typechecker=beartype)
    def _should_probe_gradient_balance(self) -> bool:
        """Whether this step pays for the probe's extra backward through the renderer.

        :returns: True on Lightning's own logging cadence; False when detached from a trainer.
        """
        # self.trainer raises when detached, so the private attribute is the only probe
        # that works for direct _train_step calls outside a fit loop.
        if self._trainer is None:
            return False
        every = self.trainer.log_every_n_steps
        return every > 0 and self.trainer.global_step % every == 0

    @jaxtyped(typechecker=beartype)
    def _log_gradient_time_profile(
        self,
        audio_row_norms: Float[torch.Tensor, _BATCH_SHAPE],
        t: Float[torch.Tensor, _BATCH_TIME_SHAPE],
    ) -> None:
        """Log where along the flow time axis the audio term's gradient actually lands.

        :param audio_row_norms: Per-row audio gradient norm.
        :param t: Flow time shaped ``(batch, 1)``.
        """
        from synth_setter.models.components.audio_feedback import time_bucket_means

        for index, mean in enumerate(time_bucket_means(audio_row_norms, t)):
            if torch.isfinite(mean):
                self.log(
                    f"train/audio_grad_norm_t_bucket_{index}", mean, on_step=True, on_epoch=False
                )

    def _train_step(self, batch: dict[str, torch.Tensor]) -> TrainStepOutputs:
        """Run one training forward pass and assemble every term the logger consumes.

        :param batch: Online or stored batch carrying params, noise, and audio.
        :returns: Flow loss plus whichever optional terms this configuration produces.
        """

        conditioning = self._get_conditioning_from_batch(batch)
        params = batch["params"]
        noise = batch["noise"]

        conditioning = self.encoder(conditioning)
        if self.sketch_tokens is None:
            z, keep = self.vector_field.apply_dropout(conditioning, self.hparams.cfg_dropout_rate)
            ctrl_tokens = None
        else:
            # Sketch2Sound-style coupling: rows whose audio conditioning is
            # CFG-dropped also drop every control, so training presents the
            # exact all-unconditioned state call_with_cfg queries at inference.
            joint_drop = (
                torch.rand(conditioning.shape[0], 1, device=conditioning.device)
                < self.hparams.cfg_dropout_rate
            )
            z, keep = self.vector_field.apply_dropout(
                conditioning, self.hparams.cfg_dropout_rate, drop_mask=joint_drop
            )
            ctrl_tokens = self._sketch_tokens_from_batch(
                batch, training=True, force_drop=joint_drop
            )

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

        audio_term = None
        grad_balance = None
        if self.audio_loss is not None:
            # One-step estimate of x1 from the current field; rendering it keeps
            # autograd connected so latent audio error reaches the field's weights.
            theta_hat = x_t + (1 - t) * prediction
            # `keep` zeroes CFG-dropped rows: their estimate comes from the marginal, so
            # its residual against that row's own audio is high-variance noise, not signal.
            audio_term = self.audio_loss(
                theta_hat, t, batch["audio"], encoder=self.encoder, keep=keep
            )
            if self._should_probe_gradient_balance():
                from synth_setter.models.components.audio_feedback import gradient_balance

                grad_balance = gradient_balance(
                    flow_loss=loss, audio_term=audio_term, shared=prediction
                )

        penalty = None
        if hasattr(self.vector_field, "penalty"):
            penalty = self.vector_field.penalty()

        return TrainStepOutputs(
            loss=loss, audio_term=audio_term, penalty=penalty, grad_balance=grad_balance, t=t
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._train_step(batch)
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)

        total = outputs.loss
        if outputs.audio_term is not None:
            # Dominated by high-t rows, where the weight is maximal; the gradient peaks at
            # mid t, so this scalar does not track where the term is actually teaching.
            self.log(
                "train/audio_loss", outputs.audio_term, on_step=True, on_epoch=True, prog_bar=True
            )
            total = total + outputs.audio_term

        if outputs.grad_balance is not None:
            # Set lambda_audio from the ratio, not the loss value; watch the cosine turn
            # negative for the point the audio term starts fighting the flow objective.
            ratio, cosine = outputs.grad_balance.ratio, outputs.grad_balance.cosine
            self.log("train/audio_grad_ratio", ratio, on_step=True, on_epoch=False)
            self.log("train/audio_grad_cosine", cosine, on_step=True, on_epoch=False)
            self._log_gradient_time_profile(outputs.grad_balance.audio_row_norms, outputs.t)

        if outputs.penalty is not None:
            self.log("train/penalty", outputs.penalty, on_step=True, on_epoch=True, prog_bar=True)
            total = total + outputs.penalty

        return total

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
