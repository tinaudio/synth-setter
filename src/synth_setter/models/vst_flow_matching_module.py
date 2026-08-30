"""Lightning module for flow-matching VST parameter prediction."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from synth_setter.metrics import (
    BestSwapParamMSE,
    NumberGroupSwapParamMSE,
    best_swap_per_param_mse,
    number_group_swap_per_param_mse,
)
from synth_setter.models.components.pretrained_encoder import PretrainedConditioningEncoder
from synth_setter.models.components.sketch_tokens import CONTROL_GROUPS, SketchControlTokens

_BATCH_SHAPE = "batch"
_BATCH_ANY_SHAPE = "batch ..."
_BATCH_TIME_SHAPE = "batch 1"
_FROZEN_BACKBONE_PREFIX = "encoder.backbone."

if TYPE_CHECKING:
    from synth_setter.models.components.audio_feedback import (
        AudioFeedbackLoss,
        GradientBalance,
    )


@dataclass(frozen=True)
class ConditioningKeepMasks:
    """Positive keep state for every identity-bearing conditioning stream.

    .. attribute :: content

       Content-conditioning keep state shaped ``(batch,)``.

    .. attribute :: sketch_groups

       Per-sketch-group keep state shaped ``(batch, 3)``.
    """

    content: Bool[torch.Tensor, _BATCH_SHAPE]
    sketch_groups: Bool[torch.Tensor, f"batch {len(CONTROL_GROUPS)}"]

    @classmethod
    @jaxtyped(typechecker=beartype)
    def content_only(cls, content: Bool[torch.Tensor, _BATCH_SHAPE]) -> ConditioningKeepMasks:
        """Build keep state for a run with no sketch controls configured.

        :param content: Content-conditioning keep state shaped ``(batch,)``.
        :returns: Keep state whose sketch groups are all absent.
        """
        return cls(
            content=content,
            sketch_groups=torch.zeros(
                content.shape[0],
                len(CONTROL_GROUPS),
                dtype=torch.bool,
                device=content.device,
            ),
        )

    @property
    @jaxtyped(typechecker=beartype)
    def identity_keep(self) -> Bool[torch.Tensor, _BATCH_SHAPE]:
        """Return rows retaining at least one identity-bearing stream.

        :returns: Positive identity keep state shaped ``(batch,)``.
        """
        return self.content | self.sketch_groups.any(dim=-1)


@dataclass(frozen=True)
class ControlTokenBranches:
    """Complete control-token state for the two joint-CFG branches.

    .. attribute :: conditional

       Full sketch-control tokens for the content-conditioned branch.

    .. attribute :: unconditional

       PE-only tokens for the unconditional branch.
    """

    conditional: Float[torch.Tensor, "batch tokens d_model"]
    unconditional: Float[torch.Tensor, "batch tokens d_model"]


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

    .. attribute :: conditioning_keep

       Keep state sampled for this step's conditioning streams.
    """

    loss: torch.Tensor
    audio_term: torch.Tensor | None
    penalty: torch.Tensor | None
    grad_balance: GradientBalance | None
    t: torch.Tensor
    conditioning_keep: ConditioningKeepMasks


type _TimeField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@jaxtyped(typechecker=beartype)
def joint_cfg_velocity(
    conditional_field: _TimeField,
    unconditional_field: _TimeField,
    cfg_strength: float,
) -> _TimeField:
    """Build one joint two-branch classifier-free-guidance velocity.

    :param conditional_field: Content-plus-sketch conditional time field.
    :param unconditional_field: Unconditional time field.
    :param cfg_strength: Joint classifier-free-guidance scale.
    :returns: Two-argument guided velocity field.
    """
    return lambda x, t: (
        (1 - cfg_strength) * unconditional_field(x, t) + cfg_strength * conditional_field(x, t)
    )


@jaxtyped(typechecker=beartype)
def build_guided_velocity(
    field: torch.nn.Module,
    conditioning: Shaped[torch.Tensor, "batch ..."] | None,
    cfg_strength: float,
    *,
    control_tokens: ControlTokenBranches | None = None,
) -> _TimeField:
    """Bind content and optional control tokens into the two joint-CFG branches.

    :param field: Model velocity field.
    :param conditioning: Encoded content conditioning for the conditional branch.
    :param cfg_strength: Joint classifier-free-guidance scale.
    :param control_tokens: Complete conditional/unconditional control-token state.
    :returns: Two-argument guided velocity field.
    """
    conditional = control_tokens.conditional if control_tokens is not None else None
    unconditional = control_tokens.unconditional if control_tokens is not None else None
    return joint_cfg_velocity(
        _bind_branch(field, conditioning, conditional),
        _bind_branch(field, None, unconditional),
        cfg_strength,
    )


@jaxtyped(typechecker=beartype)
def _bind_branch(
    field: torch.nn.Module,
    conditioning: Shaped[torch.Tensor, "batch ..."] | None,
    control_tokens: Float[torch.Tensor, "batch tokens d_model"] | None,
) -> _TimeField:
    """Bind one CFG branch's conditioning into a two-argument time field.

    Content conditioning binds positionally: ``ConditionalResidualMLP`` names the
    argument ``c`` while the other backbones name it ``conditioning``.

    :param field: Model velocity field.
    :param conditioning: Encoded content conditioning, or ``None`` for the
        unconditional branch.
    :param control_tokens: This branch's control tokens, or ``None`` without sketch support.
    :returns: Two-argument velocity field over parameter state and time.
    """
    if control_tokens is None:
        return lambda x, t: field(x, t, conditioning)
    return lambda x, t: field(x, t, conditioning, control_tokens=control_tokens)


@jaxtyped(typechecker=beartype)
def rk4_step(
    f: _TimeField,
    x: Float[torch.Tensor, "batch params"],
    t: Float[torch.Tensor, "batch 1"],
    dt: float | Float[torch.Tensor, "batch 1"],
) -> Float[torch.Tensor, "batch params"]:
    """Advance a two-argument time field by one classical RK4 step.

    :param f: Time field accepting only parameter state and time.
    :param x: Current parameter state.
    :param t: Current flow time.
    :param dt: Integration step in warped time.
    :returns: Parameter state after one RK4 step.
    """
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
        param_spec: str | None = None,
        conditioning: Conditioning = "mel",
        sketch_controls: SketchControls = None,
        sketch_dropout_rate: float = 0.1,
        all_conditioning_dropout_rate: float = 0.1,
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
        :param param_spec: Registered parameter spec enabling structured swap metrics.
        :param conditioning: Legacy mel/m2l mode or a fixed-shape embedding spec.
        :param sketch_controls: Optional sketch-control spec enabling concat
            control-token injection into the vector field (#2612).
        :param sketch_dropout_rate: Independent per-sketch-group drop probability.
        :param all_conditioning_dropout_rate: Probability of dropping content and
            every sketch group in one global event.
        :param audio_loss: Optional audio-feedback term on the rendered one-step
            estimate; requires an uncompiled, single-device, drop-last run (#2585).
        :param encoder_num_heads: Model-owned attention head count for sequence encoders.
        :param encoder_output_dim: Configured encoder width consumed by the vector field.
        :param warmup_steps: If positive, wrap the scheduler with a linear warmup.
        :param cfg_dropout_rate: Independent content-conditioning drop probability
            during training (CFG).
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

        # Saving hyperparameters deep-copies them, which a weight-normalized frozen encoder
        # inside the audio term cannot survive; the term is training-time only, so it is not
        # reconstructed from hparams either.
        self.save_hyperparameters(ignore=["encoder", "audio_loss"], logger=False)
        if not isinstance(encoder, PretrainedConditioningEncoder):
            # Existing load_from_checkpoint consumers reconstruct legacy encoders from hparams.
            self.hparams["encoder"] = encoder

        self.encoder = encoder
        self.vector_field = vector_field
        self._sketch_controls = resolve_sketch_controls(sketch_controls)
        self.sketch_tokens = (
            SketchControlTokens(
                d_model=vector_field.d_model,
                num_control_tokens=self._sketch_controls.num_control_tokens,
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
        metric_spec = None
        if param_spec is not None:
            from synth_setter.data.vst import param_specs

            metric_spec = param_specs[param_spec]
        self.val_param_mse_number_group_swap = (
            NumberGroupSwapParamMSE(metric_spec) if metric_spec is not None else None
        )
        self.test_param_mse_number_group_swap = (
            NumberGroupSwapParamMSE(metric_spec) if metric_spec is not None else None
        )

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

    @jaxtyped(typechecker=beartype)
    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Exclude re-resolvable frozen CLAP state from a Lightning checkpoint.

        :param checkpoint: Mutable Lightning checkpoint payload.
        :raises TypeError: A pretrained-encoder checkpoint has malformed state metadata.
        """
        if not isinstance(self.encoder, PretrainedConditioningEncoder):
            return
        state = checkpoint.get("state_dict")
        if not isinstance(state, MutableMapping):
            raise TypeError("Lightning checkpoint state_dict must be a mutable mapping")
        for key in tuple(state):
            if isinstance(key, str) and key.startswith(_FROZEN_BACKBONE_PREFIX):
                del state[key]

        hyperparameters = checkpoint.get("hyper_parameters")
        if isinstance(hyperparameters, MutableMapping):
            hyperparameters.pop("encoder", None)

    @jaxtyped(typechecker=beartype)
    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Restore current frozen CLAP state so Lightning can load trainable state strictly.

        :param checkpoint: Mutable Lightning checkpoint payload.
        :raises TypeError: A pretrained-encoder checkpoint has a malformed state dictionary.
        """
        if not isinstance(self.encoder, PretrainedConditioningEncoder):
            return
        state = checkpoint.get("state_dict")
        if not isinstance(state, MutableMapping):
            raise TypeError("Lightning checkpoint state_dict must be a mutable mapping")
        for key, value in self.state_dict().items():
            if key.startswith(_FROZEN_BACKBONE_PREFIX):
                state[key] = value

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
    def _sample_conditioning_keep_masks(
        self, batch_size: int, device: torch.device
    ) -> ConditioningKeepMasks:
        """Draw independent stream keeps and apply the global all-drop event.

        :param batch_size: Rows in the current batch.
        :param device: Device the masks are drawn on.
        :returns: Positive content and per-sketch-group keep masks.
        """
        content = torch.rand(batch_size, device=device) > self.hparams.cfg_dropout_rate
        sketch_groups = (
            torch.rand(batch_size, len(CONTROL_GROUPS), device=device)
            > self.hparams.sketch_dropout_rate
        )
        global_keep = (
            torch.rand(batch_size, device=device) > self.hparams.all_conditioning_dropout_rate
        )
        return ConditioningKeepMasks(
            content=content & global_keep,
            sketch_groups=sketch_groups & global_keep.unsqueeze(-1),
        )

    @jaxtyped(typechecker=beartype)
    def _prepare_conditioning(
        self, batch: dict[str, Shaped[torch.Tensor, ...] | None]
    ) -> tuple[
        Shaped[torch.Tensor, "batch ..."],
        Float[torch.Tensor, "batch tokens d_model"] | None,
        ConditioningKeepMasks,
    ]:
        """Encode the content stream and apply this step's sampled dropout policy.

        :param batch: Model batch carrying content conditioning, plus ``sketch_ctrl``
            whenever a sketch spec is configured.
        :returns: Post-dropout conditioning, sketch control tokens (``None`` without a
            configured spec), and the keep masks that produced both.
        """
        conditioning = self.encoder(self._get_conditioning_from_batch(batch))
        if (
            conditioning.ndim == 3
            and conditioning.shape[1] > 1
            and self._is_trainer_logging_step()
        ):
            self._log_slot_cosine(conditioning.detach())
        if self.sketch_tokens is None:
            # Legacy path: apply_dropout draws its own mask, keeping the no-sketch
            # RNG stream identical to runs from before sketch support.
            z, content_keep = self.vector_field.apply_dropout(
                conditioning, self.hparams.cfg_dropout_rate
            )
            return z, None, ConditioningKeepMasks.content_only(content_keep)

        keep = self._sample_conditioning_keep_masks(conditioning.shape[0], conditioning.device)
        z, _ = self.vector_field.apply_dropout(conditioning, keep_mask=keep.content)
        return z, self.sketch_tokens(batch["sketch_ctrl"], keep.sketch_groups), keep

    @jaxtyped(typechecker=beartype)
    def _control_token_branches_from_batch(
        self, batch: dict[str, Shaped[torch.Tensor, ...] | None]
    ) -> ControlTokenBranches | None:
        """Build complete full-sketch and PE-only control branches for inference.

        :param batch: Model batch carrying sketch controls when configured.
        :returns: Both control-token branches, or ``None`` without sketch support.
        """
        if self.sketch_tokens is None:
            return None
        controls = batch["sketch_ctrl"]
        keep = torch.ones(
            controls.shape[0], len(CONTROL_GROUPS), dtype=torch.bool, device=controls.device
        )
        return ControlTokenBranches(
            conditional=self.sketch_tokens(controls, keep),
            unconditional=self.sketch_tokens.unconditional(controls.shape[0]),
        )

    @jaxtyped(typechecker=beartype)
    def _is_trainer_logging_step(self) -> bool:
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
    def _log_slot_cosine(self, conditioning: Float[torch.Tensor, "batch slots dim"]) -> None:
        """Log how far apart the per-layer conditioning slots sit before dropout.

        A value approaching one means nominally separate slots have converged to the same read, so
        the extra slots carry nothing the field's layers can distinguish.

        :param conditioning: Detached layerwise conditioning.
        """
        slots = torch.nn.functional.normalize(conditioning, dim=-1)
        gram = slots @ slots.transpose(-2, -1)
        count = gram.shape[-1]
        off_diagonal = gram.sum(dim=(-2, -1)) - gram.diagonal(dim1=-2, dim2=-1).sum(-1)
        mean = (off_diagonal / (count * (count - 1))).mean()
        self.log("train/slot_cosine", mean, on_step=True, on_epoch=False)

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
        params = batch["params"]
        noise = batch["noise"]

        z, control_tokens, conditioning_keep = self._prepare_conditioning(batch)

        with torch.no_grad():
            t = self._sample_time(params.shape[0], params.device)
            w = self._weight_time(t)

            x0 = noise
            x1 = params

            x_t = self._sample_probability_path(x0, x1, t)
            target = self._evaluate_target_field(x0, x1, x_t, t)

        if control_tokens is None:
            prediction = self.vector_field(x_t, t, z)
        else:
            prediction = self.vector_field(x_t, t, z, control_tokens=control_tokens)

        loss = (prediction - target).square().mean(dim=-1)
        loss = loss * w
        loss = loss.mean()

        audio_term = None
        grad_balance = None
        if self.audio_loss is not None:
            # One-step estimate of x1 from the current field; rendering it keeps
            # autograd connected so latent audio error reaches the field's weights.
            theta_hat = x_t + (1 - t) * prediction
            # Fully unconditional rows estimate the marginal, so their row-specific
            # target-audio residual is high-variance noise rather than identity signal.
            audio_term = self.audio_loss(
                theta_hat,
                t,
                batch["audio"],
                keep=conditioning_keep.identity_keep,
            )
            if self._is_trainer_logging_step():
                from synth_setter.models.components.audio_feedback import gradient_balance

                grad_balance = gradient_balance(
                    flow_loss=loss, audio_term=audio_term, shared=prediction
                )

        penalty = None
        if hasattr(self.vector_field, "penalty"):
            penalty = self.vector_field.penalty()

        return TrainStepOutputs(
            loss=loss,
            audio_term=audio_term,
            penalty=penalty,
            grad_balance=grad_balance,
            t=t,
            conditioning_keep=conditioning_keep,
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

    @jaxtyped(typechecker=beartype)
    def _velocity_field(
        self,
        conditioning: Shaped[torch.Tensor, "batch ..."] | None,
        cfg_strength: float,
        control_tokens: ControlTokenBranches | None,
    ) -> _TimeField:
        """Build the time field the sampler integrates.

        A seam, not a wrapper: a subclass whose sampling velocity differs from its training
        velocity overrides this and inherits the integration loop unchanged.

        :param conditioning: Encoded content conditioning for the conditional branch.
        :param cfg_strength: Joint classifier-free-guidance scale.
        :param control_tokens: Complete control-token state, or ``None`` without sketch support.
        :returns: Two-argument velocity field over parameter state and time.
        """
        return build_guided_velocity(
            self.vector_field,
            conditioning,
            cfg_strength,
            control_tokens=control_tokens,
        )

    def _sample(
        self,
        conditioning: torch.Tensor | None,
        noise: torch.Tensor,
        steps: int,
        cfg_strength: float,
        *,
        control_tokens: ControlTokenBranches | None = None,
    ) -> torch.Tensor:
        if conditioning is not None:
            conditioning = self.encoder(conditioning)

        guided_velocity = self._velocity_field(conditioning, cfg_strength, control_tokens)
        t = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt = 1.0 / steps
        sample = noise

        for _ in range(steps):
            warped_t = self._warp_time(t)
            warped_t_plus_dt = self._warp_time(t + dt)
            warped_dt = warped_t_plus_dt - warped_t

            sample = rk4_step(guided_velocity, sample, warped_t, warped_dt)
            t = t + dt

        return sample

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.validation_sample_steps,
            self.hparams.validation_cfg_strength,
            control_tokens=self._control_token_branches_from_batch(batch),
        )

        per_param_mse = (pred_params - batch["params"]).square().mean(dim=0)
        per_param_mse_best_swap = best_swap_per_param_mse(pred_params, batch["params"])
        per_param_mse_number_group_swap = None
        if self.val_param_mse_number_group_swap is not None:
            per_param_mse_number_group_swap = number_group_swap_per_param_mse(
                pred_params,
                batch["params"],
                self.val_param_mse_number_group_swap.param_spec,
            )
        param_mse = per_param_mse.mean()
        self.log("val/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        self.val_param_mse_best_swap.update(pred_params, batch["params"])
        self.log(
            "val/param_mse_best_swap",
            self.val_param_mse_best_swap,
            on_step=False,
            on_epoch=True,
        )
        if self.val_param_mse_number_group_swap is not None:
            self.val_param_mse_number_group_swap.update(pred_params, batch["params"])
            self.log(
                "val/param_mse_number_group_swap",
                self.val_param_mse_number_group_swap,
                on_step=False,
                on_epoch=True,
            )

        outputs = {
            "param_mse": param_mse,
            "per_param_mse": per_param_mse,
            "per_param_mse_best_swap": per_param_mse_best_swap,
            "preds": pred_params,
        }
        if per_param_mse_number_group_swap is not None:
            outputs["per_param_mse_number_group_swap"] = per_param_mse_number_group_swap
        return outputs

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.test_sample_steps,
            self.hparams.test_cfg_strength,
            control_tokens=self._control_token_branches_from_batch(batch),
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
        if self.test_param_mse_number_group_swap is not None:
            self.test_param_mse_number_group_swap.update(pred_params, batch["params"])
            self.log(
                "test/param_mse_number_group_swap",
                self.test_param_mse_number_group_swap,
                on_step=False,
                on_epoch=True,
            )

        return param_mse

    def on_test_epoch_end(self) -> None:
        pass

    def predict_step(
        self, batch: dict[str, Shaped[torch.Tensor, _BATCH_ANY_SHAPE]], batch_idx: int
    ):
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
                control_tokens=self._control_token_branches_from_batch(batch),
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

    @jaxtyped(typechecker=beartype)
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Reject a non-finite gradient before clipping rescales every parameter by NaN.

        ``clip_grad_norm_`` defaults to ``error_if_nonfinite=False``, so one overflowing
        row turns the total norm into NaN and poisons all weights; the failure then
        surfaces a step later as a diverged parameter estimate. Runs under 32-bit
        precision only — an AMP ``GradScaler`` produces transient infs by design.

        :param optimizer: Optimizer whose gradients are about to be clipped.
        :param gradient_clip_val: Clip threshold Lightning resolves from the trainer.
        :param gradient_clip_algorithm: Clip algorithm Lightning resolves from the trainer.
        :raises ValueError: Any parameter carries a non-finite gradient.
        """
        corrupted = [
            name
            for name, parameter in self.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if corrupted:
            raise ValueError(
                f"non-finite gradient in {len(corrupted)} parameter(s) {corrupted}; rejecting "
                "the step at its source rather than letting clipping scale every parameter by NaN"
            )
        super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self) -> dict[str, object]:
        trainable_parameters = (
            parameter for parameter in self.trainer.model.parameters() if parameter.requires_grad
        )
        optimizer = self.hparams.optimizer(params=trainable_parameters)

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
