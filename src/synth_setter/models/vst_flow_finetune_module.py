"""Simulator-feedback finetuning of a pretrained flow (arXiv 2410.22573, section 3.2).

A frozen pretrained flow supplies the velocity, a simulator scores the one-step estimate that
velocity implies, and a small control network folds that score back in. The simulator's cost
enters the run **only** through the control network's input — the objective stays plain
conditional flow matching, which is what leaves the pretrained field's guarantees intact and
is the whole reason this is not the audio-loss arm in
:mod:`synth_setter.models.components.audio_feedback`.

Typical usage:
    trainer.fit(VSTFlowFinetuneModule(..., base_checkpoint=..., control_mode="gradient_spectral"))
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor

from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
)
from synth_setter.models.components.simulator_control import (
    DEFAULT_CONTROL_T_MIN,
    ControlledFlow,
    ControlNet,
    gradient_control_signal,
    learned_control_signal,
)
from synth_setter.models.vst_flow_matching_module import (
    ControlTokenBranches,
    TrainStepOutputs,
    VSTFlowMatchingModule,
    _TimeField,
    build_guided_velocity,
)

logger = logging.getLogger(__name__)

type ControlMode = Literal["gradient_spectral", "learned_audio", "null"]

_FROZEN_BACKBONE_PREFIX = "encoder.backbone."
_BATCH_PARAMS_SHAPE = "batch params"
_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_TIME_SHAPE = "batch 1"
_BATCH_ANY_SHAPE = "batch ..."
_BATCH_SHAPE = "batch"


@jaxtyped(typechecker=beartype)
def _validate_arm(
    *,
    control_mode: ControlMode,
    cost: torch.nn.Module | None,
    control_encoder: torch.nn.Module | None,
    audio_loss: object | None,
    rectified_sigma_min: float,
    sketch_controls: object | None,
    compiled: bool,
) -> None:
    """Reject, before anything is built, a configuration this module cannot serve.

    :param control_mode: Control arm the run selected.
    :param cost: Differentiable per-sample audio distance, if configured.
    :param control_encoder: Trainable encoder over the render residual, if configured.
    :param audio_loss: The base module's audio term, which this module cannot also carry.
    :param rectified_sigma_min: Probability-path noise scale; only zero leaves the one-step
        estimate exact.
    :param sketch_controls: Sketch-control spec, which the controlled field cannot route.
    :param compiled: Whether the run asked for ``torch.compile``.
    :raises ValueError: Any of those conditions holds.
    """
    if audio_loss is not None:
        raise ValueError(
            "audio_loss cannot be combined with simulator-feedback control; the audio term "
            "enters the objective while the control enters only as an input, and running "
            "both attributes the same simulator cost twice"
        )
    if control_mode == "gradient_spectral" and cost is None:
        raise ValueError("control_mode='gradient_spectral' requires a differentiable cost")
    if control_mode == "learned_audio" and control_encoder is None:
        raise ValueError("control_mode='learned_audio' requires a control_encoder")
    if rectified_sigma_min != 0.0:
        # theta_hat = x_t + (1 - t) * v is the exact one-step estimate only on the
        # sigma-free path; any other sigma renders parameters the flow did not imply.
        raise ValueError(
            f"simulator feedback requires rectified_sigma_min=0, got {rectified_sigma_min}"
        )
    if sketch_controls is not None:
        # ControlledFlow takes no control_tokens, so a sketch spec would train the frozen
        # field under conditioning the base never saw and then fail at validation.
        raise ValueError(
            "simulator feedback does not support sketch controls; the controlled field "
            "cannot route control tokens"
        )
    if compiled:
        from synth_setter.models.components.audio_feedback import (
            validate_audio_feedback_runtime,
        )

        # Checked here rather than only in on_train_start, which runs after setup() has
        # already compiled the field (#2585).
        validate_audio_feedback_runtime(compiled=True, world_size=1)


class VSTFlowFinetuneModule(VSTFlowMatchingModule):
    """Pretrained flow whose velocity a simulator-fed control network learns to correct."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        encoder: torch.nn.Module,
        vector_field: torch.nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., object] | None,
        *,
        base_checkpoint: str | Path,
        num_params: int,
        sample_rate: int,
        signal_length: int,
        render_batch_size: int,
        control_mode: ControlMode = "gradient_spectral",
        control_hidden_dim: int = 256,
        control_t_min: float = DEFAULT_CONTROL_T_MIN,
        cost: torch.nn.Module | None = None,
        control_encoder: torch.nn.Module | None = None,
        **base_kwargs: object,
    ) -> None:
        r"""Load a pretrained flow, freeze it, and attach the control this run trains.

        :param encoder: Conditioning encoder of the same shape the base run trained.
        :param vector_field: Velocity field of the same shape the base run trained.
        :param optimizer: ``functools.partial``-style optimizer factory.
        :param scheduler: ``functools.partial``-style scheduler factory or ``None``.
        :param base_checkpoint: Checkpoint holding the pretrained flow to refine.
        :param num_params: Parameter-vector width the field operates on.
        :param sample_rate: Render sample rate in Hz.
        :param signal_length: Rendered samples per row; must match the target audio.
        :param render_batch_size: Rows the renderer's voice holds.
        :param control_mode: ``gradient_spectral`` for the paper's equation 9,
            ``learned_audio`` for equation 10, or ``null`` for the capacity-matched
            ablation that trains the same control on a zeroed signal.
        :param control_hidden_dim: Hidden width of the control network.
        :param control_t_min: Flow time above which the control engages.
        :param cost: Differentiable per-sample audio distance; required by
            ``gradient_spectral``.
        :param control_encoder: Trainable waveform encoder over the render residual;
            required by ``learned_audio``.
        :param \*\*base_kwargs: Remaining :class:`VSTFlowMatchingModule` arguments.
        """
        _validate_arm(
            control_mode=control_mode,
            cost=cost,
            control_encoder=control_encoder,
            audio_loss=base_kwargs.get("audio_loss"),
            rectified_sigma_min=float(base_kwargs.get("rectified_sigma_min", 0.0)),  # pyright: ignore[reportArgumentType]
            sketch_controls=base_kwargs.get("sketch_controls"),
            compiled=bool(base_kwargs.get("compile", False)),
        )
        super().__init__(
            encoder=encoder,
            vector_field=vector_field,
            optimizer=optimizer,  # pyright: ignore[reportArgumentType]
            scheduler=scheduler,  # pyright: ignore[reportArgumentType]
            num_params=num_params,
            **base_kwargs,  # pyright: ignore[reportArgumentType]
        )
        # Lightning collects the subclass frame's init args, so these land in hparams and
        # get deep-copied; the group admits large weight-normalized pretrained encoders.
        self.save_hyperparameters(ignore=["cost", "control_encoder"], logger=False)
        self.num_params = num_params
        self._load_pretrained(base_checkpoint)
        self.requires_grad_(False)

        self.control_mode: ControlMode = control_mode
        if cost is not None:
            cost.requires_grad_(False)
        self.cost = cost
        self.control_encoder = control_encoder
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.render_batch_size = render_batch_size
        self.control_dim = self._control_signal_width()
        self.vector_field = ControlledFlow(
            flow=self.vector_field,
            control=ControlNet(
                field_dim=num_params,
                control_dim=self.control_dim,
                hidden_dim=control_hidden_dim,
            ),
            t_min=control_t_min,
        )
        # Bound per batch by the validation/test hooks; sampling outside one cannot score.
        self._sampling_target: Tensor | None = None
        # Lightning does not call train() before the first steps, so the override alone would
        # leave the pretrained modules in nn.Module's default training mode until then.
        self._freeze_pretrained_modes()

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> VSTFlowFinetuneModule:
        """Enter the given mode, holding every pretrained module in eval.

        Clearing ``requires_grad`` freezes weights but not normalisation running statistics,
        which keep updating in train mode. Left alone they drift the conditioning the frozen
        field sees over the course of a finetune, confounding the arms this module compares.
        The conditioning dropout CFG needs is applied explicitly, not by an ``nn.Dropout``,
        so eval here does not disable it.

        :param mode: Whether the trainable control enters training mode.
        :returns: This module.
        """
        super().train(mode)
        self._freeze_pretrained_modes()
        return self

    @jaxtyped(typechecker=beartype)
    def _freeze_pretrained_modes(self) -> None:
        """Hold the pretrained encoder and field in eval mode."""
        self.encoder.eval()
        self.vector_field.flow.eval()

    @jaxtyped(typechecker=beartype)
    def _load_pretrained(self, checkpoint: str | Path) -> None:
        """Restore every pretrained weight, refusing a checkpoint that does not fit.

        Runs before the control is attached, so the module's own shape is exactly the base
        run's: any missing or unexpected key means the wrong checkpoint, and a silent
        ``strict=False`` here would "finetune" a randomly initialised field.

        :param checkpoint: Path to a Lightning checkpoint of the base run.
        :raises ValueError: The payload has no ``state_dict``, or its keys do not match.
        """
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        # The config records a mutable path, so without this two arms started from
        # different flows would still read as comparable runs.
        logger.info("base_checkpoint path=%s sha256=%s", checkpoint, digest)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise ValueError(f"{checkpoint} holds no Lightning state_dict")
        result = self.load_state_dict(state, strict=False)
        # A frozen pretrained backbone is stripped on save and re-resolved from its own
        # weights, so its absence is expected; nothing else may be.
        missing = [k for k in result.missing_keys if not k.startswith(_FROZEN_BACKBONE_PREFIX)]
        if missing or result.unexpected_keys:
            raise ValueError(
                f"{checkpoint} does not match this model: "
                f"{len(missing)} missing key(s) {missing[:5]}, "
                f"{len(result.unexpected_keys)} unexpected key(s) {result.unexpected_keys[:5]}"
            )

    @jaxtyped(typechecker=beartype)
    def _control_signal_width(self) -> int:
        """Report the control signal's width for the configured arm.

        The learned arm's width comes from one silent forward rather than from config, because the
        encoder group's members do not agree on how they name their output width.

        :returns: Width of the vector the control network consumes.
        """
        if self.control_mode != "learned_audio":
            # The gradient signal is the scalar cost beside its per-parameter gradient; the
            # null arm matches that width so the two arms differ only in signal content.
            return 1 + self.num_params
        assert self.control_encoder is not None
        # eval for the probe: no_grad suppresses the graph but not normalization
        # bookkeeping, so a train-mode forward folds this all-zeros waveform into the
        # encoder's running statistics before training has seen a real batch.
        was_training = self.control_encoder.training
        self.control_encoder.eval()
        try:
            with torch.no_grad():
                probe = self.control_encoder(torch.zeros(1, self.signal_length))
        finally:
            self.control_encoder.train(was_training)
        return int(probe.flatten(start_dim=1).shape[-1])

    @jaxtyped(typechecker=beartype)
    def on_train_start(self) -> None:
        """Reject runtimes the differentiable renderer cannot serve (#2585).

        Applied to the null arm too, which renders nothing: an ablation that ran under a different
        compile or device setting than the arms it bounds would confound them.
        """
        from synth_setter.models.components.audio_feedback import (
            validate_audio_feedback_runtime,
        )

        validate_audio_feedback_runtime(
            compiled=bool(self.hparams["compile"]),
            world_size=self.trainer.world_size,
        )

    @jaxtyped(typechecker=beartype)
    def _render(
        self, theta_hat: Float[Tensor, _BATCH_PARAMS_SHAPE]
    ) -> Float[Tensor, _BATCH_AUDIO_SHAPE]:
        """Render a model-space estimate through the production differentiable renderer.

        The decode is what makes the estimate mean what it says: the renderer reads ``[0, 1]``
        and clamps, so feeding it model-space ``[-1, 1]`` directly still produces audio — just
        not the audio those parameters describe.

        :param theta_hat: One-step estimate in model space ``[-1, 1]``.
        :returns: Audio shaped ``(batch, signal_length)``.
        """
        return render_torchsynth_grad(
            differentiable_decode(theta_hat),
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            render_batch_size=self.render_batch_size,
        )

    @jaxtyped(typechecker=beartype)
    def _control_signal(
        self,
        theta_hat: Float[Tensor, _BATCH_PARAMS_SHAPE],
        target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
        active: Shaped[Tensor, _BATCH_SHAPE],
    ) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Score the one-step estimate against the observation, on the rows that use it.

        Only ``active`` rows are rendered. At the shipped ``control_t_min`` the control engages
        on a minority of rows, and the render is this loop's dominant cost, so scoring the whole
        batch would spend most of the step on a signal :meth:`ControlledFlow.combine` discards.

        :param theta_hat: One-step parameter estimate in model space, detached from the flow.
        :param target_audio: Observed audio shaped ``(batch, signal_length)``.
        :param active: Rows whose signal is used; the rest come back zeroed.
        :returns: Control signal shaped ``(batch, control_dim)``.
        """
        signal = torch.zeros(
            len(theta_hat), self.control_dim, device=theta_hat.device, dtype=theta_hat.dtype
        )
        # The null arm's zeroed signal is the capacity-matched ablation: same network, same
        # velocity, only the simulator's contribution removed.
        if self.control_mode == "null" or not bool(active.any()):
            return signal
        rows = active.nonzero(as_tuple=True)[0]
        scored = self._score(theta_hat[rows], target_audio[rows])
        return signal.index_copy(0, rows, scored.to(signal.dtype))

    @jaxtyped(typechecker=beartype)
    def _score(
        self,
        theta_hat: Float[Tensor, _BATCH_PARAMS_SHAPE],
        target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Run the configured arm's simulator scoring over a batch of rows.

        :param theta_hat: One-step parameter estimate in model space.
        :param target_audio: Observed audio for those rows.
        :returns: Control signal shaped ``(rows, control_dim)``.
        """
        # __init__ rejects an arm without its component, so a None here is a torn object.
        if self.control_mode == "gradient_spectral":
            assert self.cost is not None
            return gradient_control_signal(
                theta_hat=theta_hat.requires_grad_(True),
                target_audio=target_audio,
                render=self._render,
                cost=self.cost,
            )
        assert self.control_encoder is not None
        return learned_control_signal(
            theta_hat=theta_hat,
            target_audio=target_audio,
            render=self._render,
            encoder=self.control_encoder,
        )

    @jaxtyped(typechecker=beartype)
    def _one_step_estimate(
        self,
        x_t: Float[Tensor, _BATCH_PARAMS_SHAPE],
        t: Float[Tensor, _BATCH_TIME_SHAPE],
        velocity: Float[Tensor, _BATCH_PARAMS_SHAPE],
    ) -> Float[Tensor, _BATCH_PARAMS_SHAPE]:
        """Project the trajectory point to t=1 along the current velocity.

        Detached because the estimate exists to be scored, not trained through: a live graph
        here would make the finetune second-order in the frozen field. The gradient arm
        re-seeds ``requires_grad`` on the result to differentiate the render against it.

        :param x_t: Trajectory point shaped ``(batch, params)``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param velocity: Pretrained velocity at ``(x_t, t)``.
        :returns: Detached estimate of ``x_1`` shaped ``(batch, params)``.
        """
        return (x_t + (1 - t) * velocity).detach()

    @jaxtyped(typechecker=beartype)
    def _velocity_field(
        self,
        conditioning: Shaped[Tensor, _BATCH_ANY_SHAPE] | None,
        cfg_strength: float,
        control_tokens: ControlTokenBranches | None,
    ) -> _TimeField:
        """Build a sampling field whose velocity carries simulator feedback above ``t_min``.

        The control is scored on, and fed, the **conditional** velocity, because that is what
        it saw in training; the guided combination is roughly ``cfg_strength`` times larger
        and out of distribution for it, which would let evaluation metrics move for reasons
        unrelated to the learned correction. The resulting correction is then applied to the
        guided velocity, which is what the sampler integrates. This costs one extra field
        evaluation per ODE evaluation and leaves the correction computed for a trajectory
        point the sampler does not visit — both tracked in
        https://github.com/tinaudio/synth-setter/issues/2782.

        :param conditioning: Encoded content conditioning for the conditional branch.
        :param cfg_strength: Joint classifier-free-guidance scale.
        :param control_tokens: Complete control-token state; rejected at construction.
        :returns: Two-argument velocity field over parameter state and time.
        :raises RuntimeError: No observation is bound, which would silently sample the
            frozen base and report it as this arm's result.
        """
        if self._sampling_target is None:
            raise RuntimeError(
                "no observation bound for controlled sampling; _sample must run inside a "
                "validation or test batch so the estimate has something to be scored against"
            )
        target_audio = self._sampling_target
        guided = build_guided_velocity(
            self.vector_field.flow, conditioning, cfg_strength, control_tokens=control_tokens
        )

        @jaxtyped(typechecker=beartype)
        def controlled(
            x_t: Float[Tensor, _BATCH_PARAMS_SHAPE], t: Float[Tensor, _BATCH_TIME_SHAPE]
        ) -> Float[Tensor, _BATCH_PARAMS_SHAPE]:
            """Evaluate the guided velocity and fold in this evaluation's feedback.

            :param x_t: Trajectory point shaped ``(batch, params)``.
            :param t: Flow time shaped ``(batch, 1)``.
            :returns: Corrected velocity shaped ``(batch, params)``.
            """
            guided_velocity = guided(x_t, t)
            conditional = self.vector_field.flow(x_t, t, conditioning)
            signal = self._sampling_control(x_t, t, conditional, target_audio)
            # combine() minus its input is the gated correction, so a disengaged row
            # contributes exactly zero and leaves the guided velocity bit-identical.
            correction = self.vector_field.combine(conditional, t, signal) - conditional
            return guided_velocity + correction

        return controlled

    @jaxtyped(typechecker=beartype)
    def _sampling_control(
        self,
        x_t: Float[Tensor, _BATCH_PARAMS_SHAPE],
        t: Float[Tensor, _BATCH_TIME_SHAPE],
        velocity: Float[Tensor, _BATCH_PARAMS_SHAPE],
        target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Score the one-step estimate at one ODE evaluation, on engaged rows only.

        No conditioning-keep term here: dropout is a training-time device, so at sampling
        every row's estimate is tied to its own observation.

        :param x_t: Trajectory point shaped ``(batch, params)``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param velocity: Guided velocity at ``(x_t, t)``.
        :param target_audio: Observation shaped ``(batch, signal_length)``.
        :returns: Control signal shaped ``(batch, control_dim)``.
        """
        active = t.squeeze(-1) >= self.vector_field.t_min
        # Standalone Trainer.validate/test/predict run under inference_mode, which marks
        # every tensor made inside it as an inference tensor that can never carry a graph —
        # torch.enable_grad() cannot reopen it, so the gradient arm's autograd.grad raises.
        # Mid-fit validation happens to work because Lightning builds that loop with
        # inference_mode=False, which is why this only shows up outside a fit.
        with torch.inference_mode(False):
            estimate = self._one_step_estimate(x_t, t, velocity).clone()
            return self._control_signal(estimate, target_audio.clone(), active)

    @jaxtyped(typechecker=beartype)
    def on_validation_batch_start(
        self,
        batch: dict[str, Shaped[Tensor, _BATCH_ANY_SHAPE]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Bind this batch's observation so the sampler can score estimates against it.

        :param batch: Validation batch carrying the observation.
        :param batch_idx: Lightning's batch index.
        :param dataloader_idx: Lightning's dataloader index.
        """
        self._sampling_target = batch["audio"]

    @jaxtyped(typechecker=beartype)
    def on_validation_batch_end(
        self,
        outputs: object,
        batch: dict[str, Shaped[Tensor, _BATCH_ANY_SHAPE]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Release the observation so a stale target cannot leak into the next batch.

        :param outputs: Whatever ``validation_step`` returned.
        :param batch: The batch just finished.
        :param batch_idx: Lightning's batch index.
        :param dataloader_idx: Lightning's dataloader index.
        """
        self._sampling_target = None

    @jaxtyped(typechecker=beartype)
    def on_predict_batch_start(
        self,
        batch: dict[str, Shaped[Tensor, _BATCH_ANY_SHAPE]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Bind this batch's observation for the prediction sampler.

        The predict split always projects the audio column, so prediction can score against the
        same observation validation does rather than falling back to the frozen base.

        :param batch: Predict batch carrying the observation.
        :param batch_idx: Lightning's batch index.
        :param dataloader_idx: Lightning's dataloader index.
        """
        self._sampling_target = batch["audio"]

    @jaxtyped(typechecker=beartype)
    def on_predict_batch_end(
        self,
        outputs: object,
        batch: dict[str, Shaped[Tensor, _BATCH_ANY_SHAPE]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Release the observation bound for the prediction sampler.

        :param outputs: Whatever ``predict_step`` returned.
        :param batch: The batch just finished.
        :param batch_idx: Lightning's batch index.
        :param dataloader_idx: Lightning's dataloader index.
        """
        self._sampling_target = None

    @jaxtyped(typechecker=beartype)
    def on_test_batch_start(
        self,
        batch: dict[str, Shaped[Tensor, _BATCH_ANY_SHAPE]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Bind this batch's observation for the test sampler.

        :param batch: Test batch carrying the observation.
        :param batch_idx: Lightning's batch index.
        :param dataloader_idx: Lightning's dataloader index.
        """
        self._sampling_target = batch["audio"]

    @jaxtyped(typechecker=beartype)
    def on_test_batch_end(
        self,
        outputs: object,
        batch: dict[str, Shaped[Tensor, _BATCH_ANY_SHAPE]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Release the observation bound for the test sampler.

        :param outputs: Whatever ``test_step`` returned.
        :param batch: The batch just finished.
        :param batch_idx: Lightning's batch index.
        :param dataloader_idx: Lightning's dataloader index.
        """
        self._sampling_target = None

    @jaxtyped(typechecker=beartype)
    def _log_control_telemetry(
        self,
        control_input: Float[Tensor, _BATCH_ANY_SHAPE],
        active: Shaped[Tensor, _BATCH_SHAPE],
    ) -> None:
        """Log whole-batch control activity and signal magnitudes when attached.

        :param control_input: Sanitized, gated signal handed to the control network.
        :param active: Rows whose control gate is active.
        """
        if self._trainer is None:
            return
        metrics = {
            "train/control_active_fraction": active.to(dtype=control_input.dtype).mean(),
            "train/control_signal_norm": control_input.norm(dim=-1).mean(),
        }
        if self.control_mode == "gradient_spectral":
            metrics.update(
                {
                    "train/control_cost": control_input[:, 0].abs().mean(),
                    "train/control_grad_norm": control_input[:, 1:].norm(dim=-1).mean(),
                }
            )
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=int(control_input.shape[0]),
        )

    @jaxtyped(typechecker=beartype)
    def _train_step(self, batch: dict[str, Shaped[Tensor, _BATCH_ANY_SHAPE]]) -> TrainStepOutputs:
        """Run one finetuning step: estimate, score, correct, and match the flow.

        :param batch: Batch carrying params, noise, and target audio.
        :returns: The conditional flow-matching loss over the corrected velocity.
        """
        params = batch["params"]
        z, _, conditioning_keep = self._prepare_conditioning(batch)  # pyright: ignore[reportArgumentType]

        with torch.no_grad():
            t = self._sample_time(params.shape[0], params.device)
            w = self._weight_time(t)
            x_t = self._sample_probability_path(batch["noise"], params, t)
            target = self._evaluate_target_field(batch["noise"], params, x_t, t)
            velocity = self.vector_field.flow(x_t, t, z)

        # A row is scored only if its signal can reach the correction (above t_min) and
        # means anything (its estimate is conditioned on its own audio). A fully
        # unconditional row estimates the marginal, so its render/target residual is
        # high-variance noise rather than identity signal — the base module gates its audio
        # term on the same mask, and leaving it ungated here would teach each arm a
        # different amount of noise and confound the comparison between them.
        active = (t.squeeze(-1) >= self.vector_field.t_min) & conditioning_keep.identity_keep
        control_input = self._control_signal(
            self._one_step_estimate(x_t, t, velocity), batch["audio"], active
        )
        self._log_control_telemetry(control_input, active)
        prediction = self.vector_field.combine(velocity, t, control_input)

        loss = ((prediction - target).square().mean(dim=-1) * w).mean()
        return TrainStepOutputs(
            loss=loss,
            audio_term=None,
            penalty=None,
            grad_balance=None,
            t=t,
            conditioning_keep=conditioning_keep,
        )
