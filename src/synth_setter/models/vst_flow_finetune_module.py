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
    TrainStepOutputs,
    VSTFlowMatchingModule,
)

type ControlMode = Literal["gradient_spectral", "learned_audio", "null"]

_FROZEN_BACKBONE_PREFIX = "encoder.backbone."
_BATCH_PARAMS_SHAPE = "batch params"
_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_TIME_SHAPE = "batch 1"
_BATCH_ANY_SHAPE = "batch ..."


@jaxtyped(typechecker=beartype)
def _validate_arm(
    *,
    control_mode: ControlMode,
    cost: torch.nn.Module | None,
    control_encoder: torch.nn.Module | None,
    audio_loss: object | None,
) -> None:
    """Reject an arm whose required component is absent before anything is built.

    :param control_mode: Control arm the run selected.
    :param cost: Differentiable per-sample audio distance, if configured.
    :param control_encoder: Trainable encoder over the render residual, if configured.
    :param audio_loss: The base module's audio term, which this module cannot also carry.
    :raises ValueError: The arm is missing its component, or an audio term was supplied.
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
        )
        super().__init__(
            encoder=encoder,
            vector_field=vector_field,
            optimizer=optimizer,  # pyright: ignore[reportArgumentType]
            scheduler=scheduler,  # pyright: ignore[reportArgumentType]
            num_params=num_params,
            **base_kwargs,  # pyright: ignore[reportArgumentType]
        )
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

    @jaxtyped(typechecker=beartype)
    def _load_pretrained(self, checkpoint: str | Path) -> None:
        """Restore every pretrained weight, refusing a checkpoint that does not fit.

        Runs before the control is attached, so the module's own shape is exactly the base
        run's: any missing or unexpected key means the wrong checkpoint, and a silent
        ``strict=False`` here would "finetune" a randomly initialised field.

        :param checkpoint: Path to a Lightning checkpoint of the base run.
        :raises ValueError: The payload has no ``state_dict``, or its keys do not match.
        """
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
        if self.control_encoder is None:
            # The gradient signal is the scalar cost beside its per-parameter gradient; the
            # null arm matches that width so the two arms differ only in signal content.
            return 1 + self.num_params
        with torch.no_grad():
            probe = self.control_encoder(torch.zeros(1, self.signal_length))
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
    ) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Score the one-step estimate against the observation for the configured arm.

        :param theta_hat: One-step parameter estimate in model space, detached from the flow.
        :param target_audio: Observed audio shaped ``(batch, signal_length)``.
        :returns: Control signal shaped ``(batch, control_dim)``.
        """
        if self.control_mode == "null":
            # Same control network on the same velocity, with the simulator's contribution
            # zeroed: the capacity-matched arm that isolates what the feedback itself buys.
            return torch.zeros(
                len(theta_hat), self.control_dim, device=theta_hat.device, dtype=theta_hat.dtype
            )
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

        control_input = self._control_signal(
            self._one_step_estimate(x_t, t, velocity), batch["audio"]
        )
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
