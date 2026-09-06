"""Reinforce Adjoint Matching post-training of a pretrained flow (arXiv 2605.10759).

A frozen copy of the pretrained field anchors the regression, a lagged EMA copy draws the
on-policy endpoints, and a black-box reward scores them. Each step samples
``num_samples_per_row`` endpoints per target audio, normalises their rewards within the
group, noises every endpoint ``num_targets_per_sample`` times, and regresses the policy
onto the RAM target; no reward gradient and no SDE rollout are involved.

Typical usage:
    trainer.fit(VSTFlowRAMModule(..., base_checkpoint=..., reward=RenderedAudioReward(...)))
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor

from synth_setter.models.components.pretrained_flow import load_pretrained_flow
from synth_setter.models.components.ram import (
    group_relative_advantages,
    power_law_flow_time,
    ram_velocity_target,
)
from synth_setter.models.vst_flow_matching_module import (
    VSTFlowMatchingModule,
    build_guided_velocity,
    integrate_flow,
)

_BATCH_ANY_SHAPE = "batch ..."
_BATCH_PARAMS_SHAPE = "batch params"
_BATCH_TIME_SHAPE = "batch 1"
_SCALAR_SHAPE = ""


@jaxtyped(typechecker=beartype)
def _validate_ram_settings(
    *,
    num_samples_per_row: int,
    num_targets_per_sample: int,
    reward_multiplier: float,
    sampling_steps: int,
    ema_decay: float,
    ema_warmup_rate: float | None,
    time_power_law_alpha: float,
    base_kwargs: dict[str, object],
) -> None:
    """Reject settings under which the RAM loss is undefined or degenerate.

    :param num_samples_per_row: Endpoints sampled per conditioning row.
    :param num_targets_per_sample: Noise draws per endpoint.
    :param reward_multiplier: Scale applied to normalised advantages.
    :param sampling_steps: RK4 steps used to draw endpoints.
    :param ema_decay: Lag of the sampling copy toward the policy.
    :param ema_warmup_rate: Per-step ramp of the lag, or ``None`` for a fixed lag.
    :param time_power_law_alpha: Exponent of the flow-time law.
    :param base_kwargs: Remaining :class:`VSTFlowMatchingModule` arguments.
    :raises ValueError: Any setting is out of range, or the base run carries a term the RAM
        loss cannot combine with.
    """
    if num_samples_per_row < 2:
        raise ValueError(
            f"num_samples_per_row must be at least 2 for a group-relative advantage, "
            f"got {num_samples_per_row}"
        )
    if num_targets_per_sample < 1:
        raise ValueError(f"num_targets_per_sample must be positive, got {num_targets_per_sample}")
    if not math.isfinite(reward_multiplier) or reward_multiplier <= 0.0:
        raise ValueError(f"reward_multiplier must be finite and positive, got {reward_multiplier}")
    if sampling_steps < 1:
        raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError(f"ema_decay must lie in [0, 1), got {ema_decay}")
    if ema_warmup_rate is not None and ema_warmup_rate <= 0.0:
        raise ValueError(f"ema_warmup_rate must be positive or None, got {ema_warmup_rate}")
    if time_power_law_alpha < 0.0:
        raise ValueError(f"time_power_law_alpha must be non-negative, got {time_power_law_alpha}")
    if base_kwargs.get("audio_loss") is not None:
        raise ValueError(
            "audio_loss cannot be combined with RAM; the reward carries the audio term"
        )
    if base_kwargs.get("sketch_controls") is not None:
        raise ValueError("sketch_controls are not supported by RAM post-training")
    if float(base_kwargs.get("rectified_sigma_min", 0.0)) != 0.0:  # pyright: ignore[reportArgumentType]
        raise ValueError(
            "RAM's reward direction assumes the sigma-free path; set rectified_sigma_min=0"
        )
    if base_kwargs.get("compile"):
        # The reward renders through torchsynth, which graph-breaks under compile (#2585).
        raise ValueError("RAM post-training is incompatible with torch.compile; set compile=false")


class VSTFlowRAMModule(VSTFlowMatchingModule):
    """Pretrained flow post-trained toward a black-box reward with the RAM regression."""

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
        reward: torch.nn.Module,
        num_samples_per_row: int = 8,
        num_targets_per_sample: int = 4,
        reward_multiplier: float = 100.0,
        sampling_steps: int = 20,
        sampling_cfg_strength: float = 2.0,
        ema_decay: float = 0.9,
        ema_warmup_rate: float | None = 0.01,
        time_power_law_alpha: float = 1.0,
        **base_kwargs: object,
    ) -> None:
        r"""Load the pretrained flow and clone it into the frozen reference and lagged sampler.

        :param encoder: Conditioning encoder of the same shape the base run trained; frozen.
        :param vector_field: Velocity field of the same shape the base run trained; the policy.
        :param optimizer: ``functools.partial``-style optimizer factory.
        :param scheduler: ``functools.partial``-style scheduler factory or ``None``.
        :param base_checkpoint: Checkpoint holding the pretrained flow to post-train.
        :param num_params: Parameter-vector width the field operates on.
        :param reward: Module mapping ``(sampled_params, target_audio)`` to a per-row reward;
            higher is better, and it is never differentiated.
        :param num_samples_per_row: Endpoints sampled per target row; the advantage group.
        :param num_targets_per_sample: Independent noise draws per endpoint.
        :param reward_multiplier: Scale on the normalised advantage; the paper's reward
            coefficient, which trades reward against staying near the reference.
        :param sampling_steps: RK4 steps for drawing endpoints during training.
        :param sampling_cfg_strength: Content guidance used when drawing endpoints.
        :param ema_decay: Lag of the sampling copy: ``old <- decay * old + (1 - decay) * policy``
            after every optimizer step.
        :param ema_warmup_rate: Ramps the lag as ``min(rate * step, decay)`` so early steps
            track the policy closely; ``None`` applies ``ema_decay`` from the first step.
        :param time_power_law_alpha: Flow times follow density ``(1 - t) ** alpha``; 0 is uniform.
        :param \*\*base_kwargs: Remaining :class:`VSTFlowMatchingModule` arguments.
        """
        _validate_ram_settings(
            num_samples_per_row=num_samples_per_row,
            num_targets_per_sample=num_targets_per_sample,
            reward_multiplier=reward_multiplier,
            sampling_steps=sampling_steps,
            ema_decay=ema_decay,
            ema_warmup_rate=ema_warmup_rate,
            time_power_law_alpha=time_power_law_alpha,
            base_kwargs=base_kwargs,
        )
        super().__init__(
            encoder=encoder,
            vector_field=vector_field,
            optimizer=optimizer,  # pyright: ignore[reportArgumentType]
            scheduler=scheduler,  # pyright: ignore[reportArgumentType]
            num_params=num_params,
            **base_kwargs,  # pyright: ignore[reportArgumentType]
        )
        # Lightning collects this frame's init args too; the base already recorded the encoder
        # under its own rules, and the reward is training-time only.
        self.save_hyperparameters(ignore=["encoder", "reward"], logger=False)
        load_pretrained_flow(self, base_checkpoint)
        self.num_params = num_params
        self.num_samples_per_row = num_samples_per_row
        self.num_targets_per_sample = num_targets_per_sample
        self.reward_multiplier = reward_multiplier
        self.sampling_steps = sampling_steps
        self.sampling_cfg_strength = sampling_cfg_strength
        self.ema_decay = ema_decay
        self.ema_warmup_rate = ema_warmup_rate
        self.time_power_law_alpha = time_power_law_alpha
        self.reward = reward.requires_grad_(False)
        self.encoder.requires_grad_(False)
        self.reference_field = copy.deepcopy(self.vector_field).requires_grad_(False)
        self.old_field = copy.deepcopy(self.vector_field).requires_grad_(False)
        self._freeze_modes()

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> VSTFlowRAMModule:
        """Enter the given mode, holding every frozen module in eval.

        :param mode: Whether the policy enters training mode.
        :returns: This module.
        """
        super().train(mode)
        self._freeze_modes()
        return self

    @jaxtyped(typechecker=beartype)
    def on_train_start(self) -> None:
        """Reject a multi-device fit: the render reward mutates one shared voice (#2585)."""
        from synth_setter.models.components.audio_feedback import (
            validate_audio_feedback_runtime,
        )

        validate_audio_feedback_runtime(compiled=False, world_size=self.trainer.world_size)

    @jaxtyped(typechecker=beartype)
    def _freeze_modes(self) -> None:
        """Hold the encoder, reference, sampler, and reward in eval mode."""
        for module in (self.encoder, self.reference_field, self.old_field, self.reward):
            module.eval()

    @jaxtyped(typechecker=beartype)
    def _sample_time(self, n: int, device: torch.device) -> Float[Tensor, _BATCH_TIME_SHAPE]:
        """Draw flow times under the configured noise-biased power law.

        :param n: Number of rows.
        :param device: Device the times are drawn on.
        :returns: Flow times shaped ``(n, 1)``.
        """
        return power_law_flow_time(torch.rand(n, 1, device=device), self.time_power_law_alpha)

    @jaxtyped(typechecker=beartype)
    def _sample_endpoints(
        self, conditioning: Shaped[Tensor, _BATCH_ANY_SHAPE]
    ) -> Float[Tensor, _BATCH_PARAMS_SHAPE]:
        """Draw one on-policy endpoint per conditioning row from the lagged sampler.

        :param conditioning: Encoded content conditioning, already repeated per sample.
        :returns: Model-space parameter rows.
        """
        velocity = build_guided_velocity(self.old_field, conditioning, self.sampling_cfg_strength)
        noise = torch.randn(conditioning.shape[0], self.num_params, device=conditioning.device)
        return integrate_flow(velocity, noise, self.sampling_steps, warp_time=self._warp_time)

    @jaxtyped(typechecker=beartype)
    def training_step(
        self, batch: dict[str, Shaped[Tensor, _BATCH_ANY_SHAPE]], batch_idx: int
    ) -> Float[Tensor, _SCALAR_SHAPE]:
        """Sample, score, noise, and regress the policy onto the RAM target (algorithm 1).

        :param batch: Batch carrying content conditioning and the target ``audio``.
        :param batch_idx: Unused Lightning batch index.
        :returns: Mean squared error between the policy velocity and the RAM target.
        """
        group = self.num_samples_per_row
        repeats = self.num_targets_per_sample
        with torch.no_grad():
            conditioning = self.encoder(self._get_conditioning_from_batch(batch))  # pyright: ignore[reportArgumentType]
            conditioning = conditioning.repeat_interleave(group, dim=0)
            endpoints = self._sample_endpoints(conditioning)
            rewards = self.reward(endpoints, batch["audio"].repeat_interleave(group, dim=0))
            advantages = self.reward_multiplier * group_relative_advantages(rewards, group)

            conditioning = conditioning.repeat_interleave(repeats, dim=0)
            endpoints = endpoints.repeat_interleave(repeats, dim=0)
            advantages = advantages.repeat_interleave(repeats, dim=0)
            t = self._sample_time(endpoints.shape[0], endpoints.device)
            noise = torch.randn_like(endpoints)
            x_t = self._sample_probability_path(noise, endpoints, t)
            reference = self.reference_field(x_t, t, conditioning)
            target = ram_velocity_target(
                reference=reference,
                old=self.old_field(x_t, t, conditioning),
                endpoint=endpoints,
                noise=noise,
                advantage=advantages,
            )

        velocity = self.vector_field(x_t, t, conditioning)
        loss = (velocity - target).square().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/reward", rewards.mean(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/abs_advantage", advantages.abs().mean(), on_step=True, on_epoch=False)
        # How far the policy has drifted from the anchor; the KL side of the trade-off.
        self.log(
            "train/reference_delta",
            (velocity - reference).square().mean(),
            on_step=True,
            on_epoch=False,
        )
        return loss

    @jaxtyped(typechecker=beartype)
    def optimizer_step(self, *args: object, **kwargs: object) -> None:
        r"""Step the optimizer, then lag the sampling copy toward the updated policy.

        :param \*args: Lightning's positional ``optimizer_step`` arguments.
        :param \*\*kwargs: Lightning's keyword ``optimizer_step`` arguments.
        """
        super().optimizer_step(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        self._update_old_field()

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def _update_old_field(self) -> None:
        """Move the lagged sampler toward the policy with the (possibly warmed-up) decay."""
        rate = self.ema_warmup_rate
        mix = self.ema_decay if rate is None else min(rate * self.global_step, self.ema_decay)
        for source, target in zip(
            self.vector_field.parameters(), self.old_field.parameters(), strict=True
        ):
            target.mul_(mix).add_(source.detach(), alpha=1.0 - mix)
