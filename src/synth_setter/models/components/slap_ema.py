"""Moving-average target updates for SLAP training.

Source: Pliploop/SLAP commit b49290186ee354d34798f9947110a375f9e3f5a7.

Typical usage:
    update = MovingAverageWeightUpdate(every_n_steps=2)
    update.on_optimizer_step(trainer, slap_module, completed_step=trainer.global_step + 1)
"""

from __future__ import annotations

import math
from typing import Literal, cast

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from lightning.pytorch import LightningModule, Trainer
from torch import nn


class MovingAverageWeightUpdate:
    """Update target arms after effective optimizer steps."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        initial_tau: float = 0.996,
        final_tau: float = 1.0,
        every_n_steps: int = 1,
        update_method: Literal["cos", "exp", "lin"] = "cos",
    ) -> None:
        """Configure optimizer-step cadence and target-retention schedule.

        :param initial_tau: Target retention before optimization begins.
        :param final_tau: Target retention after all estimated optimizer steps.
        :param every_n_steps: Effective optimizer-step interval between updates.
        :param update_method: Interpolation from initial to final retention.
        :raises ValueError: If the cadence or interpolation method is invalid.
        """
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        if update_method not in {"cos", "exp", "lin"}:
            raise ValueError(f"Unknown update method {update_method}")
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.every_n_steps = every_n_steps
        self.update_method = update_method

    @jaxtyped(typechecker=beartype)
    def tau_at_step(self, completed_step: int, total_steps: int) -> float:
        """Return retention derived from completed optimizer-step progress.

        :param completed_step: Number of optimizer steps already completed.
        :param total_steps: Estimated optimizer steps in the complete fit.
        :returns: Scheduled retention, clamped to ``final_tau`` at completion.
        :raises ValueError: If ``total_steps`` is not positive.
        """
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if completed_step >= total_steps:
            return self.final_tau
        progress = max(0.0, completed_step / total_steps)
        distance = self.final_tau - self.initial_tau
        if self.update_method == "lin":
            return self.initial_tau + distance * progress
        if self.update_method == "cos":
            interpolation = (1 - math.cos(math.pi * progress)) / 2
            return self.initial_tau + distance * interpolation
        interpolation = (1 - 2 ** (-progress)) / (1 - 2**-1)
        return self.initial_tau + distance * interpolation

    @jaxtyped(typechecker=beartype)
    def on_optimizer_step(
        self,
        trainer: Trainer,
        module: LightningModule,
        completed_step: int,
    ) -> None:
        """Update target arms when a configured optimizer-step boundary is reached.

        :param trainer: Active trainer supplying the estimated step count.
        :param module: SLAP module carrying online and target arms.
        :param completed_step: Number of optimizer steps completed, including this one.
        """
        if completed_step % self.every_n_steps != 0:
            return
        total_steps = int(trainer.estimated_stepping_batches)
        tau = self.tau_at_step(completed_step, total_steps)
        self.update_weights(
            cast(nn.Module, getattr(module, "audio_encoder")),
            cast(nn.Module, getattr(module, "audio_ema")),
            tau=tau,
        )
        self.update_weights(
            cast(nn.Module, getattr(module, "text_encoder")),
            cast(nn.Module, getattr(module, "text_ema")),
            tau=tau,
        )
        module.log("MA rate", tau, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def update_weights(self, online: nn.Module, target: nn.Module, *, tau: float) -> None:
        """EMA floating state and copy integral buffers from online to target.

        :param online: Gradient-trained arm supplying parameters and buffers.
        :param target: Moving-average arm updated in place.
        :param tau: Retention applied to floating-point state.
        :raises ValueError: If target state has no corresponding online state.
        """
        online_parameters = dict(online.named_parameters())
        target_parameters = dict(target.named_parameters())
        online_buffers = dict(online.named_buffers())
        target_buffers = dict(target.named_buffers())
        has_unknown_state = not target_parameters.keys() <= online_parameters.keys() or not (
            target_buffers.keys() <= online_buffers.keys()
        )
        if has_unknown_state:
            raise ValueError("target state names must exist in online state")

        for name, target_parameter in target_parameters.items():
            target_parameter.lerp_(online_parameters[name], 1 - tau)
        for name, target_buffer in target_buffers.items():
            online_buffer = online_buffers[name]
            if target_buffer.is_floating_point() or target_buffer.is_complex():
                target_buffer.lerp_(online_buffer, 1 - tau)
            else:
                target_buffer.copy_(online_buffer)
