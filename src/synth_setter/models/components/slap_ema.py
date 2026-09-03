"""Moving-average target updates from the SLAP reference implementation.

Source: Pliploop/SLAP commit b49290186ee354d34798f9947110a375f9e3f5a7.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sized
from typing import Literal, cast

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from lightning.pytorch import LightningModule, Trainer
from torch import nn


class MovingAverageWeightUpdate:
    """Update target parameters using the reference SLAP schedule."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        initial_tau: float = 0.996,
        final_tau: float = 1.0,
        every_n_steps: int = 1,
        update_method: Literal["cos", "exp", "lin"] = "cos",
    ) -> None:
        """Configure the target update cadence and tau schedule.

        :param initial_tau: Target retention at the start of training.
        :param final_tau: Target retention at the end of training.
        :param every_n_steps: Training-batch interval between target updates.
        :param update_method: Schedule used to move from initial to final tau.
        :raises ValueError: If ``update_method`` is unsupported.
        """
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.every_n_steps = every_n_steps
        self.current_tau = initial_tau
        methods: dict[str, Callable[[LightningModule, Trainer], float]] = {
            "cos": self.update_tau_cos,
            "exp": self.update_tau_exp,
            "lin": self.update_tau_lin,
        }
        try:
            self.update_tau = methods[update_method]
        except KeyError as exc:
            raise ValueError(f"Unknown update method {update_method}") from exc

    @jaxtyped(typechecker=beartype)
    def on_train_batch_end(
        self,
        trainer: Trainer,
        module: LightningModule,
        batch_idx: int,
    ) -> None:
        """Update both target arms at the configured batch interval.

        :param trainer: Active trainer supplying schedule bounds.
        :param module: SLAP module carrying online and target arms.
        :param batch_idx: Zero-based batch position in the current epoch.
        """
        if batch_idx % self.every_n_steps != 0:
            return
        self.update_weights(
            cast(nn.Module, getattr(module, "audio_encoder")),
            cast(nn.Module, getattr(module, "audio_ema")),
        )
        self.update_weights(
            cast(nn.Module, getattr(module, "text_encoder")),
            cast(nn.Module, getattr(module, "text_ema")),
        )
        module.log("MA rate", self.current_tau, prog_bar=False, logger=True)
        self.current_tau = self.update_tau(module, trainer)

    @jaxtyped(typechecker=beartype)
    def update_tau_cos(self, module: LightningModule, trainer: Trainer) -> float:
        """Return the cosine moving-average schedule value.

        :param module: Module supplying the global optimizer step.
        :param trainer: Trainer supplying dataloader and epoch bounds.
        :returns: Target retention for the next update.
        """
        max_steps = len(cast(Sized, trainer.train_dataloader)) * cast(int, trainer.max_epochs)
        phase = math.pi * module.global_step / max_steps
        return self.final_tau - (self.final_tau - self.initial_tau) * (math.cos(phase) + 1) / 2

    @jaxtyped(typechecker=beartype)
    def update_tau_exp(self, module: LightningModule, trainer: Trainer) -> float:
        """Return the exponential moving-average schedule value.

        :param module: Module supplying the global optimizer step.
        :param trainer: Trainer supplying the dataloader length.
        :returns: Target retention for the next update.
        """
        half_life = len(cast(Sized, trainer.train_dataloader)) * self.final_tau
        return 1 - self.initial_tau * 2 ** (-trainer.global_step / half_life)

    @jaxtyped(typechecker=beartype)
    def update_tau_lin(self, module: LightningModule, trainer: Trainer) -> float:
        """Return the linear moving-average schedule value.

        :param module: Module supplying the global optimizer step.
        :param trainer: Trainer supplying dataloader and epoch bounds.
        :returns: Target retention for the next update.
        """
        max_steps = len(cast(Sized, trainer.train_dataloader)) * cast(int, trainer.max_epochs)
        return self.initial_tau + (
            (self.final_tau - self.initial_tau) * module.global_step / max_steps
        )

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def update_weights(self, online: nn.Module, target: nn.Module) -> None:
        """Move target parameters toward the corresponding online parameters.

        :param online: Gradient-trained arm supplying current parameters.
        :param target: Moving-average arm updated in place.
        """
        for (_, online_parameter), (_, target_parameter) in zip(
            online.named_parameters(), target.named_parameters(), strict=True
        ):
            target_parameter.data = (
                self.current_tau * target_parameter.data
                + (1 - self.current_tau) * online_parameter.data
            )
