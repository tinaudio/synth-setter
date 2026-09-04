"""Lightning training module for Siamese Language-Audio Pretraining.

Source: Pliploop/SLAP commit b49290186ee354d34798f9947110a375f9e3f5a7.
Paper: https://arxiv.org/abs/2506.17815.

Typical usage:
    model = hydra.utils.instantiate(cfg.model)
    trainer.fit(model, datamodule=datamodule)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Literal, cast

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from lightning.pytorch import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from synth_setter.models.components.slap import BYOLLoss, SiameseArm
from synth_setter.models.components.slap_ema import MovingAverageWeightUpdate

OptimizerFactory = Callable[..., torch.optim.Optimizer]
SchedulerFactory = Callable[..., torch.optim.lr_scheduler.LRScheduler]
type BatchTensor = Float[Tensor, "batch ..."]
type ModelBatch = Mapping[str, BatchTensor | None]
type ScalarTensor = Float[Tensor, ""]


@jaxtyped(typechecker=beartype)
def _paired_inputs(
    batch: ModelBatch,
    audio_input_key: Literal["audio", "mel"],
) -> tuple[BatchTensor, BatchTensor]:
    """Return non-null audio-modality and parameter tensors from a batch.

    :param batch: Collated model batch carrying paired modalities.
    :param audio_input_key: Batch key supplying the audio arm input.
    :returns: Audio-modality and parameter tensors in matching row order.
    :raises ValueError: If either required modality is absent.
    """
    audio_input = batch.get(audio_input_key)
    params = batch.get("params")
    if audio_input is None or params is None:
        raise ValueError(f"SLAP batches require non-null {audio_input_key} and params")
    return audio_input, params


class SLAPModule(LightningModule):
    """Train paired audio and parameter encoders using the reference SLAP objective."""

    _ema_optimizer_steps: Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        audio_encoder: SiameseArm,
        text_encoder: SiameseArm,
        loss_fn: BYOLLoss,
        optimizer: OptimizerFactory,
        *,
        audio_input_key: Literal["audio", "mel"] = "audio",
        scheduler: SchedulerFactory | None = None,
        ma_callback: MovingAverageWeightUpdate | None = None,
        compile: bool | str = False,
    ) -> None:
        """Build online and moving-average modality arms.

        :param audio_encoder: Online Siamese arm consuming audio or mel batches.
        :param text_encoder: Online Siamese arm consuming the paired second modality.
        :param loss_fn: Reference BYOL loss over online predictions and target projections.
        :param optimizer: Partially configured optimizer factory.
        :param audio_input_key: Batch key supplying the audio arm input.
        :param scheduler: Optional partially configured scheduler factory.
        :param ma_callback: Target-weight update policy.
        :param compile: Whether and how to compile both online and target arms.
        """
        super().__init__()
        self.register_buffer("_ema_optimizer_steps", torch.zeros((), dtype=torch.long))
        self.save_hyperparameters(
            ignore=[
                "audio_encoder",
                "text_encoder",
                "loss_fn",
                "optimizer",
                "scheduler",
                "ma_callback",
            ],
            logger=False,
        )
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_input_key: Literal["audio", "mel"] = audio_input_key
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.ma_callback = ma_callback or MovingAverageWeightUpdate()

        self.audio_ema = deepcopy(audio_encoder)
        self.audio_ema.transform = nn.Identity()
        self.text_ema = deepcopy(text_encoder)
        self.text_ema.transform = nn.Identity()
        self._freeze_targets()

    @jaxtyped(typechecker=beartype)
    def _freeze_targets(self) -> None:
        """Keep target arms outside gradient and train-mode state changes."""
        self.audio_ema.requires_grad_(False)
        self.audio_ema.eval()
        self.text_ema.requires_grad_(False)
        self.text_ema.eval()

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> SLAPModule:
        """Set online mode while keeping moving-average targets in evaluation mode.

        :param mode: Whether online modules enter training mode.
        :returns: This module after applying the requested mode.
        """
        super().train(mode)
        self.audio_ema.eval()
        self.text_ema.eval()
        return self

    @jaxtyped(typechecker=beartype)
    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Remove target-predictor tensors that target arms do not own.

        :param checkpoint: Lightning checkpoint whose state is migrated in place.
        """
        state_dict = cast(dict[str, Tensor], checkpoint["state_dict"])
        predictor_prefixes = (
            "audio_ema.transform.",
            "audio_ema._orig_mod.transform.",
            "text_ema.transform.",
            "text_ema._orig_mod.transform.",
        )
        for name in tuple(state_dict):
            if name.startswith(predictor_prefixes):
                state_dict.pop(name)
        if "_ema_optimizer_steps" not in state_dict:
            global_step = checkpoint.get("global_step", 0)
            completed_steps = global_step if isinstance(global_step, int) else 0
            state_dict["_ema_optimizer_steps"] = torch.tensor(completed_steps, dtype=torch.long)

    @jaxtyped(typechecker=beartype)
    def _losses(self, batch: ModelBatch) -> dict[str, ScalarTensor]:
        """Compute online predictions against moving-average projections.

        :param batch: Collated model batch carrying paired modalities.
        :returns: Scalar reference loss terms.
        :raises ValueError: If either arm omits its required projector or predictor.
        """
        audio, params = _paired_inputs(batch, self.audio_input_key)
        _, _, audio_prediction = self.audio_encoder(audio)
        _, _, text_prediction = self.text_encoder(params)
        with torch.no_grad():
            _, audio_projection_ema, _ = self.audio_ema(audio)
            _, text_projection_ema, _ = self.text_ema(params)

        values = (audio_prediction, text_prediction, audio_projection_ema, text_projection_ema)
        if any(value is None for value in values):
            raise ValueError("SLAP arms require projectors and prediction transforms")
        qa, qt, za_ema, zt_ema = cast(
            tuple[BatchTensor, BatchTensor, BatchTensor, BatchTensor], values
        )
        return self.loss_fn(qa, qt, za_ema, zt_ema)

    @jaxtyped(typechecker=beartype)
    def training_step(self, batch: ModelBatch, batch_idx: int) -> ScalarTensor:
        """Return and log the reference SLAP objective for one paired batch.

        :param batch: Collated paired-modality training batch.
        :param batch_idx: Unused zero-based batch position.
        :returns: Scalar objective optimized by Lightning.
        """
        del batch_idx
        losses = self._losses(batch)
        self.log_dict({f"loss/train/{name}": value for name, value in losses.items()})
        return losses["total_loss"]

    @jaxtyped(typechecker=beartype)
    def validation_step(self, batch: ModelBatch, batch_idx: int) -> None:
        """Log validation losses for checkpoint selection.

        :param batch: Collated paired-modality validation batch.
        :param batch_idx: Unused zero-based batch position.
        """
        del batch_idx
        losses = self._losses(batch)
        self.log_dict(
            {f"loss/val/{name}": value for name, value in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    @jaxtyped(typechecker=beartype)
    def test_step(self, batch: ModelBatch, batch_idx: int) -> None:
        """Log checkpoint-reloaded test losses.

        :param batch: Collated paired-modality test batch.
        :param batch_idx: Unused zero-based batch position.
        """
        del batch_idx
        losses = self._losses(batch)
        self.log_dict(
            {f"loss/test/{name}": value for name, value in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    @jaxtyped(typechecker=beartype)
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], object] | None = None,
    ) -> None:
        """Run EMA only after Lightning performs an effective optimizer step.

        :param epoch: Zero-based training epoch.
        :param batch_idx: Zero-based batch position within the epoch.
        :param optimizer: Lightning-managed optimizer.
        :param optimizer_closure: Closure that evaluates the accumulated objective.
        """
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        active_scaler = scaler if isinstance(scaler, GradScaler) else None
        scale_before = active_scaler.get_scale() if active_scaler is not None else None
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        optimizer_step_skipped = (
            active_scaler is not None
            and scale_before is not None
            and active_scaler.get_scale() < scale_before
        )
        if optimizer_step_skipped:
            return
        completed_step = int(self._ema_optimizer_steps.item()) + 1
        self._ema_optimizer_steps.fill_(completed_step)
        self.ma_callback.on_optimizer_step(self.trainer, self, completed_step)

    @jaxtyped(typechecker=beartype)
    def setup(self, stage: str) -> None:
        """Compile online and target arms for fitting when requested.

        :param stage: Lightning lifecycle stage.
        """
        compile_mode = self.hparams["compile"]
        if not compile_mode or stage != "fit":
            return
        mode = compile_mode if isinstance(compile_mode, str) else "default"
        self.audio_encoder = torch.compile(self.audio_encoder, mode=mode)
        self.text_encoder = torch.compile(self.text_encoder, mode=mode)
        self.audio_ema = torch.compile(self.audio_ema, mode=mode)
        self.text_ema = torch.compile(self.text_ema, mode=mode)
        self._freeze_targets()

    @jaxtyped(typechecker=beartype)
    def configure_optimizers(self) -> dict[str, object]:
        """Construct the configured optimizer and optional scheduler.

        :returns: Lightning optimizer configuration with an optional epoch scheduler.
        """
        parameters = [
            *self.audio_encoder.parameters(),
            *self.text_encoder.parameters(),
            *self.loss_fn.parameters(),
        ]
        optimizer = self.optimizer_factory(params=parameters)
        if self.scheduler_factory is None:
            return {"optimizer": optimizer}
        scheduler = self.scheduler_factory(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
