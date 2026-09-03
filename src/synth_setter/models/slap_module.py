"""Lightning training module for Siamese Language-Audio Pretraining.

Source: Pliploop/SLAP commit b49290186ee354d34798f9947110a375f9e3f5a7.
Paper: https://arxiv.org/abs/2506.17815.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import cast

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from lightning.pytorch import LightningModule
from torch import Tensor

from synth_setter.models.components.slap import BYOLLoss, SiameseArm
from synth_setter.models.components.slap_ema import MovingAverageWeightUpdate

OptimizerFactory = Callable[..., torch.optim.Optimizer]
SchedulerFactory = Callable[..., torch.optim.lr_scheduler.LRScheduler]
type BatchTensor = Float[Tensor, "batch ..."]
type ModelBatch = Mapping[str, BatchTensor | None]
type ScalarTensor = Float[Tensor, ""]


class SLAPModule(LightningModule):
    """Train paired audio and parameter encoders using the reference SLAP objective."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        audio_encoder: SiameseArm,
        text_encoder: SiameseArm,
        loss_fn: BYOLLoss,
        optimizer: OptimizerFactory,
        scheduler: SchedulerFactory | None = None,
        ma_callback: MovingAverageWeightUpdate | None = None,
        *,
        compile: bool | str = False,
    ) -> None:
        """Build online and moving-average modality arms.

        :param audio_encoder: Online Siamese arm consuming waveform batches.
        :param text_encoder: Online Siamese arm consuming the paired second modality.
        :param loss_fn: Reference BYOL loss over online predictions and target projections.
        :param optimizer: Partially configured optimizer factory.
        :param scheduler: Optional partially configured scheduler factory.
        :param ma_callback: Target-weight update policy.
        :param compile: Whether and how to compile both online and target arms.
        """
        super().__init__()
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
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.ma_callback = ma_callback or MovingAverageWeightUpdate()

        self.audio_ema = deepcopy(audio_encoder)
        self.audio_ema.requires_grad_(False)
        self.text_ema = deepcopy(text_encoder)
        self.text_ema.requires_grad_(False)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _paired_inputs(batch: ModelBatch) -> tuple[BatchTensor, BatchTensor]:
        """Return non-null audio and parameter tensors from a synth-setter batch.

        :param batch: Collated model batch carrying paired modalities.
        :returns: Audio and parameter tensors in matching row order.
        :raises ValueError: If either required modality is absent.
        """
        audio = batch.get("audio")
        params = batch.get("params")
        if audio is None or params is None:
            raise ValueError("SLAP batches require non-null audio and params")
        return audio, params

    @jaxtyped(typechecker=beartype)
    def _losses(self, batch: ModelBatch) -> dict[str, ScalarTensor]:
        """Compute online predictions against moving-average projections.

        :param batch: Collated model batch carrying paired modalities.
        :returns: Scalar reference loss terms.
        :raises ValueError: If either arm omits its required projector or predictor.
        """
        audio, params = self._paired_inputs(batch)
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
        return self.loss_fn(qt, qa, za_ema, zt_ema)

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
        self.log_dict({f"loss/val/{name}": value for name, value in losses.items()})

    @jaxtyped(typechecker=beartype)
    def test_step(self, batch: ModelBatch, batch_idx: int) -> None:
        """Log checkpoint-reloaded test losses.

        :param batch: Collated paired-modality test batch.
        :param batch_idx: Unused zero-based batch position.
        """
        del batch_idx
        losses = self._losses(batch)
        self.log_dict({f"loss/test/{name}": value for name, value in losses.items()})

    @jaxtyped(typechecker=beartype)
    def on_train_batch_end(self, outputs: object, batch: ModelBatch, batch_idx: int) -> None:
        """Delegate target updates to the reference moving-average policy.

        :param outputs: Lightning-wrapped training-step output.
        :param batch: Collated batch whose optimization just completed.
        :param batch_idx: Zero-based batch position used by the update cadence.
        """
        del outputs, batch
        self.ma_callback.on_train_batch_end(self.trainer, self, batch_idx)

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
