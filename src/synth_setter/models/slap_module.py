"""Lightning training module for Siamese Language-Audio Pretraining.

Source: Pliploop/SLAP commit b49290186ee354d34798f9947110a375f9e3f5a7.
Paper: https://arxiv.org/abs/2506.17815.

Typical usage:
    model = hydra.utils.instantiate(cfg.model)
    trainer.fit(model, datamodule=datamodule)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Literal, cast

import torch
from beartype import beartype
from jaxtyping import Float, Int64, jaxtyped
from lightning.pytorch import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from synth_setter.evaluation.paired_retrieval import RetrievalBatches, gathered_retrieval_metrics
from synth_setter.models.components.slap import BYOLLoss, SiameseArm
from synth_setter.models.components.slap_ema import MovingAverageWeightUpdate

OptimizerFactory = Callable[..., torch.optim.Optimizer]
SchedulerFactory = Callable[..., torch.optim.lr_scheduler.LRScheduler]
_BATCH_ROWS = "batch"
type BatchTensor = Float[Tensor, "batch ..."]
type ModelBatch = Mapping[str, BatchTensor | Int64[Tensor, _BATCH_ROWS] | None]
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
        param_encoder: SiameseArm | None = None,
        loss_fn: BYOLLoss | None = None,
        optimizer: OptimizerFactory | None = None,
        *,
        text_encoder: SiameseArm | None = None,
        audio_input_key: Literal["audio", "mel"] = "audio",
        scheduler: SchedulerFactory | None = None,
        ma_callback: MovingAverageWeightUpdate | None = None,
        compile: bool | str = False,
        retrieval_eval: bool = False,
    ) -> None:
        """Build online and moving-average modality arms.

        :param audio_encoder: Online Siamese arm consuming audio or mel batches.
        :param param_encoder: Online Siamese arm consuming parameter batches.
        :param loss_fn: Reference BYOL loss over online predictions and target projections.
        :param optimizer: Partially configured optimizer factory.
        :param text_encoder: Deprecated alias preserving the caller's encoder architecture.
        :param audio_input_key: Batch key supplying the audio arm input.
        :param scheduler: Optional partially configured scheduler factory.
        :param ma_callback: Target-weight update policy.
        :param compile: Whether and how to compile both online and target arms.
        :param retrieval_eval: Score full held-out galleries; requires int64 ``sample_id`` batches.
        :raises ValueError: If arm names are ambiguous or required dependencies are absent.
        """
        super().__init__()
        if (param_encoder is None) == (text_encoder is None):
            raise ValueError("Provide exactly one of param_encoder and text_encoder")
        if loss_fn is None or optimizer is None:
            raise ValueError("loss_fn and optimizer are required")
        if text_encoder is not None:
            warnings.warn(
                "text_encoder is deprecated; use param_encoder", DeprecationWarning, stacklevel=2
            )
            param_encoder = text_encoder
        assert param_encoder is not None
        self.register_buffer("_ema_optimizer_steps", torch.zeros((), dtype=torch.long))
        self.save_hyperparameters(
            ignore=[
                "audio_encoder",
                "param_encoder",
                "text_encoder",
                "loss_fn",
                "optimizer",
                "scheduler",
                "ma_callback",
            ],
            logger=False,
        )
        self.audio_encoder = audio_encoder
        self.param_encoder = param_encoder
        self.audio_input_key: Literal["audio", "mel"] = audio_input_key
        self.loss_fn = loss_fn
        self.retrieval_eval = retrieval_eval
        self._retrieval_batches: RetrievalBatches = {}
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.ma_callback = ma_callback or MovingAverageWeightUpdate()

        self.audio_ema = deepcopy(audio_encoder)
        self.audio_ema.transform = nn.Identity()
        self.param_ema = deepcopy(param_encoder)
        self.param_ema.transform = nn.Identity()
        self._freeze_targets()

    @property
    @jaxtyped(typechecker=beartype)
    def text_encoder(self) -> nn.Module:
        """Expose the parameter arm to legacy export consumers without duplicate state."""
        return cast(nn.Module, self.param_encoder)

    @property
    @jaxtyped(typechecker=beartype)
    def text_ema(self) -> nn.Module:
        """Expose the parameter EMA projection arm to legacy export consumers."""
        return cast(nn.Module, self.param_ema)

    @jaxtyped(typechecker=beartype)
    def _freeze_targets(self) -> None:
        """Keep target arms outside gradient and train-mode state changes."""
        self.audio_ema.requires_grad_(False)
        self.audio_ema.eval()
        self.param_ema.requires_grad_(False)
        self.param_ema.eval()

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> SLAPModule:
        """Set online mode while keeping moving-average targets in evaluation mode.

        :param mode: Whether online modules enter training mode.
        :returns: This module after applying the requested mode.
        """
        super().train(mode)
        self.audio_ema.eval()
        self.param_ema.eval()
        return self

    @jaxtyped(typechecker=beartype)
    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Migrate parameter-arm names without replacing the checkpoint architecture.

        :param checkpoint: Lightning checkpoint whose state is migrated in place.
        :raises ValueError: If canonical and legacy names coexist.
        """
        hparams = cast(dict[str, object], checkpoint.get("hyper_parameters", {}))
        if "text_encoder" in hparams:
            if "param_encoder" in hparams:
                raise ValueError("ambiguous parameter encoder hyperparameters")
            hparams["param_encoder"] = hparams.pop("text_encoder")
        state_dict = cast(dict[str, Tensor], checkpoint["state_dict"])
        for legacy, canonical in (
            ("text_encoder.", "param_encoder."),
            ("text_ema.", "param_ema."),
        ):
            legacy_keys = [name for name in state_dict if name.startswith(legacy)]
            if legacy_keys and any(name.startswith(canonical) for name in state_dict):
                raise ValueError(f"ambiguous checkpoint state: {legacy} and {canonical}")
            for name in legacy_keys:
                state_dict[canonical + name.removeprefix(legacy)] = state_dict.pop(name)
        predictor_prefixes = (
            "audio_ema.transform.",
            "audio_ema._orig_mod.transform.",
            "param_ema.transform.",
            "param_ema._orig_mod.transform.",
        )
        for name in tuple(state_dict):
            if name.startswith(predictor_prefixes):
                state_dict.pop(name)
        for arm_name in ("audio_encoder", "param_encoder", "audio_ema", "param_ema"):
            arm = getattr(self, arm_name)
            if hasattr(arm, "_orig_mod"):
                continue
            compiled_prefix = f"{arm_name}._orig_mod."
            for name in tuple(state_dict):
                if not name.startswith(compiled_prefix):
                    continue
                eager_name = f"{arm_name}." + name.removeprefix(compiled_prefix)
                if eager_name in state_dict:
                    raise ValueError(f"ambiguous compiled checkpoint state: {eager_name}")
                state_dict[eager_name] = state_dict.pop(name)
        if "_ema_optimizer_steps" not in state_dict:
            global_step = checkpoint.get("global_step", 0)
            completed_steps = global_step if isinstance(global_step, int) else 0
            state_dict["_ema_optimizer_steps"] = torch.tensor(completed_steps, dtype=torch.long)

    @jaxtyped(typechecker=beartype)
    def _losses(
        self, batch: ModelBatch, dataloader_idx: int | None = None
    ) -> dict[str, ScalarTensor]:
        """Compute online predictions against moving-average projections.

        :param batch: Collated model batch carrying paired modalities.
        :param dataloader_idx: Evaluation loader receiving predictions, or ``None`` for training.
        :returns: Scalar reference loss terms.
        :raises ValueError: If arms omit required outputs or evaluation IDs are missing/invalid.
        """
        audio, params = _paired_inputs(batch, self.audio_input_key)
        _, _, audio_prediction = self.audio_encoder(audio)
        _, _, param_prediction = self.param_encoder(params)
        with torch.no_grad():
            _, audio_projection_ema, _ = self.audio_ema(audio)
            _, param_projection_ema, _ = self.param_ema(params)

        values = (audio_prediction, param_prediction, audio_projection_ema, param_projection_ema)
        if any(value is None for value in values):
            raise ValueError("SLAP arms require projectors and prediction transforms")
        qa, qt, za_ema, zt_ema = cast(
            tuple[BatchTensor, BatchTensor, BatchTensor, BatchTensor], values
        )
        if self.retrieval_eval and dataloader_idx is not None:
            ids = batch.get("sample_id")
            if ids is None or ids.dtype != torch.int64 or ids.shape != (len(qa),):
                raise ValueError("retrieval evaluation requires one int64 sample_id per row")
            self._retrieval_batches.setdefault(dataloader_idx, []).append(
                (qa.detach().cpu(), qt.detach().cpu(), ids.detach().cpu())
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
    def validation_step(self, batch: ModelBatch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Log validation losses for checkpoint selection.

        :param batch: Collated paired-modality validation batch.
        :param batch_idx: Unused zero-based batch position.
        :param dataloader_idx: Loader namespace for split-local source identities.
        """
        del batch_idx
        losses = self._losses(batch, dataloader_idx)
        self.log_dict(
            {f"loss/val/{name}": value for name, value in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    @jaxtyped(typechecker=beartype)
    def test_step(self, batch: ModelBatch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Log checkpoint-reloaded test losses.

        :param batch: Collated paired-modality test batch.
        :param batch_idx: Unused zero-based batch position.
        :param dataloader_idx: Loader namespace for split-local source identities.
        """
        del batch_idx
        losses = self._losses(batch, dataloader_idx)
        self.log_dict(
            {f"loss/test/{name}": value for name, value in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    @jaxtyped(typechecker=beartype)
    def on_validation_epoch_start(self) -> None:
        """Discard prior validation and sanity-check observations."""
        self._retrieval_batches.clear()

    @jaxtyped(typechecker=beartype)
    def on_test_epoch_start(self) -> None:
        """Discard observations from any previous evaluation run."""
        self._retrieval_batches.clear()

    @jaxtyped(typechecker=beartype)
    def _log_retrieval(self, stage: Literal["val", "test"]) -> None:
        """Log rank-global metrics without averaging rank-local retrieval scores.

        :param stage: Logging namespace for the completed evaluation loop.
        """
        if not self.retrieval_eval:
            return
        try:
            if stage == "val" and self.trainer.sanity_checking:
                return
            for index, metrics in gathered_retrieval_metrics(self._retrieval_batches).items():
                loaders = (
                    self.trainer.val_dataloaders
                    if stage == "val"
                    else self.trainer.test_dataloaders
                )
                suffix = (
                    f"/dataloader_idx_{index}"
                    if isinstance(loaders, (list, tuple)) and len(loaders) > 1
                    else ""
                )
                self.log_dict(
                    {
                        f"retrieval/{stage}/{name}{suffix}": value
                        for name, value in metrics.items()
                    },
                    sync_dist=False,
                    add_dataloader_idx=False,
                )
        finally:
            self._retrieval_batches.clear()

    @jaxtyped(typechecker=beartype)
    def on_validation_epoch_end(self) -> None:
        """Score the full validation gallery, excluding partial sanity checks."""
        self._log_retrieval("val")

    @jaxtyped(typechecker=beartype)
    def on_test_epoch_end(self) -> None:
        """Score the full checkpoint-reloaded test gallery."""
        self._log_retrieval("test")

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
        self.param_encoder = torch.compile(self.param_encoder, mode=mode)
        self.audio_ema = torch.compile(self.audio_ema, mode=mode)
        self.param_ema = torch.compile(self.param_ema, mode=mode)
        self._freeze_targets()

    @jaxtyped(typechecker=beartype)
    def configure_optimizers(self) -> dict[str, object]:
        """Construct the configured optimizer and optional scheduler.

        :returns: Lightning optimizer configuration with an optional epoch scheduler.
        """
        parameters = [
            *self.audio_encoder.parameters(),
            *self.param_encoder.parameters(),
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
