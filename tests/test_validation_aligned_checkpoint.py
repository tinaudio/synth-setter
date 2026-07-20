"""Compatibility tests for validation-aligned checkpoint selection."""

from pathlib import Path
from typing import Literal, cast

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.utils.data import DataLoader, TensorDataset

from synth_setter.utils.callbacks import ValidationAlignedModelCheckpoint


class _ManualOptimizationModule(pl.LightningModule):
    """Expose Lightning's pre-optimizer checkpoint contract."""

    def __init__(self, checkpoint: ValidationAlignedModelCheckpoint | None = None) -> None:
        """Initialize the manual optimizer fixture.

        :param checkpoint: Optional callback inspected before ``on_train_end``.
        """
        super().__init__()
        setattr(self, "automatic_optimization", False)
        self.checkpoint = checkpoint
        self.recovery_weight_at_epoch_end: float | None = None
        self.layer = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.layer.weight)
        self.saved_models: dict[int, dict[str, torch.Tensor]] = {}

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Save the current-step state before updating.

        :param batch: Unused synthetic training batch.
        :param batch_idx: Unused batch index.
        :returns: Loss used for the manual update.
        """
        del batch, batch_idx
        optimizer = cast(LightningOptimizer, self.optimizers())
        self.saved_models[self.global_step] = {
            key: value.detach().clone() for key, value in self.layer.state_dict().items()
        }
        loss = (self.layer.weight - 1).square().mean()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> None:
        """Publish a stable monitored metric.

        :param batch: Unused synthetic validation batch.
        :param batch_idx: Unused batch index.
        """
        del batch, batch_idx
        self.log("val/score", 1.0, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        """Capture the step recovery file before ``on_train_end`` can overwrite it."""
        if self.checkpoint is None or not self.checkpoint.last_model_path:
            return
        saved = torch.load(self.checkpoint.last_model_path, map_location="cpu", weights_only=False)
        self.recovery_weight_at_epoch_end = saved["state_dict"]["layer.weight"].item()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use a visible update so pre- and post-optimizer states differ.

        :returns: SGD optimizer for the scalar test layer.
        """
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Provide deterministic training data.

        :returns: Synthetic training loader.
        """
        return DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Validate after each training step.

        :returns: Synthetic validation loader.
        """
        return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)


def test_checkpoint_state_key_remains_resume_compatible() -> None:
    """Existing ModelCheckpoint state restores into the aligned callback."""
    kwargs = {"monitor": "val/score", "mode": "min", "every_n_train_steps": 10}
    existing = ModelCheckpoint(**kwargs)
    aligned = ValidationAlignedModelCheckpoint(**kwargs)

    assert aligned.state_key == existing.state_key


@pytest.mark.parametrize("save_last", [True, "link"])
def test_checkpoint_manual_optimization_recovery_uses_pre_update_weights(
    tmp_path: Path, save_last: bool | Literal["link"]
) -> None:
    """The recovery file preserves Lightning's manual-optimization step semantics.

    :param tmp_path: Temporary checkpoint directory.
    :param save_last: Recovery checkpoint mode under test.
    """
    checkpoint = ValidationAlignedModelCheckpoint(
        dirpath=tmp_path,
        monitor="val/score",
        mode="min",
        save_last=save_last,
        save_top_k=1,
        every_n_train_steps=1,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        callbacks=[checkpoint],
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        limit_train_batches=2,
        limit_val_batches=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        val_check_interval=1,
    )

    trainer.fit(_ManualOptimizationModule())

    last = torch.load(checkpoint.last_model_path, map_location="cpu", weights_only=False)
    assert last["global_step"] == 2
    assert last["state_dict"]["layer.weight"].item() == pytest.approx(0.2)


@pytest.mark.parametrize("save_last", [True, "link"])
def test_checkpoint_manual_optimization_saves_recovery_before_validation(
    tmp_path: Path, save_last: bool | Literal["link"]
) -> None:
    """Manual optimization writes recovery state before a monitored metric exists.

    :param tmp_path: Temporary checkpoint directory.
    :param save_last: Recovery checkpoint mode under test.
    """
    checkpoint = ValidationAlignedModelCheckpoint(
        dirpath=tmp_path,
        monitor="val/score",
        mode="min",
        save_last=save_last,
        save_on_train_epoch_end=False,
        save_top_k=1,
        every_n_train_steps=1,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        callbacks=[checkpoint],
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=1,
    )

    module = _ManualOptimizationModule(checkpoint)
    trainer.fit(module)

    expected_pre_update = module.saved_models[0]["weight"].item()
    assert module.recovery_weight_at_epoch_end == expected_pre_update == 0.0
