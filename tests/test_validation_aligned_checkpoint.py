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
    """Expose a visible manual-optimizer update for recovery tests."""

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

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Apply one visible manual-optimizer update.

        :param batch: Unused synthetic training batch.
        :param batch_idx: Unused batch index.
        :returns: Loss used for the manual update.
        """
        del batch, batch_idx
        optimizer = cast(LightningOptimizer, self.optimizers())
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
        """Use visible model and momentum updates at each boundary.

        :returns: Momentum SGD optimizer for the scalar test layer.
        """
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

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
def test_checkpoint_manual_optimization_recovery_uses_current_state(
    tmp_path: Path, save_last: bool | Literal["link"]
) -> None:
    """The recovery file keeps model and optimizer at the completed update boundary.

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

    module = _ManualOptimizationModule()
    trainer.fit(module)

    last = torch.load(checkpoint.last_model_path, map_location="cpu", weights_only=False)
    assert last["global_step"] == 2
    assert last["state_dict"]["layer.weight"].item() == pytest.approx(module.layer.weight.item())
    checkpoint_momentum = last["optimizer_states"][0]["state"][0]["momentum_buffer"]
    live_momentum = trainer.optimizers[0].state[module.layer.weight]["momentum_buffer"]
    assert torch.equal(checkpoint_momentum, live_momentum)


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

    assert module.recovery_weight_at_epoch_end == pytest.approx(module.layer.weight.item())
