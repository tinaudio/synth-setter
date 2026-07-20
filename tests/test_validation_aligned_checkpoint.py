"""Compatibility tests for validation-aligned checkpoint selection."""

from pathlib import Path
from typing import cast

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.utils.data import DataLoader, TensorDataset

from synth_setter.utils.callbacks import ValidationAlignedModelCheckpoint


class _ManualOptimizationModule(pl.LightningModule):
    """Expose Lightning's pre-optimizer checkpoint contract."""

    def __init__(self) -> None:
        super().__init__()
        setattr(self, "automatic_optimization", False)
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use a visible update so pre- and post-optimizer states differ.

        :returns: SGD optimizer for the scalar test layer.
        """
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Provide two training steps.

        :returns: Two-batch synthetic loader.
        """
        return DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Validate after each training step.

        :returns: One-batch synthetic loader.
        """
        return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)


def test_checkpoint_state_key_remains_resume_compatible() -> None:
    """Existing ModelCheckpoint state restores into the aligned callback."""
    kwargs = {"monitor": "val/score", "mode": "min", "every_n_train_steps": 10}
    existing = ModelCheckpoint(**kwargs)
    aligned = ValidationAlignedModelCheckpoint(**kwargs)

    assert aligned.state_key == existing.state_key


def test_checkpoint_manual_optimization_recovery_uses_pre_update_weights(tmp_path: Path) -> None:
    """The recovery file preserves Lightning's manual-optimization step semantics.

    :param tmp_path: Temporary checkpoint directory.
    """
    checkpoint = ValidationAlignedModelCheckpoint(
        dirpath=tmp_path,
        monitor="val/score",
        mode="min",
        save_last=True,
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
