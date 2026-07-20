"""Tiny Lightning fixtures for checkpoint-selection entrypoint tests."""

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class ValidationTrajectoryDataModule(pl.LightningDataModule):
    """Provide deterministic training and validation data."""

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Provide deterministic training data for validation-cadence tests.

        :returns: Synthetic training dataloader.
        """
        return DataLoader(TensorDataset(torch.zeros(6, 1)), batch_size=1)

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Provide deterministic validation data.

        :returns: Synthetic validation dataloader.
        """
        return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)


class ValidationTrajectoryModule(pl.LightningModule):
    """Expose a deterministic validation metric trajectory and weight marker."""

    def __init__(self) -> None:
        """Initialize the trainable scalar and checkpoint-visible batch counter."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("trained_batches", torch.tensor(0))

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Advance the checkpoint-visible weight marker.

        :param batch: Unused synthetic training batch.
        :param batch_idx: Unused batch index.
        :returns: A differentiable zero loss.
        """
        del batch, batch_idx
        self.trained_batches += 1
        return self.weight.square()

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> None:
        """Log a deterministic improving-then-worsening metric trajectory.

        :param batch: Unused synthetic validation batch.
        :param batch_idx: Unused batch index.
        """
        del batch, batch_idx
        scores = {2: 3.0, 4: 1.0, 6: 2.0}
        self.log("val/score", scores[self.global_step], on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Keep weights fixed apart from the explicit batch marker.

        :returns: Zero-learning-rate SGD optimizer.
        """
        return torch.optim.SGD(self.parameters(), lr=0.0)
