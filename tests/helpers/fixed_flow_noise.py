"""Deterministic Lightning callback for real single-example training tests."""

import torch
from lightning import Callback, LightningModule, Trainer


class FixedFlowNoise(Callback):
    """Supply one fixed real-noise row to single-example train and test loops."""

    def __init__(self, seed: int) -> None:
        """Store the deterministic noise seed.

        :param seed: Torch RNG seed used to construct the fixed noise row.
        """
        super().__init__()
        self.seed = seed

    def _set_batch_noise(self, batch: dict[str, torch.Tensor]) -> None:
        """Replace collated noise with a deterministic normal draw.

        :param batch: Mutable production flow batch.
        """
        generator = torch.Generator(device=batch["params"].device).manual_seed(self.seed)
        batch["noise"] = torch.randn(
            batch["params"].shape,
            dtype=batch["params"].dtype,
            device=batch["params"].device,
            generator=generator,
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Supply the fixed flow-noise row before training.

        :param trainer: Unused Lightning trainer hook argument.
        :param pl_module: Unused model hook argument.
        :param batch: Mutable production flow batch.
        :param batch_idx: Unused batch index.
        """
        del trainer, pl_module, batch_idx
        self._set_batch_noise(batch)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Supply the same flow-noise row before checkpoint sampling.

        :param trainer: Unused Lightning trainer hook argument.
        :param pl_module: Unused model hook argument.
        :param batch: Mutable production flow batch.
        :param batch_idx: Unused batch index.
        :param dataloader_idx: Unused dataloader index.
        """
        del trainer, pl_module, batch_idx, dataloader_idx
        self._set_batch_noise(batch)
        torch.manual_seed(self.seed)
