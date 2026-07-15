"""Callback test double that simulates a mid-fit training crash."""

from lightning.pytorch import Callback, LightningModule, Trainer


class RaiseOnTrainBatchEnd(Callback):
    """Raise after the first training batch to exercise Lightning's crash hooks."""

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Simulate a failure after the first training batch.

        :param trainer: Active trainer handling the simulated failure.
        :param pl_module: Model that completed the batch.
        :param outputs: Training-step output for the batch.
        :param batch: Training batch that completed.
        :param batch_idx: Zero-based index of the completed batch.
        :raises RuntimeError: Always, to trigger checkpoint-on-exception handling.
        """
        raise RuntimeError("simulated mid-fit crash")
