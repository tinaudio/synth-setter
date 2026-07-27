"""Callback test double that simulates a mid-fit training crash."""

from lightning.pytorch import Callback, LightningModule, Trainer


class _RaiseOnTrainBatchEnd(Callback):
    """Raise after the first training batch to exercise Lightning's crash hooks."""

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        raise RuntimeError("simulated mid-fit crash")
