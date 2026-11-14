"""Teardown-guard tests for the completed-fit wrapper in ``synth_setter.cli.train``."""

import logging

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from synth_setter.cli.train import _completed_fit_survives_teardown


class _OneLinearStep(LightningModule):
    """Smallest module that can take real optimizer steps over a tensor dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Return a scalar loss over the batch.

        :param batch: Single-tensor batch from a ``TensorDataset``.
        :param batch_idx: Unused; present for the Lightning hook signature.
        :returns: Mean squared activation, which is differentiable.
        """
        del batch_idx
        return self.layer(batch[0]).square().mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return a plain SGD optimizer.

        :returns: Optimizer over this module's parameters.
        """
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _fit_tiny_model_failing_teardown(
    monkeypatch: pytest.MonkeyPatch, teardown_error: BaseException | None, max_steps: int
) -> Trainer:
    """Run a real CPU fit whose first dataloader teardown raises ``teardown_error``.

    Faults are injected at ``CombinedLoader.reset`` because that is where Lightning
    shuts persistent workers down (#2981); a worker aborting there is what the
    reported run hit. The resulting masking error comes from real Lightning code.

    :param monkeypatch: Fixture used to install the failing teardown.
    :param teardown_error: Error the first teardown raises; ``None`` tears down cleanly.
    :param max_steps: Optimizer steps the fit runs before stopping.
    :returns: The trainer, so callers can assert on the fit's recorded progress.
    """
    resets = {"count": 0}
    real_reset = CombinedLoader.reset

    def failing_reset(self: CombinedLoader) -> None:
        real_reset(self)
        resets["count"] += 1
        if teardown_error is not None and resets["count"] == 1:
            raise teardown_error

    monkeypatch.setattr(CombinedLoader, "reset", failing_reset)
    trainer = Trainer(
        max_steps=max_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.arange(64, dtype=torch.float32)[:, None]),
        batch_size=4,
    )
    with _completed_fit_survives_teardown(trainer):
        trainer.fit(_OneLinearStep(), train_dataloaders=loader)
    return trainer


def test_completed_fit_survives_a_dataloader_teardown_failure() -> None:
    """A fit that reached ``max_steps`` must not be failed by its own teardown."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        trainer = _fit_tiny_model_failing_teardown(
            monkeypatch,
            RuntimeError("DataLoader worker (pid 4242) is killed by signal: Aborted."),
            max_steps=5,
        )

    assert trainer.global_step == 5


def test_completed_fit_teardown_failure_logs_the_worker_error_not_the_masking_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The surviving log names the worker abort, not ``CombinedLoader.__len__``.

    Lightning's ``sized_len`` catches only ``TypeError``/``NotImplementedError``, so
    ``CombinedLoader.__len__`` raising after a reset replaces the real failure (#2981).

    :param caplog: Captures the surviving error record.
    """
    with caplog.at_level(logging.ERROR), pytest.MonkeyPatch.context() as monkeypatch:
        _fit_tiny_model_failing_teardown(
            monkeypatch,
            RuntimeError("DataLoader worker (pid 4242) is killed by signal: Aborted."),
            max_steps=5,
        )

    assert "killed by signal: Aborted" in caplog.text
    assert "iter(combined_loader)" not in caplog.text


class _FailingStep(_OneLinearStep):
    """Module whose third training step raises, leaving the fit loop unfinished."""

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Raise partway through training instead of returning a loss.

        :param batch: Single-tensor batch from a ``TensorDataset``.
        :param batch_idx: Index used to fail only once training is under way.
        :returns: The parent's loss for the steps before the failure.
        :raises RuntimeError: Once ``batch_idx`` reaches 2.
        """
        if batch_idx == 2:
            raise RuntimeError("loss blew up")
        return super().training_step(batch, batch_idx)


def test_failure_before_the_fit_completes_still_raises() -> None:
    """An unfinished fit must keep failing — the guard is not a blanket suppressor."""
    trainer = Trainer(
        max_steps=5,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.arange(64, dtype=torch.float32)[:, None]),
        batch_size=4,
    )

    with pytest.raises(RuntimeError, match="loss blew up"):
        with _completed_fit_survives_teardown(trainer):
            trainer.fit(_FailingStep(), train_dataloaders=loader)

    assert not trainer.fit_loop.done


def test_clean_teardown_leaves_the_completed_fit_untouched() -> None:
    """The guard is inert when nothing fails during teardown."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        trainer = _fit_tiny_model_failing_teardown(monkeypatch, None, max_steps=5)

    assert trainer.global_step == 5
