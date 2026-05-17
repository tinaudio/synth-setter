"""Oracle Lightning module: returns ``batch["params"]`` as its prediction.

This is a drop-in replacement for
:class:`synth_setter.models.surge_ff_module.SurgeFeedForwardModule` that
short-circuits the model and returns the ground-truth parameters as predictions.
It is used to (a) smoke-test the train/eval pipeline end-to-end without the
cost of a real model, and (b) establish a performance ceiling for downstream
audio-metric evaluation — any divergence between oracle and ground truth in
downstream metrics is necessarily a pipeline issue, not a model issue.

The module preserves the public surface of ``SurgeFeedForwardModule``: same
``__init__`` signature, same 4-tuple ``model_step`` return shape, same
``predict_step`` tuple, and the same ``{"param_mse", "per_param_mse"}`` dict
out of ``validation_step`` / ``test_step`` that
:class:`synth_setter.utils.callbacks.LogPerParamMSE` reads. Substituting it
into a Hydra config requires no caller-side changes.
"""

from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule
from torch import nn


class FakeOracleNet(nn.Module):
    """Trivial ``nn.Module`` standing in for the real feature extractor.

    Holds a single grad-bearing parameter so the surrounding Lightning module
    has something for the optimizer to step on and ``loss.backward()`` has a
    valid graph. The forward pass returns this parameter unchanged, ignoring
    its input — the surrounding oracle module multiplies the result by zero
    before adding it to the loss, so the actual numeric value never matters.
    """

    def __init__(self, d_out: int):
        """Construct the oracle stand-in net.

        :param d_out: Output width the surrounding pipeline expects. The oracle
            does not use this value at forward time; it is stored only so Hydra
            configs can wire the same ``model.net.d_out`` key as the real model.
        """
        super().__init__()
        self.d_out = d_out
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Return the trainable dummy parameter, ignoring the input tensor.

        :param mel_spec: Unused mel-spectrogram batch — accepted for API parity with
            the real feature extractor.

        :return: The single-element ``dummy`` parameter, with grad attached.
        :rtype: torch.Tensor
        """
        del mel_spec
        return self.dummy


class SurgeFakeOracleModule(LightningModule):
    """LightningModule whose predictions are an oracle copy of ``batch["params"]``."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
        compile: bool = False,  # noqa: A002 — name preserved for SurgeFeedForwardModule config-swap parity
        warmup_steps: int = 0,
    ):
        """Mirror :class:`SurgeFeedForwardModule`'s signature so configs swap cleanly.

        :param net: Stand-in feature extractor; only its parameters matter (for the
            optimizer). The oracle does not consume its forward output as a prediction.
        :param optimizer: ``functools.partial``-style optimizer factory (Hydra
            ``_partial_: true``); invoked in :meth:`configure_optimizers`.
        :param scheduler: ``functools.partial``-style scheduler factory or ``None``.
        :param compile: Whether to ``torch.compile`` the net in :meth:`setup`.
        :param warmup_steps: If positive, wrap the scheduler with a linear warmup.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the oracle 4-tuple ``(loss, preds, targets, mel_spec)``.

        ``preds`` and ``targets`` both alias ``batch["params"]`` (the oracle is
        perfect by construction). ``loss`` is zero but carries a grad path
        through ``self.net``'s dummy parameter so ``loss.backward()`` works.

        :param batch: Dict with at least ``params`` and ``mel_spec``.

        :return: 4-tuple matching :meth:`SurgeFeedForwardModule.model_step`.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        target_params = batch["params"]
        mel_spec = batch["mel_spec"]
        loss = 0.0 * self.net(mel_spec).sum()
        return loss, target_params, target_params, mel_spec

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log ``train/loss`` (always 0); return the loss for Lightning.

        :param batch: Training batch dict.
        :param batch_idx: Index of this batch within the epoch (Lightning-supplied).
        :return: Scalar loss tensor with grad attached.
        :rtype: torch.Tensor
        """
        loss, *_ = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch: dict[str, torch.Tensor], log_key: str) -> dict[str, torch.Tensor]:
        """Shared val/test body — compute per-param MSE, log it, return the dict.

        :param batch: Evaluation batch dict.
        :param log_key: Lightning log key (``"val/param_mse"`` or ``"test/param_mse"``).

        :return: Dict with ``param_mse`` (scalar) and ``per_param_mse`` (shape ``(P,)``).
        :rtype: dict[str, torch.Tensor]
        """
        _, preds, targets, _ = self.model_step(batch)
        per_param_mse = (preds - targets).square().mean(dim=0)
        param_mse = per_param_mse.mean()
        self.log(log_key, param_mse, on_step=False, on_epoch=True, prog_bar=True)
        return {"param_mse": param_mse, "per_param_mse": per_param_mse}

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Log ``val/param_mse`` and return the dict :class:`LogPerParamMSE` reads.

        :param batch: Validation batch dict.
        :param batch_idx: Index of this batch within the val epoch (Lightning-supplied).

        :return: Dict with ``param_mse`` (scalar) and ``per_param_mse`` (shape ``(P,)``).
        :rtype: dict[str, torch.Tensor]
        """
        del batch_idx
        return self._eval_step(batch, log_key="val/param_mse")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Mirror :meth:`validation_step` for the ``trainer.test()`` phase.

        :param batch: Test batch dict.
        :param batch_idx: Index of this batch within the test epoch (Lightning-supplied).

        :return: Dict with the same shape as :meth:`validation_step`'s return.
        :rtype: dict[str, torch.Tensor]
        """
        del batch_idx
        return self._eval_step(batch, log_key="test/param_mse")

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(batch["params"], batch)`` — the tuple shape ``PredictionWriter`` unpacks.

        :param batch: Predict batch dict.
        :param batch_idx: Index of this batch within the predict pass (Lightning-supplied).
        :return: Tuple of (oracle predictions, original batch).
        :rtype: tuple[torch.Tensor, dict[str, torch.Tensor]]
        """
        return batch["params"], batch

    def setup(self, stage: str) -> None:
        """Optionally ``torch.compile`` the net — kept for surge_ff_module parity.

        :param stage: Lightning lifecycle stage ("fit", "validate", "test", "predict").
        """
        del stage
        if not self.hparams.compile:
            return
        self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Instantiate the optimizer and (optional) warmup/scheduler chain.

        Mirrors :meth:`SurgeFeedForwardModule.configure_optimizers` so the same
        Hydra ``optimizer`` and ``scheduler`` partial blocks instantiate without
        modification.

        :return: Lightning's optimizer-config dict — either ``{"optimizer": opt}`` or
            ``{"optimizer": opt, "lr_scheduler": {...}}``.
        :rtype: dict[str, Any]
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        warmup_scheduler = (
            torch.optim.lr_scheduler.LinearLR(optimizer, 1e-10, 1.0, self.hparams.warmup_steps)
            if self.hparams.warmup_steps > 0
            else None
        )
        scheduler = (
            self.hparams.scheduler(optimizer=optimizer)
            if self.hparams.scheduler is not None
            else None
        )

        if warmup_scheduler is not None and scheduler is None:
            scheduler = warmup_scheduler
        elif warmup_scheduler is not None and scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.hparams.warmup_steps],
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
