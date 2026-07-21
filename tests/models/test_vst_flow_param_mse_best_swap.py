"""Pins ``val/param_mse_best_swap`` / ``test/param_mse_best_swap`` logging.

The metric is unconditional and spec-free;
specs with interchangeable blocks (surge_simple) turn the metric on, degenerate
specs (surge_4) and the default ``None`` leave it off. Driven through a real CPU
``Trainer`` so the logged-metric names are the actual contract under test.
"""

from __future__ import annotations

from functools import partial

import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset

from synth_setter.metrics import BestSwapParamMSE
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_MEL_CHANNELS = 2
_MEL_N_MELS = 4
_MEL_N_FRAMES = 5
_D_MODEL = 16


class _TinyEncoder(torch.nn.Module):
    """Conditioning encoder mapping a mel spec to a ``(B, 1, _D_MODEL)`` token."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_MEL_CHANNELS * _MEL_N_MELS * _MEL_N_FRAMES, _D_MODEL)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Map ``mel_spec`` to a single conditioning token per sample.

        :param mel_spec: Batch of mel spectrograms.
        :returns: Conditioning tensor of shape ``(B, 1, _D_MODEL)``.
        """
        return self.linear(mel_spec.flatten(start_dim=1)).unsqueeze(1)


class _FakeBatchDataset(Dataset):
    """Fixed random samples shaped like the VST datamodule's batches."""

    def __init__(self, num_params: int) -> None:
        """Materialize the fixed samples.

        :param num_params: Width of each ``params`` row.
        """
        generator = torch.Generator().manual_seed(0)
        self._params = torch.rand(4, num_params, generator=generator)
        self._mels = torch.rand(4, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES, generator=generator)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample carrying the keys the module's step functions read.

        :param index: Sample index.
        :returns: ``params`` / ``mel_spec`` sample dict.
        """
        return {"params": self._params[index], "mel_spec": self._mels[index]}


def _flow_module(num_params: int) -> VSTFlowMatchingModule:
    """Build a tiny real flow-matching module with a 1-step sampler.

    :param num_params: Parameter-vector width.
    :returns: Module wired for the fake batch shapes.
    """
    vector_field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=_D_MODEL,
            d_token=_D_MODEL,
            num_params=num_params,
            num_tokens=4,
            initial_ffn=True,
            final_ffn=False,
        ),
        num_layers=1,
        d_model=_D_MODEL,
        conditioning_dim=_D_MODEL,
        num_heads=2,
        d_ff=_D_MODEL,
        num_tokens=4,
        learn_projection=True,
        time_encoding="sinusoidal",
        zero_init=False,
    )
    return VSTFlowMatchingModule(
        encoder=_TinyEncoder(),
        vector_field=vector_field,
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=num_params,
        validation_sample_steps=1,
        validation_cfg_strength=1.0,
        test_sample_steps=1,
        test_cfg_strength=1.0,
    )


def _tiny_trainer() -> Trainer:
    """Build a minimal CPU trainer for one validation/test batch.

    :returns: Silent single-batch CPU trainer.
    """
    return Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=1,
        limit_test_batches=1,
    )


def test_ctor_instantiates_best_swap_metrics_unconditionally() -> None:
    """Both loop metrics exist without any spec plumbing."""
    module = _flow_module(6)

    assert isinstance(module.val_param_mse_best_swap, BestSwapParamMSE)
    assert isinstance(module.test_param_mse_best_swap, BestSwapParamMSE)


def test_validation_loop_logs_best_swap_alongside_param_mse() -> None:
    """``val/param_mse_best_swap`` lands beside ``val/param_mse`` and never exceeds it."""
    module = _flow_module(6)
    loader = DataLoader(_FakeBatchDataset(6), batch_size=2)

    metrics = _tiny_trainer().validate(module, dataloaders=loader)[0]

    assert "val/param_mse_best_swap" in metrics
    assert "val/param_mse" in metrics
    assert metrics["val/param_mse_best_swap"] <= metrics["val/param_mse"] + 1e-6


def test_test_loop_logs_best_swap() -> None:
    """``test/param_mse_best_swap`` is logged by the test loop."""
    module = _flow_module(6)
    loader = DataLoader(_FakeBatchDataset(6), batch_size=2)

    metrics = _tiny_trainer().test(module, dataloaders=loader)[0]

    assert "test/param_mse_best_swap" in metrics
