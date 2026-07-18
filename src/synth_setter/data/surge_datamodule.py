"""Compatibility exports for archived Surge datamodule targets."""

from __future__ import annotations

from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.data.lance_torch import LanceTensorMapDataset
from synth_setter.data.vst_datamodule import RawBatch, VSTDataModule, prepare_batch

SurgeXTDataset = LanceTensorMapDataset
SurgeDataModule = LanceVSTDataModule

__all__ = [
    "RawBatch",
    "SurgeDataModule",
    "SurgeXTDataset",
    "VSTDataModule",
    "prepare_batch",
]
