"""Deprecation shim re-exporting :mod:`synth_setter.data.vst_datamodule`.

Archived W&B run configs and external job scripts resolve ``_target_`` paths
under this old module name; the symbols now live in ``vst_datamodule``. See #1664.
"""

from __future__ import annotations

from synth_setter.data.lance_datamodule import LanceVSTDataModule, LanceVSTDataset
from synth_setter.data.vst_datamodule import (
    RawBatch,
    ShardColumn,
    ShardFile,
    ShiftedBatchSampler,
    ShuffledSampler,
    VSTDataModule,
    VSTDataset,
    WithinChunkShuffledSampler,
    prepare_batch,
)

# Deprecated aliases bind to the concrete Lance classes: the abstract bases no
# longer open any storage format, and archived ``_target_``s must stay runnable.
SurgeXTDataset = LanceVSTDataset
SurgeDataModule = LanceVSTDataModule

__all__ = [
    "RawBatch",
    "ShardColumn",
    "ShardFile",
    "ShiftedBatchSampler",
    "ShuffledSampler",
    "SurgeDataModule",
    "SurgeXTDataset",
    "VSTDataModule",
    "VSTDataset",
    "WithinChunkShuffledSampler",
    "prepare_batch",
]
