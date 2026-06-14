"""Deprecation shim re-exporting :mod:`synth_setter.data.vst_datamodule`.

Archived W&B run configs and external job scripts resolve ``_target_`` paths
under this old module name; the symbols now live in ``vst_datamodule``. See #1664.
"""

from __future__ import annotations

from synth_setter.data.vst_datamodule import (
    ShardColumn,
    ShardFile,
    ShiftedBatchSampler,
    ShuffledSampler,
    SurgeDataModule,
    SurgeXTDataset,
    VSTDataModule,
    VSTDataset,
    WithinChunkShuffledSampler,
)

__all__ = [
    "ShardColumn",
    "ShardFile",
    "ShiftedBatchSampler",
    "ShuffledSampler",
    "SurgeDataModule",
    "SurgeXTDataset",
    "VSTDataModule",
    "VSTDataset",
    "WithinChunkShuffledSampler",
]
