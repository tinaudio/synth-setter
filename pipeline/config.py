from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from pipeline.prefix import DatasetConfigId


class SplitsConfig(BaseModel):
    """Train/val/test shard counts."""

    model_config = ConfigDict(strict=True, extra="forbid")

    train: int
    val: int
    test: int


class DatasetConfig(BaseModel):
    """Validated dataset generation configuration."""

    model_config = ConfigDict(strict=True, extra="forbid")

    param_spec: str
    plugin_path: str
    output_format: Literal["hdf5", "wds"] = "hdf5"
    sample_rate: int
    shard_size: int
    num_shards: int
    base_seed: int
    splits: SplitsConfig
    preset_path: str
    channels: int
    velocity: int
    signal_duration_seconds: float
    min_loudness: float
    sample_batch_size: int

    @model_validator(mode="after")
    def _splits_sum_to_num_shards(self) -> DatasetConfig:
        """Validate that train + val + test shard counts equal num_shards."""
        total = self.splits.train + self.splits.val + self.splits.test
        if total != self.num_shards:
            raise ValueError(f"splits sum ({total}) != num_shards ({self.num_shards})")
        return self

    @model_validator(mode="after")
    def _positive_sizes(self) -> DatasetConfig:
        """Validate that numeric fields have sensible ranges."""
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if not (0 <= self.velocity <= 127):
            raise ValueError("velocity must be in [0, 127]")
        if self.signal_duration_seconds <= 0:
            raise ValueError("signal_duration_seconds must be positive")
        if self.sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be positive")
        return self


def load_dataset_config(config_path: Path) -> DatasetConfig:
    """Load and validate a dataset generation config from a YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}, got {type(raw).__name__}")
    return DatasetConfig(**raw)


def dataset_config_id_from_path(config_path: Path) -> DatasetConfigId:
    """Extract the dataset config ID (filename stem) from a config path."""
    return DatasetConfigId(config_path.stem)
