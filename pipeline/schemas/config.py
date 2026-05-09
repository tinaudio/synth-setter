from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from pipeline.schemas.prefix import DatasetConfigId

# Key in a sibling dataset config that names a base config (without the ``.yaml``
# suffix); the loader merges the base under the override before constructing
# DatasetConfig. Lets two configs that differ only in output_format share a
# parent without duplicating every field.
_EXTENDS_KEY = "_extends"


class SplitsConfig(BaseModel):
    """Train/val/test shard counts."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

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
    r2_bucket: str
    splits: SplitsConfig
    preset_path: str
    channels: int
    velocity: int
    signal_duration_seconds: float
    min_loudness: float
    sample_batch_size: int
    # Number of single-node SkyPilot clusters the launcher fans out in parallel for this
    # dataset. Default 1 matches the pre-config launcher default; the launcher's
    # `--num-workers` overrides this when explicitly passed.
    num_workers: int = 1

    @field_validator("r2_bucket")
    @classmethod
    def _r2_bucket_must_not_be_blank(cls, v: str) -> str:
        """Reject empty or whitespace-only strings."""
        if not v.strip():
            raise ValueError("r2_bucket must not be blank")
        return v

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
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        return self


def _resolve_extends(config_path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Merge a base config into ``raw`` if ``raw[_EXTENDS_KEY]`` names one.

    The base is resolved relative to ``config_path``'s directory; the override
    wins on every overlapping key. Nesting one level deep is supported (a base
    can itself extend another); to keep the surface tight, we recurse and let
    OmegaConf's merge handle the rest.
    """
    if _EXTENDS_KEY not in raw:
        return raw
    base_name = raw[_EXTENDS_KEY]
    if not isinstance(base_name, str):
        raise TypeError(
            f"{_EXTENDS_KEY} must name a base config (string), got {type(base_name).__name__}"
        )
    base_path = config_path.parent / f"{base_name}.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(
            f"_extends target not found: {base_path} (referenced from {config_path})"
        )
    with open(base_path) as f:
        base_raw = yaml.safe_load(f)
    if not isinstance(base_raw, dict):
        raise TypeError(f"Expected a YAML mapping in {base_path}, got {type(base_raw).__name__}")
    base_resolved = _resolve_extends(base_path, base_raw)
    override = {k: v for k, v in raw.items() if k != _EXTENDS_KEY}
    merged = OmegaConf.merge(OmegaConf.create(base_resolved), OmegaConf.create(override))
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def load_dataset_config(config_path: Path) -> DatasetConfig:
    """Load and validate a dataset generation config from a YAML file.

    If the file declares ``_extends: <name>``, the loader merges the named base
    config (resolved relative to ``config_path``'s directory) under the file's
    own keys before validation.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}, got {type(raw).__name__}")
    merged = _resolve_extends(config_path, raw)
    return DatasetConfig(**merged)


def dataset_config_id_from_path(config_path: Path) -> DatasetConfigId:
    """Extract the dataset config ID (filename stem) from a config path."""
    return DatasetConfigId(config_path.stem)
