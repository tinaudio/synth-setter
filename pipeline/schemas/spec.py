from __future__ import annotations

import json
import plistlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from pipeline.schemas.config import DatasetConfig, SplitsConfig
from pipeline.schemas.prefix import DatasetConfigId, make_dataset_wandb_run_id
from src.data.vst import param_specs


class ShardSpec(BaseModel):
    """Per-shard identity and pre-computed derived values."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int
    filename: str  # "shard-000000.h5"
    seed: int  # base_seed + shard_id


class DatasetPipelineSpec(BaseModel):
    """Frozen runtime specification materialized from DatasetConfig."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    run_id: str  # unique run ID: {config_id}-{YYYYMMDDTHHMMSSZ}
    created_at: datetime  # UTC, timezone-aware materialization timestamp
    code_version: str  # git commit SHA at materialization time
    is_repo_dirty: bool  # True if working tree had uncommitted changes
    param_spec: str  # name of param spec in registry (e.g. "surge_simple")
    renderer_version: str  # VST plugin version, extracted from plugin bundle
    output_format: Literal["hdf5", "wds"]  # shard file format
    sample_rate: int  # audio sample rate in Hz
    shard_size: int  # rows (samples) per shard
    base_seed: int  # deterministic seed base; per-shard seed = base_seed + shard_id
    num_params: int  # total encoded param count from param_spec registry
    splits: SplitsConfig  # train/val/test shard counts
    plugin_path: str  # VST3 plugin to render through
    preset_path: str  # VST preset to load
    velocity: int  # MIDI velocity for note rendering
    signal_duration_seconds: float  # audio length per sample in seconds
    min_loudness: float  # loudness floor — retry if below
    sample_batch_size: int  # batch size for generation efficiency
    shards: tuple[ShardSpec, ...]  # pre-computed per-shard identity (id, filename, seed)

    @property
    def num_shards(self) -> int:
        """Number of shards, derived from the shards tuple length."""
        return len(self.shards)

    @model_validator(mode="after")
    def _plugin_path_exists(self) -> DatasetPipelineSpec:
        """Validate that the plugin path exists on disk."""
        plugin = Path(self.plugin_path)
        if not plugin.exists():
            raise ValueError(f"Plugin path does not exist: {self.plugin_path}")
        return self


def extract_renderer_version(plugin_path: Path) -> str:
    """Extract version string from a VST3 plugin bundle.

    Checks Linux moduleinfo.json first (production path), then macOS Info.plist.

    Raises:
        FileNotFoundError: If neither version file exists.
        KeyError: If the version field is missing.
        json.JSONDecodeError: If moduleinfo.json is malformed.
        plistlib.InvalidFileException: If Info.plist is malformed.
    """
    moduleinfo = plugin_path / "Contents" / "moduleinfo.json"
    if moduleinfo.is_file():
        data = json.loads(moduleinfo.read_text())
        return data["Version"]

    plist = plugin_path / "Contents" / "Info.plist"
    if plist.is_file():
        data = plistlib.loads(plist.read_bytes())
        return data["CFBundleShortVersionString"]

    raise FileNotFoundError(f"No moduleinfo.json or Info.plist in {plugin_path}/Contents/")


def _get_git_sha() -> str:
    """Get the current git commit SHA."""
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _is_repo_dirty() -> bool:
    """Check if the git working tree has uncommitted changes."""
    result = subprocess.run(["git", "diff", "--quiet"], capture_output=True)  # noqa: S603, S607
    return result.returncode != 0


def materialize_spec(
    config: DatasetConfig,
    config_id: DatasetConfigId,
) -> DatasetPipelineSpec:
    """Materialize a frozen DatasetPipelineSpec from config and environment.

    Derives all runtime state internally: git SHA, repo dirty status,
    renderer version from plugin path, current UTC timestamp.
    """
    created_at = datetime.now(timezone.utc)
    return _build_pipeline_spec(
        config=config,
        config_id=config_id,
        code_version=_get_git_sha(),
        is_repo_dirty=_is_repo_dirty(),
        renderer_version=extract_renderer_version(Path(config.plugin_path)),
        created_at=created_at,
    )


def _build_pipeline_spec(
    config: DatasetConfig,
    config_id: DatasetConfigId,
    *,
    code_version: str,
    is_repo_dirty: bool,
    renderer_version: str,
    created_at: datetime,
) -> DatasetPipelineSpec:
    """Build a DatasetPipelineSpec from config and pre-resolved runtime values.

    This is the pure functional core — no I/O, no side effects.
    """
    if config.output_format != "hdf5":
        raise NotImplementedError(f"Output format {config.output_format!r} not yet supported")

    run_id = make_dataset_wandb_run_id(config_id, timestamp=created_at)

    shards = tuple(
        ShardSpec(
            shard_id=i,
            filename=f"shard-{i:06d}.h5",
            seed=config.base_seed + i,
        )
        for i in range(config.num_shards)
    )

    return DatasetPipelineSpec(
        run_id=run_id,
        created_at=created_at,
        code_version=code_version,
        is_repo_dirty=is_repo_dirty,
        param_spec=config.param_spec,
        renderer_version=renderer_version,
        output_format=config.output_format,
        sample_rate=config.sample_rate,
        shard_size=config.shard_size,
        base_seed=config.base_seed,
        num_params=len(param_specs[config.param_spec]),
        splits=config.splits,
        plugin_path=config.plugin_path,
        preset_path=config.preset_path,
        velocity=config.velocity,
        signal_duration_seconds=config.signal_duration_seconds,
        min_loudness=config.min_loudness,
        sample_batch_size=config.sample_batch_size,
        shards=shards,
    )
