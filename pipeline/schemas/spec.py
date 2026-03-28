from __future__ import annotations

import json
import plistlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from pipeline.schemas.config import DatasetConfig, SplitsConfig
from pipeline.schemas.prefix import DatasetConfigId, make_dataset_wandb_run_id
from src.data.vst import param_specs

# Hardcoded by the generator (generate_vst_dataset.py:194).
# 128 mel bins from librosa default, 401 frames from 100fps hop at 44.1kHz/4s.
# Changing these requires changing the generator — don't make configurable.
_MEL_BINS = 128
_MEL_FRAMES = 401

_EXPECTED_HDF5_DATASETS = ("audio", "mel_spec", "param_array")


class ShardSpec(BaseModel):
    """Per-shard generation specification.

    Shapes are per-row (single sample):
    - audio_shape: (channels, samples) where samples = sample_rate * duration
    - mel_shape: (mels, frames) — per-channel; full HDF5 row is (channels, *mel_shape)
    - param_shape: (num_params,) — total encoded parameter count from param_spec
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int
    filename: str
    seed: int
    row_start: int
    row_count: int
    expected_datasets: tuple[str, ...]
    audio_shape: tuple[int, int]
    mel_shape: tuple[int, int]
    param_shape: tuple[int]


class PipelineSpec(BaseModel):
    """Frozen runtime specification materialized from DatasetConfig."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    run_id: str
    created_at: datetime  # UTC, timezone-aware
    code_version: str  # git commit SHA
    is_repo_dirty: bool
    param_spec: str
    renderer_version: str
    output_format: Literal["hdf5", "wds"]
    sample_rate: int
    shard_size: int
    num_shards: int
    base_seed: int
    splits: SplitsConfig
    shards: tuple[ShardSpec, ...]


def extract_renderer_version(plugin_path: Path) -> str:
    """Extract version string from a VST3 plugin bundle.

    Checks Linux moduleinfo.json first (production path), then macOS Info.plist.

    Raises:
        FileNotFoundError: If neither version file exists.
        KeyError: If the version field is missing.
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
) -> PipelineSpec:
    """Materialize a frozen PipelineSpec from config and environment.

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
) -> PipelineSpec:
    """Build a PipelineSpec from config and pre-resolved runtime values.

    This is the pure functional core — no I/O, no side effects.
    """
    if config.output_format != "hdf5":
        raise NotImplementedError(f"Output format {config.output_format!r} not yet supported")

    run_id = make_dataset_wandb_run_id(config_id, timestamp=created_at)
    num_params = len(param_specs[config.param_spec])
    audio_shape = (config.channels, int(config.sample_rate * config.signal_duration_seconds))

    shards = tuple(
        ShardSpec(
            shard_id=i,
            filename=f"shard-{i:06d}.h5",
            seed=config.base_seed + i,
            row_start=i * config.shard_size,
            row_count=config.shard_size,
            expected_datasets=tuple(_EXPECTED_HDF5_DATASETS),
            audio_shape=audio_shape,
            mel_shape=(_MEL_BINS, _MEL_FRAMES),
            param_shape=(num_params,),
        )
        for i in range(config.num_shards)
    )

    return PipelineSpec(
        run_id=run_id,
        created_at=created_at,
        code_version=code_version,
        is_repo_dirty=is_repo_dirty,
        param_spec=config.param_spec,
        renderer_version=renderer_version,
        output_format=config.output_format,
        sample_rate=config.sample_rate,
        shard_size=config.shard_size,
        num_shards=config.num_shards,
        base_seed=config.base_seed,
        splits=config.splits,
        shards=shards,
    )
