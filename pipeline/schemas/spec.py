from __future__ import annotations

import json
import plistlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from pipeline.schemas.config import DatasetConfig, SplitsConfig
from pipeline.schemas.prefix import (
    DatasetConfigId,
    DatasetRunId,
    R2Prefix,
    make_dataset_wandb_run_id,
    make_r2_prefix,
)
from src.data.vst import param_specs

# Pinned Surge XT renderer version baked into tinaudio/synth-setter:dev-snapshot
# (built from SURGE_GIT_REF=f7b97c68 — release-xt/1.3.4). materialize_spec sets
# this directly into DatasetPipelineSpec.renderer_version so the launcher's
# code path stays interpreter-only (no pedalboard.VST3Plugin instantiation, no
# X display dependency). The worker validates the running plugin against this
# constant via check_renderer_version before rendering — see
# pipeline/entrypoints/generate_dataset.py. Bump together with SURGE_GIT_REF.
SURGE_XT_RENDERER_VERSION = "1.3.4"


def extract_renderer_version(plugin_path: Path) -> str:
    """Extract the version string from a VST3 plugin bundle.

    Tries the static-metadata files first (`Contents/moduleinfo.json` on Linux,
    `Contents/Info.plist` on macOS), then falls back to loading the plugin via
    pedalboard and reading `plugin.version`. The fallback requires a usable
    X11 display, so the launcher does not call this — it pins
    `renderer_version` to `SURGE_XT_RENDERER_VERSION` and lets the worker
    compare against this function's output (see
    `pipeline.entrypoints.generate_dataset.run`).

    Raises:
        FileNotFoundError: plugin_path does not exist.
        RuntimeError: version cannot be extracted by any method.
        json.JSONDecodeError: moduleinfo.json is malformed.
        plistlib.InvalidFileException: Info.plist is malformed.
    """
    if not plugin_path.exists():
        raise FileNotFoundError(f"Plugin path does not exist: {plugin_path}")

    moduleinfo = plugin_path / "Contents" / "moduleinfo.json"
    if moduleinfo.is_file():
        return json.loads(moduleinfo.read_text())["Version"]

    plist = plugin_path / "Contents" / "Info.plist"
    if plist.is_file():
        return plistlib.loads(plist.read_bytes())["CFBundleShortVersionString"]

    # Pedalboard fallback: prebuilt plugin bundles (e.g. Surge XT shipped via
    # .deb) don't always carry moduleinfo.json. Loading the .so via pedalboard
    # gives us VST3 factory metadata; this requires X11, so callers in
    # interpreter-only contexts (the SkyPilot launcher) must avoid it.
    from pedalboard import VST3Plugin  # noqa: PLC0415

    plugin = VST3Plugin(str(plugin_path))
    version = plugin.version
    if not version:
        raise RuntimeError(f"Could not extract version from {plugin_path}")
    return version


class ShardSpec(BaseModel):
    """Per-shard identity and pre-computed derived values."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int
    filename: str  # "shard-000000.h5"
    seed: int  # base_seed + shard_id


class DatasetPipelineSpec(BaseModel):
    """Frozen runtime specification materialized from DatasetConfig."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    run_id: DatasetRunId  # unique run ID: {config_id}-{YYYYMMDDTHHMMSSZ}
    r2_prefix: R2Prefix  # R2 storage path: data/{config_id}/{run_id}/
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
    r2_bucket: str  # Cloudflare R2 bucket name for spec + shard uploads
    splits: SplitsConfig  # train/val/test shard counts
    plugin_path: str  # VST3 plugin to render through
    preset_path: str  # VST preset to load
    channels: int  # audio channels (e.g. 2 for stereo)
    velocity: int  # MIDI velocity for note rendering
    signal_duration_seconds: float  # audio length per sample in seconds
    min_loudness: float  # loudness floor — retry if below
    sample_batch_size: int  # batch size for generation efficiency
    shards: tuple[ShardSpec, ...]  # pre-computed per-shard identity (id, filename, seed)

    @property
    def num_shards(self) -> int:
        """Number of shards, derived from the shards tuple length."""
        return len(self.shards)

    @field_validator("r2_prefix")
    @classmethod
    def _r2_prefix_must_end_with_slash(cls, value: str) -> str:
        # Upload paths concat r2_prefix with filenames (e.g. f"{prefix}{INPUT_SPEC_FILENAME}");
        # missing trailing slash silently produces a wrong key like ".../prefixinput_spec.json".
        if not value.endswith("/"):
            raise ValueError(f"r2_prefix must end with '/' (got: {value!r})")
        return value

    @field_validator("r2_bucket")
    @classmethod
    def _r2_bucket_must_not_be_blank(cls, value: str) -> str:
        # DatasetConfig.r2_bucket has the same validator; this mirror enforces the
        # invariant when specs come from a non-config path (hand-edited, externally
        # materialized) so rclone never receives a malformed `r2:/...` destination.
        if not value.strip():
            raise ValueError("r2_bucket must not be blank")
        return value


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
    plugin = Path(config.plugin_path)
    if not plugin.exists():
        raise FileNotFoundError(f"Plugin path does not exist: {config.plugin_path}")

    created_at = datetime.now(timezone.utc)
    return _build_pipeline_spec(
        config=config,
        config_id=config_id,
        code_version=_get_git_sha(),
        is_repo_dirty=_is_repo_dirty(),
        renderer_version=SURGE_XT_RENDERER_VERSION,
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
        r2_prefix=make_r2_prefix(config_id, run_id),
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
        r2_bucket=config.r2_bucket,
        splits=config.splits,
        plugin_path=config.plugin_path,
        preset_path=config.preset_path,
        channels=config.channels,
        velocity=config.velocity,
        signal_duration_seconds=config.signal_duration_seconds,
        min_loudness=config.min_loudness,
        sample_batch_size=config.sample_batch_size,
        shards=shards,
    )
