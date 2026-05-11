"""Unified dataset specification: a single Pydantic model is the spec on R2.

``DatasetSpec`` replaces the prior split between ``DatasetConfig`` (the YAML-
shaped config) and ``DatasetPipelineSpec`` (the runtime-materialized artifact).
Hydra composes a dict from groups; the entrypoint constructs ``DatasetSpec``
directly from that dict on line 1 of ``main``. Runtime fields (``git_sha``,
``created_at``, ``run_id``, ``r2_prefix``) auto-fill via ``default_factory``
when missing and pass through when present (worker reconstruction from JSON).
``shards``/``num_shards``/``num_params`` are computed deterministically from
layout + render fields.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from pipeline.schemas.prefix import (
    DEFAULT_R2_PREFIX_ROOT,
    DatasetConfigId,
    make_dataset_wandb_run_id,
    make_r2_prefix,
)
from src.data.vst import param_specs

__all__ = [
    "OUTPUT_FORMAT_TO_EXTENSION",
    "DatasetSpec",
    "RenderConfig",
    "ShardSpec",
    "dataset_config_id_from_path",
    "load_dataset_spec_yaml",
]

# Source-of-truth mapping from ``output_format`` to shard filename suffix.
# Adding a format means adding a row here; missing entries surface as KeyError
# at construction rather than producing a silently-wrong filename.
OUTPUT_FORMAT_TO_EXTENSION: dict[str, str] = {"hdf5": ".h5"}

# YAML key in legacy ``configs/dataset/*.yaml`` files that names a base config
# whose keys are merged under the override. Removed in Phase A.3 once Hydra
# defaults inheritance handles composition.
_LEGACY_EXTENDS_KEY = "_extends"


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


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ShardSpec(BaseModel):
    """Per-shard identity and pre-computed derived values."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int
    filename: str
    seed: int


class RenderConfig(BaseModel):
    """Renderer-specific configuration nested as ``DatasetSpec.render``.

    Carries every parameter the per-shard writer needs to produce audio +
    parameter arrays for its assigned shard. ``param_spec_name`` is resolved
    against the in-process registry inside the writer (not at the launcher),
    so launcher-side construction stays interpreter-only.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    plugin_path: str
    preset_path: str
    param_spec_name: str
    renderer_version: str
    sample_rate: int
    channels: int
    velocity: int
    signal_duration_seconds: float
    min_loudness: float
    sample_batch_size: int
    batch_per_shard: int

    @model_validator(mode="after")
    def _ranges_must_be_sane(self) -> RenderConfig:
        """Reject out-of-range numeric inputs and blank required strings at construction."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if not (0 <= self.velocity <= 127):
            raise ValueError("velocity must be in [0, 127]")
        if self.signal_duration_seconds <= 0:
            raise ValueError("signal_duration_seconds must be positive")
        if self.sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be positive")
        if self.batch_per_shard <= 0:
            raise ValueError("batch_per_shard must be positive")
        if not self.param_spec_name.strip():
            raise ValueError("param_spec_name must not be blank")
        if not self.renderer_version.strip():
            raise ValueError("renderer_version must not be blank")
        return self


class DatasetSpec(BaseModel):
    """Unified dataset specification — config + materialized runtime in one model.

    Construction story:

    - Hydra composes a dict from groups.
    - ``DatasetSpec(**dict)`` runs validation; runtime fields (git_sha,
      created_at, run_id, r2_prefix, is_repo_dirty) auto-fill via
      ``default_factory`` when missing and pass through when present.
    - Workers re-validate ``model_dump_json()`` from R2 and get an equal model.

    ``strict`` is intentionally off on this top-level model so JSON-mode
    round-trips coerce list→tuple and str→datetime (JSON has no native tuple
    or datetime types). ``extra="forbid"`` plus the per-field validators keep
    the trust boundary tight.
    """

    model_config = ConfigDict(extra="forbid")

    # Layout fields
    task_name: str
    output_format: Literal["hdf5"]
    train_val_test_sizes: tuple[int, int, int]
    train_val_test_seeds: tuple[int, int, int]
    base_seed: int
    r2_bucket: str
    r2_prefix_root: str = DEFAULT_R2_PREFIX_ROOT

    # Sub-model
    render: RenderConfig

    # Auto-filled runtime fields. ``default_factory`` runs only when the field
    # is missing in input — JSON-loaded specs preserve the materialization-time
    # values that the worker needs to remain consistent across the run. The
    # lambdas re-look-up the helpers per call so test monkeypatches take effect.
    git_sha: str = Field(default_factory=lambda: _get_git_sha())
    is_repo_dirty: bool = Field(default_factory=lambda: _is_repo_dirty())
    created_at: datetime = Field(default_factory=lambda: _utc_now())
    run_id: str = ""
    r2_prefix: str = ""

    @model_validator(mode="before")
    @classmethod
    def _strip_computed_field_keys(cls, data: Any) -> Any:
        """Strip ``shards`` / ``num_shards`` / ``num_params`` from input.

        ``model_dump_json`` emits computed fields, so a JSON round-trip would
        otherwise trip ``extra="forbid"`` on the recomputed values. Copies the
        input mapping so callers that hold a reference (logging, retries) see
        their dict unchanged.
        """
        if isinstance(data, dict):
            data = dict(data)
            for computed_key in ("shards", "num_shards", "num_params"):
                data.pop(computed_key, None)
        return data

    @model_validator(mode="after")
    def _populate_derived_runtime_fields(self) -> DatasetSpec:
        """Fill ``run_id`` / ``r2_prefix`` from ``task_name`` + ``created_at`` when empty.

        Empty defaults mean the input dict didn't carry materialization-time values; non-empty
        values came from a JSON-loaded spec and pass through unchanged.
        """
        if not self.run_id:
            object.__setattr__(
                self,
                "run_id",
                make_dataset_wandb_run_id(
                    DatasetConfigId(self.task_name), timestamp=self.created_at
                ),
            )
        if not self.r2_prefix:
            object.__setattr__(
                self,
                "r2_prefix",
                make_r2_prefix(
                    DatasetConfigId(self.task_name),
                    self.run_id,
                    prefix_root=self.r2_prefix_root,
                ),
            )
        return self

    @field_validator("r2_prefix")
    @classmethod
    def _r2_prefix_must_end_with_slash(cls, value: str) -> str:
        """Reject prefixes lacking a trailing ``/`` so rclone never gets ".../prefixfilename"."""
        if value and not value.endswith("/"):
            raise ValueError(f"r2_prefix must end with '/' (got: {value!r})")
        return value

    @field_validator("r2_bucket")
    @classmethod
    def _r2_bucket_must_not_be_blank(cls, value: str) -> str:
        """Reject blank buckets so rclone never receives a malformed ``r2:/...`` destination."""
        if not value.strip():
            raise ValueError("r2_bucket must not be blank")
        return value

    @field_validator("task_name")
    @classmethod
    def _task_name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank ``task_name`` so derived run_id / r2_prefix are never empty-prefixed."""
        if not value.strip():
            raise ValueError("task_name must not be blank")
        return value

    @model_validator(mode="after")
    def _split_sizes_must_be_multiples_of_batch_per_shard(self) -> DatasetSpec:
        """Each split's sample count must divide cleanly into shards.

        The renderer writes one shard at a time at ``batch_per_shard`` rows
        per shard; a split size that doesn't divide evenly would either drop
        the remainder or ship a ragged final shard — both surprises caught at
        spec-validation time rather than mid-render.
        """
        bps = self.render.batch_per_shard
        for label, size in zip(("train", "val", "test"), self.train_val_test_sizes, strict=True):
            if size < 0:
                raise ValueError(f"train_val_test_sizes[{label}] must be non-negative, got {size}")
            if size % bps != 0:
                raise ValueError(
                    f"train_val_test_sizes[{label}]={size} is not a multiple of "
                    f"render.batch_per_shard={bps}"
                )
        if sum(self.train_val_test_sizes) == 0:
            raise ValueError("train_val_test_sizes must sum to a positive count")
        return self

    @model_validator(mode="after")
    def _shard_filenames_match_output_format(self) -> DatasetSpec:
        """Defense-in-depth: every computed shard filename ends with the format's extension."""
        expected_ext = OUTPUT_FORMAT_TO_EXTENSION[self.output_format]
        for shard in self.shards:
            if not shard.filename.endswith(expected_ext):
                raise ValueError(
                    f"shard {shard.shard_id} filename {shard.filename!r} does not match "
                    f"output_format {self.output_format!r} (expected suffix {expected_ext!r})"
                )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def shards(self) -> tuple[ShardSpec, ...]:
        """Shard identities derived from total sample counts and ``batch_per_shard``."""
        bps = self.render.batch_per_shard
        total_shards = sum(self.train_val_test_sizes) // bps
        ext = OUTPUT_FORMAT_TO_EXTENSION[self.output_format]
        return tuple(
            ShardSpec(
                shard_id=i,
                filename=f"shard-{i:06d}{ext}",
                seed=self.base_seed + i,
            )
            for i in range(total_shards)
        )

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def num_shards(self) -> int:
        """Total number of shards across all splits."""
        return len(self.shards)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def num_params(self) -> int:
        """Total encoded parameter count looked up by name in the param-spec registry."""
        return len(param_specs[self.render.param_spec_name])


def dataset_config_id_from_path(config_path: Path) -> DatasetConfigId:
    """Extract the dataset config ID (filename stem) from a config path."""
    return DatasetConfigId(config_path.stem)


def _resolve_legacy_extends(config_path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Merge a base config into ``raw`` if it declares ``_extends:``.

    Bridges the old flat ``configs/dataset/*.yaml`` shape until Phase A.3
    migrates the entrypoint to ``@hydra.main`` and Hydra's defaults
    inheritance replaces this loader.
    """
    if _LEGACY_EXTENDS_KEY not in raw:
        return raw
    base_name = raw[_LEGACY_EXTENDS_KEY]
    if not isinstance(base_name, str):
        raise TypeError(
            f"{_LEGACY_EXTENDS_KEY} must name a base config (string), "
            f"got {type(base_name).__name__}"
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
    base_resolved = _resolve_legacy_extends(base_path, base_raw)
    override = {k: v for k, v in raw.items() if k != _LEGACY_EXTENDS_KEY}
    merged = OmegaConf.merge(OmegaConf.create(base_resolved), OmegaConf.create(override))
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


# Pinned plugin version baked into ``tinaudio/synth-setter:dev-snapshot``;
# legacy YAML files predate ``render.renderer_version`` so the loader injects
# this default. Phase A.2 surfaces it as a Hydra-composed config field.
_LEGACY_PINNED_RENDERER_VERSION = "1.3.4"


def _legacy_dict_to_dataset_spec_kwargs(raw: dict[str, Any], task_name: str) -> dict[str, Any]:
    """Reshape a legacy flat dataset YAML dict into ``DatasetSpec`` kwargs.

    Maps ``shard_size``/``num_shards``/``splits`` onto ``train_val_test_sizes``
    and lifts renderer fields under ``render``. Removed in Phase A.3 with the
    rest of the legacy YAML loader.
    """
    splits = raw["splits"]
    shard_size = raw["shard_size"]
    declared_num_shards = raw.get("num_shards")
    splits_total_shards = splits["train"] + splits["val"] + splits["test"]
    if declared_num_shards is not None and declared_num_shards != splits_total_shards:
        raise ValueError(
            f"legacy YAML num_shards={declared_num_shards} disagrees with "
            f"sum(splits)={splits_total_shards} (train={splits['train']}, "
            f"val={splits['val']}, test={splits['test']}); update one or the other"
        )
    train_val_test_sizes = (
        splits["train"] * shard_size,
        splits["val"] * shard_size,
        splits["test"] * shard_size,
    )
    train_val_test_seeds = (
        raw["base_seed"],
        raw["base_seed"] + 1,
        raw["base_seed"] + 2,
    )
    render = {
        "plugin_path": raw["plugin_path"],
        "preset_path": raw["preset_path"],
        "param_spec_name": raw["param_spec"],
        "renderer_version": raw.get("renderer_version", _LEGACY_PINNED_RENDERER_VERSION),
        "sample_rate": raw["sample_rate"],
        "channels": raw["channels"],
        "velocity": raw["velocity"],
        "signal_duration_seconds": raw["signal_duration_seconds"],
        "min_loudness": raw["min_loudness"],
        "sample_batch_size": raw["sample_batch_size"],
        "batch_per_shard": shard_size,
    }
    return {
        "task_name": task_name,
        "output_format": raw.get("output_format", "hdf5"),
        "train_val_test_sizes": train_val_test_sizes,
        "train_val_test_seeds": train_val_test_seeds,
        "base_seed": raw["base_seed"],
        "r2_bucket": raw["r2_bucket"],
        "render": render,
    }


def load_dataset_spec_yaml(config_path: Path) -> DatasetSpec:
    """Load a legacy flat dataset YAML and construct a ``DatasetSpec``.

    Bridge for callers that haven't migrated to ``@hydra.main``. Removed in
    Phase A.3 once the entrypoint composes the spec dict via Hydra defaults.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}, got {type(raw).__name__}")
    merged = _resolve_legacy_extends(config_path, raw)
    kwargs = _legacy_dict_to_dataset_spec_kwargs(merged, task_name=config_path.stem)
    return DatasetSpec(**kwargs)
