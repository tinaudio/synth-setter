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
from pipeline.schemas.shard_metadata import ShardMetadata as ShardMetadata
from src.data.vst import param_specs

__all__ = [
    "DatasetSpec",
    "RenderConfig",
    "ShardMetadata",
    "ShardSpec",
    "dataset_config_id_from_path",
]

# Source-of-truth mapping from ``output_format`` to shard filename suffix.
# Adding a format means adding a row here; missing entries surface as KeyError
# at construction rather than producing a silently-wrong filename.
_OUTPUT_FORMAT_TO_EXTENSION: dict[str, str] = {"hdf5": ".h5", "wds": ".tar"}


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


# Names paired with ``train_val_test_sizes`` indices in error messages.
_SPLIT_LABELS: tuple[str, str, str] = ("train", "val", "test")


def _default_run_id(data: dict[str, Any]) -> str:
    """Compute a deterministic run_id from already-validated fields."""
    return make_dataset_wandb_run_id(
        DatasetConfigId(data["task_name"]), timestamp=data["created_at"]
    )


def _default_r2_prefix(data: dict[str, Any]) -> str:
    """Compute the R2 object prefix from already-validated fields."""
    return make_r2_prefix(
        DatasetConfigId(data["task_name"]),
        data["run_id"],
        prefix_root=data.get("r2_prefix_root", DEFAULT_R2_PREFIX_ROOT),
    )


class DatasetSpec(BaseModel):
    """Unified dataset specification — config + materialized runtime in one model.

    Construction story:

    - Hydra composes a dict from groups.
    - ``DatasetSpec(**dict)`` runs validation; runtime fields (git_sha,
      created_at, run_id, r2_prefix, is_repo_dirty) auto-fill via
      ``default_factory`` when missing and pass through when present.
    - Workers re-validate ``model_dump_json()`` from R2 and get an equal model.

    Strict mode is on (the model is a trust boundary for JSON-from-R2);
    ``extra="forbid"`` plus the per-field validators keep the boundary tight.
    Frozen so the materialized artifact is immutable post-construction.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    # Layout fields. Splits use ``list[int]`` (not tuple) so JSON round-trip
    # in strict mode round-trips natively without coercion.
    task_name: str
    output_format: Literal["hdf5", "wds"]
    train_val_test_sizes: list[int] = Field(min_length=3, max_length=3)
    train_val_test_seeds: list[int] = Field(min_length=3, max_length=3)
    base_seed: int
    r2_bucket: str
    r2_prefix_root: str = DEFAULT_R2_PREFIX_ROOT

    # Sub-model
    render: RenderConfig

    # Auto-filled runtime fields. The ``default_factory`` runs only when the
    # field is missing on input — JSON-loaded specs preserve the
    # materialization-time values that workers must reuse for consistency.
    # Lambdas wrap the module-level helpers so test monkeypatches take effect.
    git_sha: str = Field(default_factory=lambda: _get_git_sha())
    is_repo_dirty: bool = Field(default_factory=lambda: _is_repo_dirty())
    created_at: datetime = Field(default_factory=lambda: _utc_now())
    run_id: str = Field(default_factory=_default_run_id)
    r2_prefix: str = Field(default_factory=_default_r2_prefix)

    @model_validator(mode="before")
    @classmethod
    def _strip_computed_field_keys(cls, data: Any) -> Any:
        """Strip ``shards`` / ``num_shards`` / ``num_params`` from input.

        ``model_dump_json`` emits computed fields, so a JSON round-trip would
        otherwise trip ``extra="forbid"`` on the recomputed values.
        """
        if isinstance(data, dict):
            for computed_key in ("shards", "num_shards", "num_params"):
                data.pop(computed_key, None)
        return data

    @field_validator("created_at", mode="before")
    @classmethod
    def _parse_iso_datetime(cls, value: Any) -> Any:
        """Parse an ISO 8601 string into a ``datetime`` so strict mode accepts the value.

        Strict-mode Python validation rejects str → datetime coercion, so JSON inputs (where
        datetime is a string) need pre-conversion here. Also normalizes the trailing ``Z``
        offset that ``model_dump_json`` emits for UTC, which Python 3.10's ``fromisoformat``
        does not accept (3.11+ does).
        """
        if isinstance(value, str):
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        return value

    @field_validator("r2_prefix")
    @classmethod
    def _r2_prefix_must_end_with_slash(cls, value: str) -> str:
        """Reject prefixes lacking a trailing ``/`` so rclone never gets ".../prefixfilename"."""
        if not value.endswith("/"):
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
        for label, size in zip(_SPLIT_LABELS, self.train_val_test_sizes, strict=True):
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
        expected_ext = _OUTPUT_FORMAT_TO_EXTENSION[self.output_format]
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
        ext = _OUTPUT_FORMAT_TO_EXTENSION[self.output_format]
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
        return len(self.shards)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def num_params(self) -> int:
        return len(param_specs[self.render.param_spec_name])


def dataset_config_id_from_path(config_path: Path) -> DatasetConfigId:
    """Extract the dataset config ID (filename stem) from a config path."""
    return DatasetConfigId(config_path.stem)
