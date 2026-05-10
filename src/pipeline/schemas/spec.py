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
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from src.pipeline.schemas.prefix import (
    DEFAULT_R2_PREFIX_ROOT,
    DatasetConfigId,
    make_dataset_wandb_run_id,
    make_r2_prefix,
)
from src.pipeline.schemas.shard_metadata import ShardMetadata as ShardMetadata

__all__ = [
    "DatasetSpec",
    "RenderConfig",
    "ShardMetadata",
    "ShardSpec",
    "dataset_config_id_from_path",
]

# Source-of-truth mapping from ``output_format`` to shard filename suffix.
# Adding a format means adding a row here; missing entries surface as KeyError
# at construction rather than producing a silently-wrong filename. Wrapped in
# ``MappingProxyType`` so the ALL_CAPS name actually denotes an immutable
# constant.
OUTPUT_FORMAT_TO_EXTENSION: Mapping[str, str] = MappingProxyType({"hdf5": ".h5", "wds": ".tar"})


# Pin git invocations to the repo root so spec construction works regardless
# of the caller's CWD (workers, ad-hoc scripts, IDEs, …). spec.py lives at
# ``<repo>/src/pipeline/schemas/spec.py`` so the repo root is four levels up.
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]


# Sentinel returned by ``_get_git_sha`` when called outside a git working
# tree (worker host without ``.git/``, fresh extract from a tarball, etc).
# Workers normally receive ``git_sha`` populated in the JSON spec from R2,
# so the default_factory only fires when something has gone off-script —
# returning a sentinel rather than raising lets the failure surface as a
# clear "spec serialized with git_sha=git-unavailable" rather than a
# CalledProcessError deep in pydantic's default_factory.
_GIT_UNAVAILABLE_SENTINEL = "git-unavailable"


def _get_git_sha() -> str:
    """Get the current git commit SHA at the repo root, or a sentinel if unavailable."""
    try:
        result = subprocess.run(  # noqa: S603 — git is a fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607 — relying on PATH-resolved git is fine here
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return _GIT_UNAVAILABLE_SENTINEL
    return result.stdout.strip()


def _is_repo_dirty() -> bool:
    """Check if the repo's git working tree has uncommitted changes (False if no git)."""
    try:
        result = subprocess.run(  # noqa: S603 — git is a fixed argv, no shell
            ["git", "diff", "--quiet"],  # noqa: S607 — relying on PATH-resolved git is fine here
            cwd=_REPO_ROOT,
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode != 0


def _utc_now() -> datetime:
    """Return ``datetime.now`` in UTC; isolated for test monkeypatching."""
    return datetime.now(timezone.utc)


class ShardSpec(BaseModel):
    """Per-shard identity and pre-computed derived values."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int
    filename: str
    # Derived value, not yet plumbed into the writer. Tracked in #884.
    seed: int


class RenderConfig(BaseModel):
    """Renderer-specific configuration nested as ``DatasetSpec.render``.

    Carries every parameter the per-shard writer needs to produce audio +
    parameter arrays for its assigned shard. ``param_spec_name`` is resolved
    against the in-process registry inside the writer; the registry is also
    consulted at construction (lazily, via ``param_spec_registry``) for an
    early validity check, so a typo surfaces as a ``ValidationError`` rather
    than a ``KeyError`` during JSON serialization. The lazy import points at
    ``src.data.vst.param_spec_registry`` (not ``src.data.vst``) so spec
    construction stays interpreter-only — pedalboard is never pulled.
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

    @field_validator("param_spec_name")
    @classmethod
    def _param_spec_name_must_be_registered(cls, value: str) -> str:
        """Reject unknown ``param_spec_name`` at construction.

        Without this check, the error surfaces as a raw ``KeyError`` deep
        inside ``DatasetSpec.num_params`` (e.g., during ``model_dump_json``),
        producing a stack trace instead of a clean validation message.
        """
        from src.data.vst.param_spec_registry import param_specs

        if value not in param_specs:
            valid = sorted(param_specs.keys())
            raise ValueError(f"param_spec_name {value!r} not in registry; valid: {valid}")
        return value

    @model_validator(mode="after")
    def _ranges_must_be_sane(self) -> RenderConfig:
        """Reject out-of-range numeric fields and a blank renderer_version."""
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
    # Per-sample reproducible seeding is not implemented yet — see #884.
    # Field is kept on the model for forward compatibility but must remain
    # empty until the writer learns to consume it; the
    # ``_train_val_test_seeds_unsupported`` validator below enforces that.
    train_val_test_seeds: list[int] = Field(default_factory=list)
    base_seed: int
    r2_bucket: str
    r2_prefix_root: str = DEFAULT_R2_PREFIX_ROOT

    # Sub-model
    render: RenderConfig

    # Auto-filled runtime fields. The factory runs only when the field is
    # missing on input — JSON-loaded specs preserve the materialization-time
    # values workers must reuse. The lambdas around ``_get_git_sha`` etc.
    # defer the lookup to call time so tests that ``monkeypatch.setattr`` on
    # the module attribute reach this resolution path.
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
        otherwise trip ``extra="forbid"`` on the recomputed values. Copy the
        input first so callers passing a reused mapping (Hydra/OmegaConf
        containers, test fixtures) don't see their dict mutated.
        """
        if isinstance(data, dict):
            data = dict(data)
            for computed_key in cls.model_computed_fields:
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

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_utc(cls, value: datetime) -> datetime:
        """Reject naive or non-UTC ``created_at`` so the run_id/prefix invariants hold.

        ``run_id`` and ``r2_prefix`` are derived from ``created_at`` and the project treats
        them as UTC by construction (see ``_utc_now``). A JSON spec providing a naive
        datetime *and* a pre-computed ``run_id`` would otherwise pass through with a
        wall-clock timestamp, breaking that invariant silently.
        """
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError(f"created_at must be timezone-aware UTC, got {value!r}")
        return value

    @field_validator("r2_prefix")
    @classmethod
    def _r2_prefix_must_end_with_slash(cls, value: str) -> str:
        """Reject prefixes lacking a trailing ``/`` so rclone never gets ".../prefixfilename"."""
        if not value.endswith("/"):
            raise ValueError(f"r2_prefix must end with '/' (got: {value!r})")
        return value

    @field_validator("r2_prefix_root")
    @classmethod
    def _r2_prefix_root_must_not_be_blank(cls, value: str) -> str:
        """Reject blank prefix roots so derived ``r2_prefix`` doesn't start with a stray ``/``."""
        if not value.strip():
            raise ValueError("r2_prefix_root must not be blank")
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
        """Reject a blank ``task_name`` so derived run_id / r2_prefix stay well-formed."""
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
    def _train_val_test_seeds_unsupported(self) -> DatasetSpec:
        """Reject populated ``train_val_test_seeds`` until per-sample seeding lands.

        The field is advertised on the spec but neither the writer nor the sampler consume it
        today, so a non-empty list would silently ship a non-reproducible dataset. Tracked in #884.
        """
        if self.train_val_test_seeds:
            raise NotImplementedError(
                "per-sample reproducible seeding is not implemented yet — see "
                "https://github.com/tinaudio/synth-setter/issues/884"
            )
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
        """Total shards across all splits (derived from ``self.shards``)."""
        return len(self.shards)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def num_params(self) -> int:
        """Encoded parameter count for ``render.param_spec_name``.

        Imported from ``param_spec_registry`` (not ``src.data.vst``) so that
        ``model_dump_json`` — which evaluates this computed field — does not
        transitively pull ``pedalboard`` into the launcher.
        """
        from src.data.vst.param_spec_registry import param_specs

        return len(param_specs[self.render.param_spec_name])


def dataset_config_id_from_path(config_path: Path) -> DatasetConfigId:
    """Extract the dataset config ID (filename stem) from a config path."""
    return DatasetConfigId(config_path.stem)
