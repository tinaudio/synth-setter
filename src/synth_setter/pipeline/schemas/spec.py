"""Unified dataset specification: a single Pydantic model is the spec on R2.

``DatasetSpec`` is the only spec model — there is no separate YAML-shaped
config or runtime-materialized artifact. Hydra composes a dict from groups;
the entrypoint constructs ``DatasetSpec`` directly from that dict on line 1
of ``main``. Runtime fields (``git_sha``, ``created_at``, ``run_id``,
``r2_prefix``) auto-fill via ``default_factory`` when missing and pass through
when present (worker reconstruction from JSON). ``shards``/``num_shards``/
``num_params`` are computed deterministically from layout + render fields.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from synth_setter.pipeline.schemas.prefix import (
    DEFAULT_R2_PREFIX_ROOT,
    DatasetConfigId,
    make_dataset_wandb_run_id,
    make_r2_prefix,
)

__all__ = [
    "EXTENSION_TO_OUTPUT_FORMAT",
    "OUTPUT_FORMAT_TO_EXTENSION",
    "DatasetSpec",
    "RenderConfig",
    "ShardSpec",
]

# Source-of-truth mapping from ``output_format`` to shard filename suffix.
# Adding a format means adding a row here; missing entries surface as KeyError
# at construction rather than producing a silently-wrong filename.
OUTPUT_FORMAT_TO_EXTENSION: dict[str, str] = {"hdf5": ".h5", "wds": ".tar"}

# Reverse lookup for dispatching shard writers/validators by file suffix;
# derived from the forward map so adding a format stays a one-place edit.
# Two formats must never share an extension — a collision would silently drop
# an entry from the dict-comprehension (last-key-wins) and route the wrong
# format downstream, so guard at import time.
EXTENSION_TO_OUTPUT_FORMAT: dict[str, str] = {v: k for k, v in OUTPUT_FORMAT_TO_EXTENSION.items()}
if len(EXTENSION_TO_OUTPUT_FORMAT) != len(OUTPUT_FORMAT_TO_EXTENSION):
    raise RuntimeError(
        "Duplicate extensions in OUTPUT_FORMAT_TO_EXTENSION — "
        "two output formats map to the same suffix: "
        f"{OUTPUT_FORMAT_TO_EXTENSION!r}"
    )


# Sentinel returned by ``_get_git_sha`` when called outside a git working
# tree (worker host without ``.git/``, fresh tarball extract, etc.). Workers
# normally receive ``git_sha`` populated in the JSON spec from R2, so the
# default_factory only fires when something has gone off-script — returning
# a sentinel rather than raising lets the failure surface as a clear
# "git_sha=git-unavailable" in the spec JSON rather than a CalledProcessError
# deep in pydantic's default_factory.
_GIT_UNAVAILABLE_SENTINEL = "git-unavailable"


def _get_git_sha() -> str:
    """Get the current git commit SHA, or a sentinel if unavailable."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return _GIT_UNAVAILABLE_SENTINEL
    return result.stdout.strip()


def _is_repo_dirty() -> bool:
    """Check if the git working tree has uncommitted changes (False if no git).

    ``git diff --quiet`` exits 0 (clean) or 1 (dirty) inside a repo, and 128
    when run outside one (``fatal: not a git repository``). Treat exit codes
    outside {0, 1} as "no usable git" — same contract as ``_get_git_sha``'s
    sentinel: a worker on a tarball-extracted host gets a benign default
    rather than a confusing ``is_repo_dirty=True``.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "diff", "--quiet"],  # noqa: S607
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    if result.returncode not in (0, 1):
        return False
    return result.returncode != 0


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _current_platform() -> str:  # noqa: DOC201,DOC203
    """Return ``sys.platform`` via a patchable indirection (tests patch this, not ``sys``)."""
    return sys.platform


_GuiToggleCadence = Literal["never", "once", "render"]
_PluginReloadCadence = Literal["once", "render"]


def _default_gui_toggle_cadence() -> _GuiToggleCadence:  # noqa: DOC201,DOC203
    """Return ``"never"`` on Darwin (validator rejects ``"render"`` — #714), else ``"once"``.

    ``"once"`` is the safer per-shard default on non-Darwin (one ``show_editor``
    warm-up per shard rather than per render). The historical pre-cadence
    behaviour ("warm up on every load") corresponds to ``"render"`` and is
    available as an opt-in.
    """
    return "never" if _current_platform() == "darwin" else "once"


class ShardSpec(BaseModel):
    """Per-shard identity and pre-computed derived values."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shard_id: int = Field(
        description="Logical shard index (0-based), independent of compute infrastructure."
    )
    filename: str = Field(
        description=(
            "Shard filename including the format-specific suffix "
            "(``shard-NNNNNN.h5`` or ``shard-NNNNNN.tar``)."
        )
    )
    seed: int = Field(description="Per-shard RNG seed, derived as ``base_seed + shard_id``.")


class RenderConfig(BaseModel):
    """Renderer-specific configuration nested as ``DatasetSpec.render``.

    Carries every parameter the per-shard writer needs to produce audio +
    parameter arrays for its assigned shard. ``param_spec_name`` is resolved
    against the in-process registry inside the writer (not at the launcher),
    so launcher-side construction stays interpreter-only.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    plugin_path: str = Field(
        description="Filesystem path to the VST3 plugin bundle the worker loads."
    )
    preset_path: str = Field(
        description=(
            "Filesystem path to the ``.fxp``/``.vstpreset`` baseline preset loaded before "
            "random parameter override."
        )
    )
    param_spec_name: str = Field(
        description=(
            "Key into the in-process param-spec registry; resolved inside the worker, "
            "not the launcher."
        )
    )
    renderer_version: str = Field(
        description="Renderer code-path version stamp recorded in shard provenance."
    )
    sample_rate: int = Field(description="Audio sample rate in Hz.")
    channels: int = Field(description="Audio channel count.")
    velocity: int = Field(description="MIDI velocity used for every render in this run (0-127).")
    signal_duration_seconds: float = Field(
        description="Duration of each rendered audio sample, in seconds."
    )
    min_loudness: float = Field(
        description="Per-sample loudness floor; renders quieter than this are rejected/retried."
    )
    samples_per_render_batch: int = Field(
        default=32,
        description="Batch size the renderer uses inside a shard.",
    )
    samples_per_shard: int = Field(
        description="Samples written per shard; each split size must be a multiple of this."
    )
    plugin_reload_cadence: _PluginReloadCadence = Field(
        default="render",
        description=(
            'How often to reload the plugin within a shard: ``"once"`` loads + applies '
            'the preset once per shard and reuses the cached instance; ``"render"`` '
            "(default, historical per-#489 behaviour) reloads on every render."
        ),
    )
    gui_toggle_cadence: _GuiToggleCadence = Field(
        default_factory=_default_gui_toggle_cadence,
        description=(
            'How often to run the ``show_editor`` warm-up on the plugin: ``"never"`` '
            'skips it entirely, ``"once"`` runs it once per shard (default on non-Darwin), '
            '``"render"`` runs it before every render. Darwin rejects ``"render"`` '
            '(SIGTRAP after ~3-4 calls, #714); the default factory yields ``"never"`` on '
            "Darwin."
        ),
    )

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
        if self.samples_per_render_batch <= 0:
            raise ValueError("samples_per_render_batch must be positive")
        if self.samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive")
        if not self.param_spec_name.strip():
            raise ValueError("param_spec_name must not be blank")
        if not self.renderer_version.strip():
            raise ValueError("renderer_version must not be blank")
        return self

    @model_validator(mode="after")
    def _gui_toggle_cadence_forbids_render_on_darwin(self) -> RenderConfig:  # noqa: DOC201,DOC203,DOC501,DOC503
        """Reject ``gui_toggle_cadence="render"`` on Darwin (SIGTRAP after ~3-4 calls, #714).

        ``"once"`` is permitted because a single ``show_editor`` call sits below
        the empirical SIGTRAP threshold.
        """
        if self.gui_toggle_cadence == "render" and _current_platform() == "darwin":
            raise ValueError(
                'gui_toggle_cadence="render" is not supported on Darwin: '
                "show_editor accumulates AppKit/CGS commit-handler state per "
                "call in unbundled python and triggers SIGTRAP after ~3-4 "
                'plugin reloads (#714). Use "once" or "never" on Darwin.'
            )
        return self


# Names paired with ``train_val_test_sizes`` indices in error messages.
_SPLIT_LABELS: tuple[str, str, str] = ("train", "val", "test")


def _default_run_id(data: dict[str, Any]) -> str:
    """Compute a deterministic run_id from already-validated layout fields."""
    return make_dataset_wandb_run_id(
        DatasetConfigId(data["task_name"]), timestamp=data["created_at"]
    )


def _default_r2_prefix(data: dict[str, Any]) -> str:
    """Compute the R2 object prefix from already-validated layout fields."""
    return make_r2_prefix(
        DatasetConfigId(data["task_name"]),
        data["run_id"],
        prefix_root=data["r2_prefix_root"],
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

    # Splits stored as immutable tuples; JSON lists are coerced by _splits_list_to_tuple.
    task_name: str = Field(
        description=(
            "Dataset config identifier; becomes the prefix of ``run_id`` and the "
            "config-id path segment of ``r2_prefix`` (``<root>/<task_name>/<run_id>/``)."
        )
    )
    output_format: Literal["hdf5", "wds"] = Field(
        description=(
            "Shard container format; ``hdf5`` writes ``.h5``, ``wds`` writes WebDataset ``.tar``."
        )
    )
    train_val_test_sizes: tuple[int, int, int] = Field(
        description=(
            "Sample counts per split; each entry must be a multiple of "
            "``render.samples_per_shard``."
        )
    )
    # Enforced by _reject_train_val_test_seeds.
    train_val_test_seeds: tuple[int, int, int] | None = Field(
        default=None,
        description=(
            "Reserved for per-sample seeding (#884); must be ``None`` until implemented — "
            "any non-None value raises ``NotImplementedError`` at construction."
        ),
    )
    base_seed: int = Field(
        description="Seed used to derive per-shard ``ShardSpec.seed`` values (``base_seed + shard_id``)."
    )
    r2_bucket: str = Field(
        description="Cloudflare R2 bucket name where shards and metadata are written."
    )
    r2_prefix_root: str = Field(
        default=DEFAULT_R2_PREFIX_ROOT,
        description="Top-level prefix segment under the bucket; slashes are stripped by ``make_r2_prefix``.",
    )

    render: RenderConfig = Field(
        description="Nested ``RenderConfig`` carrying every per-shard renderer input."
    )

    # Auto-filled runtime fields: factories fire only when the value is missing on
    # input, so JSON-loaded specs preserve materialization-time values. The lambda
    # wrappers defer lookup so ``monkeypatch.setattr`` on the module attr is
    # honored at call time.
    git_sha: str = Field(
        default_factory=lambda: _get_git_sha(),
        description=(
            "Commit SHA of the launcher's working tree at construction; sentinel "
            "``git-unavailable`` when not in a git repo."
        ),
    )
    is_repo_dirty: bool = Field(
        default_factory=lambda: _is_repo_dirty(),
        description=(
            "Whether the launcher's working tree had uncommitted changes at construction."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: _utc_now(),
        description=(
            "UTC timestamp when the spec was first constructed; preserved across "
            "worker round-trips via JSON."
        ),
    )
    run_id: str = Field(
        default_factory=_default_run_id,
        description="Deterministic W&B run ID derived from ``task_name`` and ``created_at``.",
    )
    r2_prefix: str = Field(
        default_factory=_default_r2_prefix,
        description="Full R2 object prefix (``<root>/<task_name>/<run_id>/``); must end with ``/``.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_train_val_test_seeds(cls, data: Any) -> Any:
        """Reject any non-None ``train_val_test_seeds`` — reserved for #884, not implemented.

        Runs in ``mode="before"`` so ``NotImplementedError`` propagates as-is
        instead of being wrapped in a ``ValidationError`` (which is what
        pydantic does for ``ValueError`` raised inside field validators).
        """
        if isinstance(data, dict) and data.get("train_val_test_seeds") is not None:
            raise NotImplementedError(
                "train_val_test_seeds is reserved for per-sample seeding (#884) "
                "and is not yet implemented; omit the field"
            )
        return data

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
            for computed_key in cls.model_computed_fields:
                data.pop(computed_key, None)
        return data

    @field_validator("train_val_test_sizes", mode="before")
    @classmethod
    def _splits_list_to_tuple(cls, value: Any) -> Any:
        """Coerce JSON-loaded ``list[int]`` into ``tuple[int, int, int]``.

        Tuples are immutable, so the frozen-model guarantee carries through to
        the contents (a mutable list would let ``spec.train_val_test_sizes[0] =
        x`` silently invalidate the cached ``shards`` / ``num_shards``). JSON
        has no native tuple, so model_dump_json emits a list and the worker
        round-trip lands here on validation; in-process construction can pass
        either form.
        """
        if isinstance(value, list):
            if len(value) != 3:
                raise ValueError(f"must have exactly 3 entries, got {len(value)}")
            return tuple(value)
        return value

    @field_validator("created_at", mode="before")
    @classmethod
    def _parse_iso_datetime(cls, value: Any) -> Any:
        """Parse an ISO 8601 string into a tz-aware UTC ``datetime``.

        Strict-mode Python validation rejects str → datetime coercion, so JSON inputs (where
        datetime is a string) need pre-conversion here. Also normalizes the trailing ``Z``
        offset that ``model_dump_json`` emits for UTC, which Python 3.10's ``fromisoformat``
        does not accept (3.11+ does).

        Rejects naive datetimes and non-UTC offsets so error attribution stays at the
        ``created_at`` boundary rather than surfacing later as a ``run_id`` derivation crash
        (``make_dataset_wandb_run_id`` requires tz-aware UTC).
        """
        if isinstance(value, str):
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            value = datetime.fromisoformat(value)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError(
                    f"created_at must be timezone-aware UTC; got naive datetime {value!r}"
                )
            if value.utcoffset() != timedelta(0):
                raise ValueError(
                    f"created_at must be UTC (offset 0); got offset {value.utcoffset()}"
                )
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

    @field_validator("r2_prefix_root")
    @classmethod
    def _r2_prefix_root_must_not_be_blank(cls, value: str) -> str:
        """Reject blank prefix roots so derived ``r2_prefix`` doesn't start with a stray ``/``."""
        if not value.strip():
            raise ValueError("r2_prefix_root must not be blank")
        return value

    @field_validator("task_name")
    @classmethod
    def _task_name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank ``task_name`` so derived run_id / r2_prefix are never empty-prefixed."""
        if not value.strip():
            raise ValueError("task_name must not be blank")
        return value

    @model_validator(mode="after")
    def _split_sizes_must_be_multiples_of_samples_per_shard(self) -> DatasetSpec:
        """Each split's sample count must divide cleanly into shards.

        The renderer writes one shard at a time at ``samples_per_shard`` rows
        per shard; a split size that doesn't divide evenly would either drop
        the remainder or ship a ragged final shard — both surprises caught at
        spec-validation time rather than mid-render.
        """
        sps = self.render.samples_per_shard
        for label, size in zip(_SPLIT_LABELS, self.train_val_test_sizes, strict=True):
            if size < 0:
                raise ValueError(f"train_val_test_sizes[{label}] must be non-negative, got {size}")
            if size % sps != 0:
                raise ValueError(
                    f"train_val_test_sizes[{label}]={size} is not a multiple of "
                    f"render.samples_per_shard={sps}"
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
        """Shard identities derived from total sample counts and ``samples_per_shard``."""
        sps = self.render.samples_per_shard
        total_shards = sum(self.train_val_test_sizes) // sps
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
        """Total encoded parameter count looked up by name in the param-spec registry.

        Imported from ``param_spec_registry`` (not ``synth_setter.data.vst``) so that
        ``model_dump_json`` — which evaluates this computed field — does not
        transitively pull ``pedalboard`` into the launcher.
        """
        from synth_setter.data.vst.param_spec_registry import param_specs

        return len(param_specs[self.render.param_spec_name])
