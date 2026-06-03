"""Unified dataset specification: a single Pydantic model is the spec on R2.

``DatasetSpec`` is the only spec model — there is no separate YAML-shaped
config or runtime-materialized artifact. Hydra composes a dict from groups;
the entrypoint constructs ``DatasetSpec`` directly from that dict on line 1
of ``main``. Runtime fields (``git_sha``, ``created_at``, ``run_id``,
``r2.prefix``) auto-fill via ``default_factory`` when missing and pass through
when present (worker reconstruction from JSON). ``shards``/``num_shards``/
``num_params`` are computed deterministically from layout + render fields.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
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
from synth_setter.pipeline.schemas.r2_location import R2Location

__all__ = [
    "EXTENSION_TO_OUTPUT_FORMAT",
    "OUTPUT_FORMAT_TO_EXTENSION",
    "DatasetSpec",
    "R2Location",
    "RenderConfig",
    "ShardSpec",
    "Split",
]

# Flat-form keys promoted into the nested ``r2`` dict by the back-compat shim.
# Maps the legacy top-level key → the nested ``R2Location`` field. Anchored
# here (not on ``DatasetSpec``) so the dict literal is the source of truth and
# can't drift from the validator that consumes it.
_LEGACY_FLAT_R2_KEYS: dict[str, str] = {
    "r2_bucket": "bucket",
    "r2_prefix_root": "prefix_root",
    "r2_prefix": "prefix",
}

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


def _current_platform() -> str:
    """Return ``sys.platform`` via a patchable indirection (tests patch this, not ``sys``).

    :return: Current ``sys.platform`` string.
    """
    return sys.platform


_GuiToggleCadence = Literal["never", "once", "render", "always_on"]
_PluginReloadCadence = Literal["once", "render"]


def _default_gui_toggle_cadence() -> _GuiToggleCadence:
    """Return ``"never"`` on Darwin (validator rejects ``"render"`` — #714), else ``"render"``.

    Non-Darwin keeps the historical per-render warm-up so this config switch
    doesn't change production behaviour; ``"once"`` is opt-in.

    :return: ``"never"`` on Darwin, otherwise ``"render"``.
    """
    return "never" if _current_platform() == "darwin" else "render"


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
    max_retries: int = Field(
        default=0,
        ge=0,
        description=(
            "Per-shard retry budget for transient renderer-subprocess failures "
            "(CalledProcessError). 0 keeps strict fail-fast."
        ),
    )
    parallel: bool = Field(
        default=False,
        description=(
            "When True, generate() dispatches shard renders concurrently with "
            "pool size = min(max(1, available_cpus() // 2), len(my_range)). "
            "Applies on both local-run and SkyPilot-worker contexts; peak "
            "local disk scales with pool size."
        ),
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
            'How often to realise the plugin editor during the shard: ``"never"`` '
            'skips it entirely, ``"once"`` warms once per shard, ``"render"`` warms '
            "before every render (default on non-Darwin, matching historical "
            'per-render warm-up), ``"always_on"`` holds the editor open for the '
            "whole shard render on a background thread (requires "
            '``plugin_reload_cadence="once"``). Darwin rejects ``"render"`` '
            "(SIGTRAP after ~3-4 calls, #714); "
            '``"always_on"`` is permitted on Darwin because it opens the editor '
            "once per shard, not cumulatively. The default factory yields "
            '``"never"`` on Darwin.'
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
    def _gui_toggle_cadence_forbids_render_on_darwin(self) -> RenderConfig:
        """Reject ``gui_toggle_cadence="render"`` on Darwin (SIGTRAP after ~3-4 calls, #714).

        ``"once"`` is permitted because a single ``show_editor`` call sits below
        the empirical SIGTRAP threshold.

        :return: ``self`` unchanged when the combination is permitted.
        :raises ValueError: ``gui_toggle_cadence="render"`` combined with Darwin.
        """
        if self.gui_toggle_cadence == "render" and _current_platform() == "darwin":
            raise ValueError(
                'gui_toggle_cadence="render" is not supported on Darwin: '
                "show_editor accumulates AppKit/CGS commit-handler state per "
                "call in unbundled python and triggers SIGTRAP after ~3-4 "
                'plugin reloads (#714). Use "once" or "never" on Darwin.'
            )
        return self

    @model_validator(mode="after")
    def _always_on_requires_plugin_reload_once(self) -> RenderConfig:
        """Reject ``gui_toggle_cadence="always_on"`` unless the plugin is loaded once per shard.

        Holding the editor open binds it to a single live ``VST3Plugin`` instance;
        reloading per render would invalidate the editor handle mid-shard.

        :return: ``self`` unchanged when the combination is permitted.
        :raises ValueError: ``gui_toggle_cadence="always_on"`` combined with
            ``plugin_reload_cadence != "once"``.
        """
        if self.gui_toggle_cadence == "always_on" and self.plugin_reload_cadence != "once":
            raise ValueError(
                'gui_toggle_cadence="always_on" requires plugin_reload_cadence="once" '
                "so the editor stays bound to a single live plugin instance for the "
                'whole shard. Set plugin_reload_cadence="once" to opt in.'
            )
        return self


# Names paired with ``train_val_test_sizes`` indices in error messages.
_SPLIT_LABELS: tuple[str, str, str] = ("train", "val", "test")

# Typed alias for split names — narrows the ``str`` parameter on layout helpers
# (``R2Location.split_h5_uri``, ``DatasetSpec.split_shard_ranges`` keys) so a
# typo lands as a type error rather than a silent miss at runtime.
Split = Literal["train", "val", "test"]


def _default_run_id(data: dict[str, Any]) -> str:
    """Compute a deterministic run_id from already-validated layout fields."""
    return make_dataset_wandb_run_id(
        DatasetConfigId(data["task_name"]), timestamp=data["created_at"]
    )


def _default_r2_location(data: dict[str, Any]) -> dict[str, Any]:
    """Build a partial ``r2`` dict (no ``bucket``) when the ``r2`` field was omitted.

    The DatasetSpec model_validator promotes the legacy flat keys and fills
    partial ``r2`` dicts before this factory ever fires — this path covers the
    "no ``r2`` block at all" case. ``bucket`` is intentionally omitted so the
    nested ``R2Location`` validator fails with Pydantic's standard missing-
    required-field error on ``r2.bucket`` (rather than this factory inventing
    a placeholder that would mask the real misconfiguration).

    :param data: Already-validated DatasetSpec field data exposed to the factory.
    :returns: Dict shaped like ``R2Location.model_fields`` minus ``bucket``;
        ``R2Location`` validation then raises the missing-field error.
    """
    return {
        "prefix_root": DEFAULT_R2_PREFIX_ROOT,
        "prefix": make_r2_prefix(
            DatasetConfigId(data["task_name"]),
            data["run_id"],
            prefix_root=DEFAULT_R2_PREFIX_ROOT,
        ),
    }


def _coerce_created_at_to_datetime(value: Any) -> datetime | None:
    """Best-effort parse of ``created_at`` for pre-validation prefix derivation.

    The ``mode='before'`` model validator sees raw input — Python datetimes for
    in-process construction, ISO strings for JSON-loaded specs.

    :param value: Raw ``created_at`` input from the user dict (datetime, string, or other).
    :returns: A tz-aware UTC datetime when parsing succeeds, else ``None``. ``None``
        signals the caller to fall back to the field's ``default_factory`` so the
        ``created_at`` field validator can surface the proper error attribution.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed


def _fill_default_r2_prefix(data: dict[str, Any], r2: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``r2`` with ``prefix`` derived from layout fields when missing.

    Mirrors the prior ``_default_r2_prefix`` factory exactly:
    ``<prefix_root>/<task_name>/<run_id>/``. ``run_id`` and ``created_at`` are
    materialized in ``data`` when absent (using the same factories the field
    defaults would have used), so the worker's JSON-round-trip preservation
    contract holds end-to-end: any value derived here is observed by the
    field validators too. The shim falls back to the caller's ``r2`` unchanged
    whenever it can't safely build a prefix — every such path leaves a
    downstream validator (on ``created_at``, ``task_name``, ``prefix_root``)
    in charge of the error attribution.

    :param data: Raw input dict to ``DatasetSpec`` (mutated in-place when run_id /
        created_at need filling so the field defaults observe the same value).
    :param r2: Raw nested ``r2`` sub-dict (may be missing ``prefix``).
    :returns: A ``r2`` dict either filled with a derived ``prefix`` or returned
        verbatim when prefix derivation is not safely available.
    """
    if not _can_derive_prefix(data, r2):
        return r2
    if data.get("created_at") is None:
        data["created_at"] = _utc_now()
    if not data.get("run_id"):
        created_at = _coerce_created_at_to_datetime(data["created_at"])
        if created_at is None:
            return r2
        data["run_id"] = make_dataset_wandb_run_id(
            DatasetConfigId(data["task_name"]), timestamp=created_at
        )
    filled = dict(r2)
    filled["prefix"] = make_r2_prefix(
        DatasetConfigId(data["task_name"]),
        data["run_id"],
        prefix_root=filled.get("prefix_root", DEFAULT_R2_PREFIX_ROOT),
    )
    return filled


def _can_derive_prefix(data: dict[str, Any], r2: dict[str, Any]) -> bool:
    """Return True when layout fields can produce a derived ``prefix`` cleanly.

    Defers cases that would otherwise trip ``make_r2_prefix`` (blank task_name,
    blank prefix_root) to the field validators downstream so the right boundary
    surfaces the error.

    :param data: Raw input dict to ``DatasetSpec``.
    :param r2: Raw nested ``r2`` sub-dict.
    :returns: ``True`` iff prefix derivation will succeed without raising.
    """
    task_name = data.get("task_name")
    if not isinstance(task_name, str) or not task_name.strip():
        return False
    prefix_root = r2.get("prefix_root", DEFAULT_R2_PREFIX_ROOT)
    if not isinstance(prefix_root, str) or not prefix_root.strip("/").strip():
        return False
    return True


class DatasetSpec(BaseModel):
    """Unified dataset specification — config + materialized runtime in one model.

    Construction story:

    - Hydra composes a dict from groups.
    - ``DatasetSpec(**dict)`` runs validation; runtime fields (git_sha,
      created_at, run_id, r2.prefix, is_repo_dirty) auto-fill via
      ``default_factory`` when missing and pass through when present.
    - Workers re-validate ``model_dump_json()`` from R2 and get an equal model.

    Strict mode is on (the model is a trust boundary for JSON-from-R2);
    ``extra="forbid"`` plus the per-field validators keep the boundary tight.
    Frozen so the materialized artifact is immutable post-construction.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: task_name

        Dataset config identifier; prefix of ``run_id`` and ``r2.prefix``.

    .. attribute :: output_format

        Shard container format (``hdf5`` writes ``.h5``, ``wds`` writes ``.tar``).

    .. attribute :: train_val_test_sizes

        Sample counts per split; each entry must be a multiple of
        ``render.samples_per_shard``.

    .. attribute :: train_val_test_seeds

        Reserved for per-sample seeding (#884); must be ``None``.

    .. attribute :: base_seed

        Seed used to derive per-shard ``ShardSpec.seed`` values.

    .. attribute :: render

        Nested ``RenderConfig`` carrying every per-shard renderer input.

    .. attribute :: mask_degenerate_bins

        Whether finalize substitutes ``std=1.0`` at zero-variance mel bins
        instead of raising; ``False`` is the strict production default.

    .. attribute :: git_sha

        Commit SHA of the launcher's working tree at construction.

    .. attribute :: is_repo_dirty

        Whether the launcher's working tree had uncommitted changes.

    .. attribute :: created_at

        UTC timestamp when the spec was first constructed.

    .. attribute :: run_id

        Deterministic W&B run ID derived from ``task_name`` and ``created_at``.

    .. attribute :: r2

        Nested R2 storage location (bucket + prefix_root + materialized prefix).
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", validate_default=True)

    # Splits stored as immutable tuples; JSON lists are coerced by _splits_list_to_tuple.
    task_name: str = Field(
        description=(
            "Dataset config identifier; becomes the prefix of ``run_id`` and the "
            "config-id path segment of ``r2.prefix`` (``<root>/<task_name>/<run_id>/``)."
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

    render: RenderConfig = Field(
        description="Nested ``RenderConfig`` carrying every per-shard renderer input."
    )

    mask_degenerate_bins: bool = Field(
        default=False,
        description=(
            "Whether the finalize stats fold substitutes ``std=1.0`` at zero-variance "
            "mel bins instead of raising; ``False`` is the strict production default. "
            "Smoke configs override to ``True`` because tiny renders have constant "
            "attack-time frames and channels below the source's active bandwidth."
        ),
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
    r2: R2Location = Field(
        # Returns a dict that Pydantic re-validates as R2Location via
        # ``validate_default=True``; pyright doesn't see the coercion.
        default_factory=_default_r2_location,  # type: ignore[arg-type]
        description=(
            "Nested R2 storage location (bucket + prefix_root + materialized prefix). "
            "Replaces the legacy flat ``r2_bucket`` / ``r2_prefix_root`` / ``r2_prefix`` "
            "fields; the model validator promotes legacy-form input dicts into this shape."
        ),
    )

    @classmethod
    def from_hydra_cfg(cls, cfg: DictConfig) -> DatasetSpec:
        """Build from a Hydra-composed cfg, dropping non-spec groups before resolving.

        A composed dataset cfg carries groups that aren't spec fields
        (``datamodule``, ``paths``, ``hydra``, ``logger``, …) whose interpolations
        may reference resolvers only available under ``@hydra.main`` (e.g.
        ``datamodule.dataset_root: ${hydra:runtime.output_dir}/data``). Masking to
        the model's own fields *before* ``resolve=True`` means those subtrees are
        never evaluated, so the spec resolves under a plain ``compose()`` too.

        :param cfg: Composed dataset cfg; only keys matching ``cls.model_fields``
            survive the mask, so non-spec groups need not resolve.
        :returns: Validated spec built from the masked, resolved mapping.
        :raises TypeError: ``cfg`` is not mapping-shaped (e.g. a ``ListConfig``,
            which ``masked_copy`` rejects with ``ValueError``) or the masked cfg
            did not resolve to a mapping — both normalized to one stable type.
        """
        spec_keys = [k for k in cfg if isinstance(k, str) and k in cls.model_fields]
        try:
            masked = OmegaConf.masked_copy(cfg, spec_keys)
        except ValueError as exc:
            raise TypeError(f"composed config is not a mapping: {type(cfg).__name__}") from exc
        raw = OmegaConf.to_container(masked, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"composed config is not a mapping: {type(raw).__name__}")
        return cls(**{k: v for k, v in raw.items() if isinstance(k, str)})

    @model_validator(mode="before")
    @classmethod
    def _normalize_r2_input(cls, data: Any) -> Any:
        """Promote legacy flat ``r2_bucket`` / ``r2_prefix_root`` / ``r2_prefix`` into ``r2``.

        Back-compat shim for materialized ``input_spec.json`` files already
        written to R2 before the nested ``R2Location`` migration. Mixed input
        (any legacy key AND an explicit ``r2``) is rejected — promotion would
        otherwise have to pick a precedence rule with no good answer. After
        promotion the missing ``prefix`` is filled from layout fields the same
        way the previous flat ``_default_r2_prefix`` factory did.

        :param data: Raw input to the validator (typically a dict; pass-through otherwise).
        :returns: Same input unchanged if no normalization is needed; otherwise a
            new dict with legacy keys promoted under ``r2`` and ``prefix`` filled.
        :raises ValueError: ``data`` contains both nested ``r2`` AND any legacy flat
            ``r2_*`` key — that combination is ambiguous and must be rewritten.
        """
        if not isinstance(data, dict):
            return data
        legacy_present = {k for k in _LEGACY_FLAT_R2_KEYS if k in data}
        if legacy_present and "r2" in data:
            raise ValueError(
                f"DatasetSpec received both nested 'r2' and legacy flat keys "
                f"{sorted(legacy_present)}; pass one shape, not both"
            )
        if not legacy_present and "r2" not in data:
            return data
        data = dict(data)
        if legacy_present:
            promoted: dict[str, Any] = {}
            for legacy_key, nested_key in _LEGACY_FLAT_R2_KEYS.items():
                if legacy_key in data:
                    promoted[nested_key] = data.pop(legacy_key)
            data["r2"] = promoted
        r2 = data["r2"]
        if isinstance(r2, dict) and "prefix" not in r2:
            data["r2"] = _fill_default_r2_prefix(data, r2)
        return data

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

    @model_validator(mode="before")
    @classmethod
    def _drop_null_run_id(cls, data: Any) -> Any:
        """Drop ``run_id`` when ``None`` so its ``default_factory`` fires.

        ``configs/dataset.yaml`` materializes ``run_id: null`` so the finalize
        workflow's ``run_id=<value>`` Hydra override resolves against an
        existing key. When that override is not pinned, the composed cfg
        arrives with ``run_id=None`` — letting it reach the field validator
        would fail strict ``str`` validation instead of falling back to the
        ``task_name``+``created_at`` default factory.

        :param data: Raw input to the validator (typically a dict; pass-through otherwise).
        :returns: ``data`` unchanged, or a copy with the ``None`` ``run_id`` popped.
        """
        if isinstance(data, dict) and data.get("run_id", "sentinel") is None:
            data = dict(data)
            data.pop("run_id")
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

    @field_validator("task_name")
    @classmethod
    def _task_name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank ``task_name`` so derived run_id / r2.prefix are never empty-prefixed."""
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
    def split_shard_ranges(self) -> dict[Split, tuple[int, int]]:
        """Half-open ``[lo, hi)`` shard-index ranges per split.

        :returns: Mapping from split name to ``(lo, hi)``; ``hi`` is exclusive
            and ``hi - lo`` equals ``size // render.samples_per_shard``. The
            ranges concatenate in train→val→test order so their union covers
            ``[0, num_shards)`` with no gaps.
        """
        sps = self.render.samples_per_shard
        train_n, val_n, test_n = (sz // sps for sz in self.train_val_test_sizes)
        return {
            "train": (0, train_n),
            "val": (train_n, train_n + val_n),
            "test": (train_n + val_n, train_n + val_n + test_n),
        }

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
