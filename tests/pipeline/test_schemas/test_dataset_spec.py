"""Behavioral tests for the unified DatasetSpec model."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from synth_setter.data.vst import param_specs
from synth_setter.pipeline.schemas.spec import (
    DatasetSpec,
    RenderConfig,
    ShardSpec,
)

FIXED_NOW = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)


def _valid_render_kwargs(plugin_path: str = "/fake/Plugin.vst3") -> dict[str, Any]:
    return {
        "plugin_path": plugin_path,
        "preset_path": "presets/surge-base.vstpreset",
        "param_spec_name": "surge_simple",
        "renderer_version": "1.3.4",
        "sample_rate": 16000,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "samples_per_render_batch": 32,
        "samples_per_shard": 100,
    }


def _valid_spec_kwargs(plugin_path: str = "/fake/Plugin.vst3", **overrides: Any) -> dict[str, Any]:
    """Return DatasetSpec kwargs that build a 3-shard hdf5 spec by default.

    Uses the nested ``r2`` shape (post-R2Location-migration). The back-compat
    shim that promotes legacy flat ``r2_bucket`` / ``r2_prefix_root`` /
    ``r2_prefix`` keys lives in its own test class (``TestLegacyFlatR2Compat``).
    """
    kwargs: dict[str, Any] = {
        "task_name": "ci-smoke-test",
        "output_format": "hdf5",
        "train_val_test_sizes": [300, 0, 0],
        "base_seed": 42,
        "r2": {"bucket": "intermediate-data"},
        "render": _valid_render_kwargs(plugin_path),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def patch_runtime_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub git/timestamp factories so DatasetSpec construction is deterministic."""
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "abc123def456")
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: FIXED_NOW)


# ---------------------------------------------------------------------------
# ShardSpec
# ---------------------------------------------------------------------------


class TestShardSpec:
    """Tests for ShardSpec frozen-model invariants."""

    def test_shard_spec_is_frozen(self) -> None:
        """Mutating a ShardSpec field after construction raises ValidationError."""
        shard = ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42)
        with pytest.raises(ValidationError):
            shard.shard_id = 99  # type: ignore[misc]

    def test_shard_spec_rejects_extra_fields(self) -> None:
        """ShardSpec rejects unknown keyword args under ``extra='forbid'``."""
        with pytest.raises(ValidationError):
            ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42, extra="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RenderConfig
# ---------------------------------------------------------------------------


class TestRenderConfig:
    """Tests for RenderConfig field validation."""

    def test_render_config_rejects_extra_fields(self) -> None:
        """RenderConfig rejects unknown keyword args under ``extra='forbid'``."""
        kwargs = _valid_render_kwargs()
        kwargs["surprise"] = "value"
        with pytest.raises(ValidationError):
            RenderConfig(**kwargs)

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("sample_rate", 0, "sample_rate must be positive"),
            ("channels", 0, "channels must be >= 1"),
            ("velocity", 200, r"velocity must be in \[0, 127\]"),
            ("signal_duration_seconds", 0.0, "signal_duration_seconds must be positive"),
            ("samples_per_render_batch", 0, "samples_per_render_batch must be positive"),
            ("samples_per_shard", 0, "samples_per_shard must be positive"),
            ("param_spec_name", "   ", "param_spec_name must not be blank"),
            ("renderer_version", "", "renderer_version must not be blank"),
        ],
    )
    def test_render_config_range_validators(self, field: str, bad_value: Any, match: str) -> None:
        """Out-of-range or blank values for required fields raise ValidationError."""
        kwargs = _valid_render_kwargs()
        kwargs[field] = bad_value
        with pytest.raises(ValidationError, match=match):
            RenderConfig(**kwargs)

    def test_render_config_velocity_bounds_are_inclusive(self) -> None:
        """Velocity 0 and 127 are both accepted (inclusive MIDI range)."""
        for valid in (0, 127):
            cfg = RenderConfig(**{**_valid_render_kwargs(), "velocity": valid})
            assert cfg.velocity == valid


# ---------------------------------------------------------------------------
# DatasetSpec — construction & runtime-field auto-fill
# ---------------------------------------------------------------------------


class TestDatasetSpecConstruction:
    """Tests for DatasetSpec construction and runtime-field auto-fill."""

    def test_fresh_construction_fills_runtime_fields(self, patch_runtime_io: None) -> None:
        """Default construction populates git_sha, created_at, run_id, r2.prefix from factories."""
        spec = DatasetSpec(**_valid_spec_kwargs())

        assert spec.git_sha == "abc123def456"
        assert spec.is_repo_dirty is False
        assert spec.created_at == FIXED_NOW
        assert spec.run_id == "ci-smoke-test-20260328T120000000Z"
        assert spec.r2.prefix == "data/ci-smoke-test/ci-smoke-test-20260328T120000000Z/"

    def test_run_id_uses_explicit_value_when_present(self, patch_runtime_io: None) -> None:
        """An explicit run_id in the input dict passes through instead of being regenerated."""
        spec = DatasetSpec(**_valid_spec_kwargs(run_id="custom-run-id-001"))
        assert spec.run_id == "custom-run-id-001"

    def test_r2_prefix_uses_explicit_value_when_present(self, patch_runtime_io: None) -> None:
        """An explicit r2.prefix in the input dict passes through instead of being recomputed."""
        spec = DatasetSpec(
            **_valid_spec_kwargs(
                r2={"bucket": "intermediate-data", "prefix": "custom/prefix/here/"}
            )
        )
        assert spec.r2.prefix == "custom/prefix/here/"

    def test_r2_prefix_root_default_is_data(self, patch_runtime_io: None) -> None:
        """Default ``r2.prefix_root`` produces a prefix beginning with ``data/``."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.r2.prefix.startswith("data/")

    def test_r2_prefix_root_custom_threads_through(self, patch_runtime_io: None) -> None:
        """A custom ``r2.prefix_root`` threads through into the materialized prefix."""
        spec = DatasetSpec(
            **_valid_spec_kwargs(r2={"bucket": "intermediate-data", "prefix_root": "experiments"})
        )
        assert spec.r2.prefix.startswith("experiments/")

    def test_dataset_spec_strict_rejects_extra_fields(self, patch_runtime_io: None) -> None:
        """Unknown top-level keys are rejected at the trust boundary."""
        kwargs = _valid_spec_kwargs(unexpected_field="surprise")
        with pytest.raises(ValidationError):
            DatasetSpec(**kwargs)

    def test_model_validate_does_not_mutate_input_dict(self, patch_runtime_io: None) -> None:
        """Passing a dict to model_validate must not strip computed keys from the caller's dict."""
        first = DatasetSpec(**_valid_spec_kwargs())
        round_trip = first.model_dump(mode="json")
        before_keys = set(round_trip)
        DatasetSpec.model_validate(round_trip)
        assert set(round_trip) == before_keys
        assert {"shards", "num_shards", "num_params"}.issubset(round_trip)


class TestDatasetSpecValidators:
    """Tests for DatasetSpec field-level and cross-field validators."""

    def test_r2_bucket_blank_raises(self, patch_runtime_io: None) -> None:
        """Blank or whitespace-only ``r2.bucket`` raises (rclone would receive a malformed URI)."""
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="r2_bucket must not be blank"):
                DatasetSpec(**_valid_spec_kwargs(r2={"bucket": blank}))

    def test_task_name_blank_raises(self, patch_runtime_io: None) -> None:
        """Blank task_name raises (derived run_id / r2.prefix would be empty-prefixed)."""
        with pytest.raises(ValidationError, match="task_name must not be blank"):
            DatasetSpec(**_valid_spec_kwargs(task_name="   "))

    def test_r2_prefix_root_blank_raises(self, patch_runtime_io: None) -> None:
        """Blank ``r2.prefix_root`` raises.

        Prevents derived ``r2.prefix`` from starting with a stray ``/``.
        """
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="r2_prefix_root must not be blank"):
                DatasetSpec(
                    **_valid_spec_kwargs(r2={"bucket": "intermediate-data", "prefix_root": blank})
                )

    def test_explicit_r2_prefix_missing_trailing_slash_raises(
        self, patch_runtime_io: None
    ) -> None:
        """An explicit ``r2.prefix`` missing the trailing ``/`` raises (concat trap)."""
        with pytest.raises(ValidationError, match="r2_prefix must end with"):
            DatasetSpec(
                **_valid_spec_kwargs(r2={"bucket": "intermediate-data", "prefix": "data/no/slash"})
            )

    def test_split_size_not_multiple_of_samples_per_shard_raises(
        self, patch_runtime_io: None
    ) -> None:
        """A split size that doesn't divide evenly into ``samples_per_shard`` raises."""
        with pytest.raises(ValidationError, match="not a multiple"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[150, 0, 0]))

    def test_negative_split_size_raises(self, patch_runtime_io: None) -> None:
        """Negative split sizes raise rather than silently truncating shards."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[-100, 0, 0]))

    def test_zero_total_split_raises(self, patch_runtime_io: None) -> None:
        """Three-zero splits raise (a dataset with zero samples is a config error)."""
        with pytest.raises(ValidationError, match="must sum to a positive count"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[0, 0, 0]))

    def test_invalid_output_format_literal_raises(self, patch_runtime_io: None) -> None:
        """An output_format outside the supported Literal set is rejected.

        ``parquet`` stays outside the Literal even as new formats (wds) join — pinning
        the rejection here prevents a typo (``parquet`` vs. ``wds``) from sneaking in.
        """
        with pytest.raises(ValidationError):
            DatasetSpec(**_valid_spec_kwargs(output_format="parquet"))

    def test_wds_output_format_constructs(self, patch_runtime_io: None) -> None:
        """``output_format='wds'`` is accepted (unblocks the wds writer landing in PR-13)."""
        spec = DatasetSpec(**_valid_spec_kwargs(output_format="wds"))
        assert spec.output_format == "wds"

    def test_strict_mode_rejects_int_for_string_field(self, patch_runtime_io: None) -> None:
        """Strict mode rejects silent int→str coercion at the trust boundary."""
        with pytest.raises(ValidationError):
            DatasetSpec(**_valid_spec_kwargs(task_name=12345))

    def test_strict_mode_rejects_string_for_bool_field(self, patch_runtime_io: None) -> None:
        """Strict mode rejects silent str→bool coercion (e.g., ``is_repo_dirty="false"``)."""
        with pytest.raises(ValidationError):
            DatasetSpec(**_valid_spec_kwargs(is_repo_dirty="false"))

    @pytest.mark.parametrize("bad_length", [[300, 0], [300, 0, 0, 0]])
    def test_train_val_test_sizes_must_be_length_three(
        self, patch_runtime_io: None, bad_length: list[int]
    ) -> None:
        """train_val_test_sizes must be exactly length 3 — not 2, not 4."""
        with pytest.raises(ValidationError):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=bad_length))

    @pytest.mark.parametrize(
        "bad_value",
        [[42, 43, 44], (42, 43, 44), [1, 2, 3, 4], "anything", 0],
    )
    def test_train_val_test_seeds_setting_raises_not_implemented(
        self, patch_runtime_io: None, bad_value: Any
    ) -> None:
        """Setting train_val_test_seeds raises NotImplementedError — reserved for #884."""
        with pytest.raises(NotImplementedError, match="reserved for per-sample seeding"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_seeds=bad_value))

    def test_train_val_test_seeds_defaults_to_none(self, patch_runtime_io: None) -> None:
        """Omitting train_val_test_seeds yields the default None — field is optional."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.train_val_test_seeds is None

    def test_train_val_test_seeds_explicit_none_is_allowed(self, patch_runtime_io: None) -> None:
        """Explicit None passes (NotImplementedError gate fires only on non-None)."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_seeds=None))
        assert spec.train_val_test_seeds is None

    def test_explicit_empty_r2_prefix_raises(self, patch_runtime_io: None) -> None:
        """Empty ``r2.prefix`` raises via the ``_prefix_must_end_with_slash`` validator.

        The default_factory is bypassed when a value is supplied.
        """
        with pytest.raises(ValidationError, match="r2_prefix must end with"):
            DatasetSpec(**_valid_spec_kwargs(r2={"bucket": "intermediate-data", "prefix": ""}))

    def test_z_suffixed_created_at_string_parses(self, patch_runtime_io: None) -> None:
        """``model_dump_json``'s ``Z``-suffixed UTC timestamps round-trip on Python 3.10.

        ``datetime.fromisoformat`` rejects the ``Z`` offset on 3.10 (accepts on 3.11+); the
        ``_parse_iso_datetime`` validator normalizes it before strict-mode validation runs.
        """
        payload = {**_valid_spec_kwargs(), "created_at": "2026-03-28T12:00:00Z"}
        spec = DatasetSpec(**payload)
        assert spec.created_at == FIXED_NOW

    def test_malformed_created_at_string_raises(self, patch_runtime_io: None) -> None:
        """A malformed datetime string surfaces a clear validation error, not a silent coerce."""
        payload = {**_valid_spec_kwargs(), "created_at": "not-a-datetime"}
        with pytest.raises(ValidationError):
            DatasetSpec(**payload)

    def test_naive_created_at_string_raises(self, patch_runtime_io: None) -> None:
        """Naive ISO datetimes are rejected at the ``created_at`` boundary.

        ``make_dataset_wandb_run_id`` downstream needs tz-aware UTC; without this check
        an invalid ``created_at`` would surface as a run_id derivation crash with the
        wrong error attribution.
        """
        payload = {**_valid_spec_kwargs(), "created_at": "2026-03-28T12:00:00"}
        with pytest.raises(ValidationError, match="created_at must be timezone-aware UTC"):
            DatasetSpec(**payload)

    def test_non_utc_created_at_string_raises(self, patch_runtime_io: None) -> None:
        """Non-UTC ISO offsets are rejected so run_id derivation produces correct timestamps."""
        payload = {**_valid_spec_kwargs(), "created_at": "2026-03-28T12:00:00+05:00"}
        with pytest.raises(ValidationError, match="created_at must be UTC"):
            DatasetSpec(**payload)

    def test_train_val_test_sizes_stored_as_immutable_tuple(self, patch_runtime_io: None) -> None:
        """Splits stored as ``tuple`` so in-place mutation can't invalidate cached ``shards``.

        ``frozen=True`` only blocks attribute reassignment, not in-place mutation of
        contained-types. Tuple makes in-place mutation a ``TypeError``, preserving the
        frozen contract end-to-end.
        """
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert isinstance(spec.train_val_test_sizes, tuple)
        with pytest.raises(TypeError):
            spec.train_val_test_sizes[0] = 999  # type: ignore[index]

    def test_train_val_test_sizes_accepts_json_list(self, patch_runtime_io: None) -> None:
        """JSON-loaded specs deliver lists (no native tuple type); coerce them to tuples."""
        payload = {**_valid_spec_kwargs(), "train_val_test_sizes": [300, 0, 0]}
        spec = DatasetSpec(**payload)
        assert spec.train_val_test_sizes == (300, 0, 0)
        assert isinstance(spec.train_val_test_sizes, tuple)


# ---------------------------------------------------------------------------
# DatasetSpec — computed fields
# ---------------------------------------------------------------------------


class TestDatasetSpecComputedFields:
    """Tests for the ``shards`` / ``num_shards`` / ``num_params`` computed fields."""

    def test_shards_count_matches_total_size_div_batch(self, patch_runtime_io: None) -> None:
        """num_shards = sum(train_val_test_sizes) / samples_per_shard."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[400, 100, 100]))
        assert spec.num_shards == 6
        assert len(spec.shards) == 6

    def test_shard_seeds_are_base_plus_shard_id(self, patch_runtime_io: None) -> None:
        """Per-shard seed equals ``base_seed + shard_id``."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[300, 0, 0]))
        assert [s.seed for s in spec.shards] == [42, 43, 44]

    def test_shard_filenames_zero_padded_six_digits(self, patch_runtime_io: None) -> None:
        """Shard filenames use the ``shard-NNNNNN`` six-digit zero-padded form."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[300, 0, 0]))
        assert spec.shards[0].filename == "shard-000000.h5"
        assert spec.shards[-1].filename == "shard-000002.h5"

    def test_shard_filename_extension_matches_output_format(self, patch_runtime_io: None) -> None:
        """Shard filenames carry the extension implied by ``output_format``."""
        hdf5_spec = DatasetSpec(**_valid_spec_kwargs(output_format="hdf5"))
        assert all(s.filename.endswith(".h5") for s in hdf5_spec.shards)
        wds_spec = DatasetSpec(**_valid_spec_kwargs(output_format="wds"))
        assert all(s.filename.endswith(".tar") for s in wds_spec.shards)

    def test_num_params_resolved_from_registry(self, patch_runtime_io: None) -> None:
        """``num_params`` matches the registry's length for the spec's ``param_spec_name``."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.num_params == len(param_specs["surge_simple"])

    def test_unknown_param_spec_name_raises_at_compute(self, patch_runtime_io: None) -> None:
        """An unknown ``param_spec_name`` raises only when ``num_params`` is materialized."""
        kwargs = _valid_spec_kwargs()
        kwargs["render"] = {**kwargs["render"], "param_spec_name": "nonexistent_synth"}
        spec = DatasetSpec(**kwargs)
        with pytest.raises(KeyError):
            _ = spec.num_params


# ---------------------------------------------------------------------------
# DatasetSpec — JSON round-trip
# ---------------------------------------------------------------------------


class TestDatasetSpecRoundTrip:
    """Tests for ``model_dump_json`` → ``model_validate_json`` round-trip preservation."""

    def test_json_round_trip_preserves_runtime_fields(self, patch_runtime_io: None) -> None:
        """git_sha, is_repo_dirty, created_at, run_id, r2 all survive a JSON round-trip."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        json_str = spec.model_dump_json()
        restored = DatasetSpec.model_validate_json(json_str)

        assert restored.git_sha == spec.git_sha
        assert restored.is_repo_dirty == spec.is_repo_dirty
        assert restored.created_at == spec.created_at
        assert restored.run_id == spec.run_id
        assert restored.r2 == spec.r2

    def test_json_round_trip_preserves_shards(self, patch_runtime_io: None) -> None:
        """Computed ``shards`` / ``num_shards`` round-trip identically through JSON."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[400, 100, 100]))
        restored = DatasetSpec.model_validate_json(spec.model_dump_json())
        assert restored.shards == spec.shards
        assert restored.num_shards == spec.num_shards

    def test_json_round_trip_works_without_plugin_on_disk(self, patch_runtime_io: None) -> None:
        """JSON round-trip does not require the plugin path to exist on disk (worker side)."""
        spec = DatasetSpec(**_valid_spec_kwargs(plugin_path="/nonexistent/path.vst3"))
        restored = DatasetSpec.model_validate_json(spec.model_dump_json())
        assert restored.render.plugin_path == "/nonexistent/path.vst3"

    def test_json_round_trip_rebuilds_with_no_runtime_drift(self, patch_runtime_io: None) -> None:
        """Worker reconstructing a spec from R2 sees the same git_sha/run_id the launcher used."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        # Simulate worker on a different commit; default_factory must not run for JSON-loaded.
        json_str = spec.model_dump_json()

        def _drift_sha() -> str:
            return "f" * 40

        # Patch the factory; if pass-through works the factory shouldn't be called.
        import synth_setter.pipeline.schemas.spec as spec_mod

        original = spec_mod._get_git_sha
        spec_mod._get_git_sha = _drift_sha
        try:
            restored = DatasetSpec.model_validate_json(json_str)
        finally:
            spec_mod._get_git_sha = original

        assert restored.git_sha == "abc123def456"


# ---------------------------------------------------------------------------
# Graceful git helpers — _get_git_sha / _is_repo_dirty under missing .git/
# ---------------------------------------------------------------------------


class TestGitHelpersGraceful:
    """``_get_git_sha`` and ``_is_repo_dirty`` must not crash when ``.git/`` is unavailable."""

    def test_get_git_sha_returns_sentinel_on_called_process_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-zero ``git rev-parse`` exit returns the sentinel rather than raising."""
        import synth_setter.pipeline.schemas.spec as spec_mod

        def _raise(*args: object, **kwargs: object) -> object:
            raise subprocess.CalledProcessError(returncode=128, cmd=["git", "rev-parse", "HEAD"])

        monkeypatch.setattr(spec_mod.subprocess, "run", _raise)
        assert spec_mod._get_git_sha() == spec_mod._GIT_UNAVAILABLE_SENTINEL

    def test_get_git_sha_returns_sentinel_when_git_binary_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ``git`` binary yields the sentinel — worker hosts without git survive."""
        import synth_setter.pipeline.schemas.spec as spec_mod

        def _raise(*args: object, **kwargs: object) -> object:
            raise FileNotFoundError("git")

        monkeypatch.setattr(spec_mod.subprocess, "run", _raise)
        assert spec_mod._get_git_sha() == spec_mod._GIT_UNAVAILABLE_SENTINEL

    def test_is_repo_dirty_returns_false_when_git_binary_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ``git`` binary makes ``_is_repo_dirty`` return False rather than raise."""
        import synth_setter.pipeline.schemas.spec as spec_mod

        def _raise(*args: object, **kwargs: object) -> object:
            raise FileNotFoundError("git")

        monkeypatch.setattr(spec_mod.subprocess, "run", _raise)
        assert spec_mod._is_repo_dirty() is False

    def test_is_repo_dirty_returns_false_when_outside_git_worktree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``git diff --quiet`` exits 128 outside a worktree — treat as "no git", not "dirty"."""
        import synth_setter.pipeline.schemas.spec as spec_mod

        class _FakeCompleted:
            returncode = 128

        def _fake(*args: object, **kwargs: object) -> _FakeCompleted:
            return _FakeCompleted()

        monkeypatch.setattr(spec_mod.subprocess, "run", _fake)
        assert spec_mod._is_repo_dirty() is False


# ---------------------------------------------------------------------------
# Bare import is launcher-pure — no pedalboard / synth_setter.data.vst.core load
# ---------------------------------------------------------------------------


class TestSpecImportStaysLauncherPure:
    """``import synth_setter.pipeline.schemas.spec`` alone must not pull heavy modules."""

    def test_bare_spec_import_does_not_pull_data_vst_core(self) -> None:
        """Bare spec import must not transitively load ``data.vst.core`` or ``pedalboard``.

        ``spec.py``'s only ``synth_setter.data.vst`` import is inside
        ``DatasetSpec.num_params`` and runs lazily. If it is re-promoted to
        module level — or another heavy import is added at module load — this
        test fails immediately, preserving the launcher's interpreter-only
        contract.
        """
        script = (
            "import sys\n"
            "import synth_setter.pipeline.schemas.spec  # noqa: F401\n"
            "for name in ('synth_setter.data.vst.core', 'synth_setter.data.vst', 'pedalboard'):\n"
            "    assert name not in sys.modules, (\n"
            "        f'{name!r} leaked into spec module import; '\n"
            "        f'this breaks the launcher-pure invariant'\n"
            "    )\n"
        )
        result = subprocess.run(  # noqa: S603 — sys.executable + literal script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"bare spec import is no longer launcher-pure:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )


# ---------------------------------------------------------------------------
# Stronger pedalboard-free invariant — construction + serialization must not
# pull pedalboard either. Catches regressions where the lazy import path
# inside ``num_params`` (or its callers via ``model_dump_json``) silently
# starts loading native deps.
# ---------------------------------------------------------------------------


class TestSpecConstructionStaysPedalboardFree:
    """Importing schemas + building/serializing a DatasetSpec must not load pedalboard.

    Run in a fresh subprocess so the parent test session — where earlier tests
    import ``synth_setter.data.vst.core`` (and other modules that pull pedalboard
    transitively) — does not poison the check.
    """

    def test_dataset_spec_num_params_does_not_import_pedalboard(self) -> None:
        """``num_params`` must remain interpreter-only across every serialize.

        ``num_params`` is emitted by ``model_dump_json`` so its lazy import path
        runs in every serialize call.
        """
        script = (
            "import sys\n"
            "from synth_setter.pipeline.schemas.spec import DatasetSpec\n"
            "spec = DatasetSpec(\n"
            "    task_name='ci', output_format='hdf5', train_val_test_sizes=[1, 0, 0],\n"
            "    base_seed=0, r2={'bucket': 'b'},\n"
            "    render={\n"
            "        'plugin_path': '/tmp/x.vst3', 'preset_path': '/tmp/x.vstpreset',\n"
            "        'param_spec_name': 'surge_simple', 'renderer_version': 'v1',\n"
            "        'sample_rate': 16000, 'channels': 1, 'velocity': 64,\n"
            "        'signal_duration_seconds': 1.0, 'min_loudness': -30.0,\n"
            "        'samples_per_render_batch': 1, 'samples_per_shard': 1,\n"
            "    },\n"
            ")\n"
            "_ = spec.num_params\n"
            "_ = spec.model_dump_json()\n"
            "assert 'pedalboard' not in sys.modules, sorted(\n"
            "    m for m in sys.modules if m.startswith('pedalboard')\n"
            ")\n"
        )
        result = subprocess.run(  # noqa: S603 — sys.executable + literal script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"pedalboard leaked into spec serialization:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )


# ---------------------------------------------------------------------------
# DatasetSpec — back-compat shim for legacy flat r2_bucket / r2_prefix_root /
# r2_prefix keys (input_spec.json files materialized before the R2Location
# migration must still parse and re-emit in the new shape).
# ---------------------------------------------------------------------------


class TestLegacyFlatR2Compat:
    """Legacy flat ``r2_*`` keys on input must promote into the nested ``r2`` model."""

    def _legacy_kwargs(self, **overrides: Any) -> dict[str, Any]:  # noqa: DOC101,DOC103,DOC201,DOC203
        """Return spec kwargs that mirror a pre-migration input dict (flat r2_* keys)."""
        kwargs: dict[str, Any] = {
            "task_name": "ci-smoke-test",
            "output_format": "hdf5",
            "train_val_test_sizes": [300, 0, 0],
            "base_seed": 42,
            "r2_bucket": "intermediate-data",
            "render": _valid_render_kwargs(),
        }
        kwargs.update(overrides)
        return kwargs

    def test_legacy_r2_bucket_only_promotes_and_derives_prefix(  # noqa: DOC101,DOC103
        self, patch_runtime_io: None
    ) -> None:
        """Pre-migration spec with only ``r2_bucket`` → derived prefix lands on ``r2.prefix``."""
        spec = DatasetSpec(**self._legacy_kwargs())
        assert spec.r2.bucket == "intermediate-data"
        assert spec.r2.prefix_root == "data"
        assert spec.r2.prefix == "data/ci-smoke-test/ci-smoke-test-20260328T120000000Z/"

    def test_all_three_legacy_keys_promote_into_nested_r2(self, patch_runtime_io: None) -> None:  # noqa: DOC101,DOC103
        """Full legacy triple promotes into ``r2`` and preserves explicit ``r2_prefix``."""
        spec = DatasetSpec(
            **self._legacy_kwargs(
                r2_prefix_root="experiments",
                r2_prefix="experiments/legacy/custom/",
            )
        )
        assert spec.r2.bucket == "intermediate-data"
        assert spec.r2.prefix_root == "experiments"
        assert spec.r2.prefix == "experiments/legacy/custom/"

    def test_mixed_nested_and_legacy_input_raises(self, patch_runtime_io: None) -> None:  # noqa: DOC101,DOC103
        """Mixing ``r2`` and any legacy key is ambiguous — must fail-fast."""
        with pytest.raises(ValidationError, match="both nested 'r2' and legacy flat keys"):
            DatasetSpec(
                **self._legacy_kwargs(r2={"bucket": "intermediate-data"}),
            )

    def test_legacy_json_specs_round_trip_in_new_form(self, patch_runtime_io: None) -> None:  # noqa: DOC101,DOC103
        """Old input_spec.json files in R2 parse + re-emit in the new nested shape.

        Worker reconstruction contract: an old JSON spec (flat keys) parses
        identically to a freshly-materialized spec; re-serializing produces
        the new nested form so the next round-trip is on the new shape.
        """
        import json as _json

        spec = DatasetSpec(**self._legacy_kwargs())
        # Emulate an old input_spec.json: nested computed-field keys stripped + flat r2_* present.
        round_trip = spec.model_dump(mode="json")
        round_trip.pop("r2")
        round_trip["r2_bucket"] = "intermediate-data"
        round_trip["r2_prefix_root"] = "data"
        round_trip["r2_prefix"] = "data/ci-smoke-test/ci-smoke-test-20260328T120000000Z/"
        restored = DatasetSpec.model_validate_json(_json.dumps(round_trip))

        assert restored.r2.bucket == "intermediate-data"
        assert restored.r2.prefix == "data/ci-smoke-test/ci-smoke-test-20260328T120000000Z/"
        assert "r2" in restored.model_dump_json()
        # Re-emitted JSON does not contain the legacy flat keys.
        emitted = _json.loads(restored.model_dump_json())
        for legacy_key in ("r2_bucket", "r2_prefix_root", "r2_prefix"):
            assert legacy_key not in emitted
