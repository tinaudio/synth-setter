"""Behavioral tests for the unified DatasetSpec model."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pipeline.schemas.spec import (
    DatasetSpec,
    RenderConfig,
    ShardSpec,
    dataset_config_id_from_path,
    load_dataset_spec_yaml,
)
from src.data.vst import param_specs

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
        "sample_batch_size": 32,
        "batch_per_shard": 100,
    }


def _valid_spec_kwargs(plugin_path: str = "/fake/Plugin.vst3", **overrides: Any) -> dict[str, Any]:
    """Return DatasetSpec kwargs that build a 3-shard hdf5 spec by default."""
    kwargs: dict[str, Any] = {
        "task_name": "ci-smoke-test",
        "output_format": "hdf5",
        "train_val_test_sizes": (300, 0, 0),
        "train_val_test_seeds": (123, 456, 789),
        "base_seed": 42,
        "r2_bucket": "intermediate-data",
        "render": _valid_render_kwargs(plugin_path),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def patch_runtime_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub git/timestamp factories so DatasetSpec construction is deterministic."""
    monkeypatch.setattr("pipeline.schemas.spec._get_git_sha", lambda: "abc123def456")
    monkeypatch.setattr("pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("pipeline.schemas.spec._utc_now", lambda: FIXED_NOW)


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
            ("sample_batch_size", 0, "sample_batch_size must be positive"),
            ("batch_per_shard", 0, "batch_per_shard must be positive"),
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

    def test_render_config_is_frozen(self) -> None:
        """Mutating a RenderConfig field after construction raises ValidationError.

        Pins ``frozen=True``: ``DatasetSpec.shards`` / ``num_params`` /
        ``num_shards`` are ``@cached_property`` over ``render.batch_per_shard``
        and ``render.param_spec_name``, so post-construction mutation could
        leave cached values stale.
        """
        cfg = RenderConfig(**_valid_render_kwargs())
        with pytest.raises(ValidationError):
            cfg.sample_rate = 48000  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DatasetSpec — construction & runtime-field auto-fill
# ---------------------------------------------------------------------------


class TestDatasetSpecConstruction:
    """Tests for DatasetSpec construction and runtime-field auto-fill."""

    def test_fresh_construction_fills_runtime_fields(self, patch_runtime_io: None) -> None:
        """Default construction populates git_sha, created_at, run_id, r2_prefix from factories."""
        spec = DatasetSpec(**_valid_spec_kwargs())

        assert spec.git_sha == "abc123def456"
        assert spec.is_repo_dirty is False
        assert spec.created_at == FIXED_NOW
        assert spec.run_id == "ci-smoke-test-20260328T120000000Z"
        assert spec.r2_prefix == "data/ci-smoke-test/ci-smoke-test-20260328T120000000Z/"

    def test_run_id_uses_explicit_value_when_present(self, patch_runtime_io: None) -> None:
        """An explicit run_id in the input dict passes through instead of being regenerated."""
        spec = DatasetSpec(**_valid_spec_kwargs(run_id="custom-run-id-001"))
        assert spec.run_id == "custom-run-id-001"

    def test_r2_prefix_uses_explicit_value_when_present(self, patch_runtime_io: None) -> None:
        """An explicit r2_prefix in the input dict passes through instead of being recomputed."""
        spec = DatasetSpec(**_valid_spec_kwargs(r2_prefix="custom/prefix/here/"))
        assert spec.r2_prefix == "custom/prefix/here/"

    def test_r2_prefix_root_default_is_data(self, patch_runtime_io: None) -> None:
        """Default ``r2_prefix_root`` produces a prefix beginning with ``data/``."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.r2_prefix.startswith("data/")

    def test_r2_prefix_root_custom_threads_through(self, patch_runtime_io: None) -> None:
        """A custom ``r2_prefix_root`` threads through into the materialized prefix."""
        spec = DatasetSpec(**_valid_spec_kwargs(r2_prefix_root="experiments"))
        assert spec.r2_prefix.startswith("experiments/")

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

    def test_dataset_spec_is_frozen(self, patch_runtime_io: None) -> None:
        """Mutating a DatasetSpec field after construction raises ValidationError.

        Pins ``frozen=True``: ``shards`` / ``num_shards`` / ``num_params``
        are ``@cached_property`` over input fields, so allowing reassignment
        of (e.g.) ``train_val_test_sizes`` after first access would leave the
        cached values stale.
        """
        spec = DatasetSpec(**_valid_spec_kwargs())
        with pytest.raises(ValidationError):
            spec.task_name = "renamed-after-construction"  # type: ignore[misc]


class TestDatasetSpecValidators:
    """Tests for DatasetSpec field-level and cross-field validators."""

    def test_r2_bucket_blank_raises(self, patch_runtime_io: None) -> None:
        """Blank or whitespace-only r2_bucket raises (rclone would receive a malformed URI)."""
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="r2_bucket must not be blank"):
                DatasetSpec(**_valid_spec_kwargs(r2_bucket=blank))

    def test_task_name_blank_raises(self, patch_runtime_io: None) -> None:
        """Blank task_name raises (derived run_id / r2_prefix would be empty-prefixed)."""
        with pytest.raises(ValidationError, match="task_name must not be blank"):
            DatasetSpec(**_valid_spec_kwargs(task_name="   "))

    def test_explicit_r2_prefix_missing_trailing_slash_raises(
        self, patch_runtime_io: None
    ) -> None:
        """An explicit r2_prefix lacking the trailing ``/`` raises (rclone path concat trap)."""
        with pytest.raises(ValidationError, match="r2_prefix must end with"):
            DatasetSpec(**_valid_spec_kwargs(r2_prefix="data/no/slash"))

    def test_split_size_not_multiple_of_batch_per_shard_raises(
        self, patch_runtime_io: None
    ) -> None:
        """A split size that doesn't divide evenly into ``batch_per_shard`` raises."""
        with pytest.raises(ValidationError, match="not a multiple"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(150, 0, 0)))

    def test_negative_split_size_raises(self, patch_runtime_io: None) -> None:
        """Negative split sizes raise rather than silently truncating shards."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(-100, 0, 0)))

    def test_zero_total_split_raises(self, patch_runtime_io: None) -> None:
        """Three-zero splits raise (a dataset with zero samples is a config error)."""
        with pytest.raises(ValidationError, match="must sum to a positive count"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(0, 0, 0)))

    def test_invalid_output_format_literal_raises(self, patch_runtime_io: None) -> None:
        """An output_format outside ``Literal['hdf5']`` is rejected by Pydantic literal check."""
        with pytest.raises(ValidationError):
            DatasetSpec(**_valid_spec_kwargs(output_format="parquet"))


# ---------------------------------------------------------------------------
# DatasetSpec — computed fields
# ---------------------------------------------------------------------------


class TestDatasetSpecComputedFields:
    """Tests for the ``shards`` / ``num_shards`` / ``num_params`` computed fields."""

    def test_shards_count_matches_total_size_div_batch(self, patch_runtime_io: None) -> None:
        """num_shards = sum(train_val_test_sizes) / batch_per_shard."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(400, 100, 100)))
        assert spec.num_shards == 6
        assert len(spec.shards) == 6

    def test_shard_seeds_are_base_plus_shard_id(self, patch_runtime_io: None) -> None:
        """Per-shard seed equals ``base_seed + shard_id``."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(300, 0, 0)))
        assert [s.seed for s in spec.shards] == [42, 43, 44]

    def test_shard_filenames_zero_padded_six_digits(self, patch_runtime_io: None) -> None:
        """Shard filenames use the ``shard-NNNNNN`` six-digit zero-padded form."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(300, 0, 0)))
        assert spec.shards[0].filename == "shard-000000.h5"
        assert spec.shards[-1].filename == "shard-000002.h5"

    def test_shard_filename_extension_matches_output_format(self, patch_runtime_io: None) -> None:
        """Shard filenames carry the extension implied by ``output_format``."""
        spec = DatasetSpec(**_valid_spec_kwargs(output_format="hdf5"))
        assert all(s.filename.endswith(".h5") for s in spec.shards)

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
        """git_sha, is_repo_dirty, created_at, run_id, r2_prefix all survive a JSON round-trip."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        json_str = spec.model_dump_json()
        restored = DatasetSpec.model_validate_json(json_str)

        assert restored.git_sha == spec.git_sha
        assert restored.is_repo_dirty == spec.is_repo_dirty
        assert restored.created_at == spec.created_at
        assert restored.run_id == spec.run_id
        assert restored.r2_prefix == spec.r2_prefix

    def test_json_round_trip_preserves_shards(self, patch_runtime_io: None) -> None:
        """Computed ``shards`` / ``num_shards`` round-trip identically through JSON."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(400, 100, 100)))
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
        import pipeline.schemas.spec as spec_mod

        original = spec_mod._get_git_sha
        spec_mod._get_git_sha = _drift_sha
        try:
            restored = DatasetSpec.model_validate_json(json_str)
        finally:
            spec_mod._get_git_sha = original

        assert restored.git_sha == "abc123def456"


# ---------------------------------------------------------------------------
# dataset_config_id_from_path
# ---------------------------------------------------------------------------


class TestDatasetConfigIdFromPath:
    """Tests for ``dataset_config_id_from_path`` filename-stem extraction."""

    def test_extracts_stem(self) -> None:
        """Returns the filename stem (no extension, no parent directories)."""
        assert (
            dataset_config_id_from_path(Path("configs/dataset/surge-simple-480k-10k.yaml"))
            == "surge-simple-480k-10k"
        )


# ---------------------------------------------------------------------------
# load_dataset_spec_yaml — legacy bridge (removed in A.3)
# ---------------------------------------------------------------------------

LEGACY_YAML_KWARGS: dict[str, Any] = {
    "param_spec": "surge_simple",
    "plugin_path": "plugins/Surge XT.vst3",
    "output_format": "hdf5",
    "sample_rate": 16000,
    "shard_size": 100,
    "num_shards": 3,
    "base_seed": 42,
    "r2_bucket": "intermediate-data",
    "splits": {"train": 3, "val": 0, "test": 0},
    "preset_path": "presets/surge-base.vstpreset",
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "sample_batch_size": 32,
}


@pytest.fixture()
def legacy_yaml(tmp_path: Path) -> Path:
    """Write a legacy-shape YAML config to tmp_path and return its path."""
    import yaml

    path = tmp_path / "ci-smoke-test.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(LEGACY_YAML_KWARGS), sort_keys=False))
    return path


class TestLoadDatasetSpecYaml:
    """Tests for the legacy YAML → DatasetSpec bridge (removed in PR-3)."""

    def test_legacy_yaml_round_trips_into_dataset_spec(
        self, patch_runtime_io: None, legacy_yaml: Path
    ) -> None:
        """A legacy flat YAML loads into a valid DatasetSpec with expected derived fields."""
        spec = load_dataset_spec_yaml(legacy_yaml)
        assert spec.task_name == "ci-smoke-test"
        assert spec.output_format == "hdf5"
        assert spec.train_val_test_sizes == (300, 0, 0)
        assert spec.render.plugin_path == "plugins/Surge XT.vst3"
        assert spec.render.batch_per_shard == 100
        assert spec.num_shards == 3

    def test_legacy_yaml_extends_merges_base(self, patch_runtime_io: None, tmp_path: Path) -> None:
        """A child YAML's ``_extends`` merges its base, with child values overriding."""
        import yaml

        base = tmp_path / "base.yaml"
        base.write_text(yaml.safe_dump(copy.deepcopy(LEGACY_YAML_KWARGS), sort_keys=False))
        child = tmp_path / "child.yaml"
        child.write_text("_extends: base\nvelocity: 110\n")

        spec = load_dataset_spec_yaml(child)
        assert spec.output_format == "hdf5"
        assert spec.render.plugin_path == "plugins/Surge XT.vst3"
        assert spec.render.velocity == 110

    def test_legacy_yaml_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """A nonexistent YAML path raises FileNotFoundError with a clear message."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_dataset_spec_yaml(tmp_path / "nonexistent.yaml")

    def test_legacy_yaml_extends_missing_base_raises(self, tmp_path: Path) -> None:
        """A YAML extending a missing base raises FileNotFoundError naming the missing target."""
        child = tmp_path / "child.yaml"
        child.write_text("_extends: nonexistent\nvelocity: 110\n")
        with pytest.raises(FileNotFoundError, match="_extends target not found"):
            load_dataset_spec_yaml(child)

    def test_legacy_yaml_num_shards_mismatch_raises(
        self, patch_runtime_io: None, tmp_path: Path
    ) -> None:
        """Legacy YAML where num_shards disagrees with sum(splits) raises a clear error."""
        import yaml as _yaml

        bad = copy.deepcopy(LEGACY_YAML_KWARGS)
        bad["num_shards"] = 99  # splits sum to 3 — they disagree
        path = tmp_path / "mismatch.yaml"
        path.write_text(_yaml.safe_dump(bad, sort_keys=False))
        with pytest.raises(ValueError, match="num_shards=99 disagrees with sum"):
            load_dataset_spec_yaml(path)

    def test_legacy_yaml_without_num_shards_loads(
        self, patch_runtime_io: None, tmp_path: Path
    ) -> None:
        """Legacy YAML missing num_shards still loads (the check only fires when present)."""
        import yaml as _yaml

        raw = copy.deepcopy(LEGACY_YAML_KWARGS)
        del raw["num_shards"]
        path = tmp_path / "no_num_shards.yaml"
        path.write_text(_yaml.safe_dump(raw, sort_keys=False))
        spec = load_dataset_spec_yaml(path)
        assert spec.num_shards == 3
