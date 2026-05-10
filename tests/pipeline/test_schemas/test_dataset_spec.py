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
    def test_shard_spec_is_frozen(self) -> None:
        shard = ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42)
        with pytest.raises(ValidationError):
            shard.shard_id = 99  # type: ignore[misc]

    def test_shard_spec_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42, extra="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RenderConfig
# ---------------------------------------------------------------------------


class TestRenderConfig:
    def test_render_config_rejects_extra_fields(self) -> None:
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
        kwargs = _valid_render_kwargs()
        kwargs[field] = bad_value
        with pytest.raises(ValidationError, match=match):
            RenderConfig(**kwargs)

    def test_render_config_velocity_bounds_are_inclusive(self) -> None:
        for valid in (0, 127):
            cfg = RenderConfig(**{**_valid_render_kwargs(), "velocity": valid})
            assert cfg.velocity == valid


# ---------------------------------------------------------------------------
# DatasetSpec — construction & runtime-field auto-fill
# ---------------------------------------------------------------------------


class TestDatasetSpecConstruction:
    def test_fresh_construction_fills_runtime_fields(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs())

        assert spec.git_sha == "abc123def456"
        assert spec.is_repo_dirty is False
        assert spec.created_at == FIXED_NOW
        assert spec.run_id == "ci-smoke-test-20260328T120000000Z"
        assert spec.r2_prefix == "data/ci-smoke-test/ci-smoke-test-20260328T120000000Z/"

    def test_run_id_uses_explicit_value_when_present(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs(run_id="custom-run-id-001"))
        assert spec.run_id == "custom-run-id-001"

    def test_r2_prefix_uses_explicit_value_when_present(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs(r2_prefix="custom/prefix/here/"))
        assert spec.r2_prefix == "custom/prefix/here/"

    def test_r2_prefix_root_default_is_data(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.r2_prefix.startswith("data/")

    def test_r2_prefix_root_custom_threads_through(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs(r2_prefix_root="experiments"))
        assert spec.r2_prefix.startswith("experiments/")

    def test_dataset_spec_strict_rejects_extra_fields(self, patch_runtime_io: None) -> None:
        kwargs = _valid_spec_kwargs(unexpected_field="surprise")
        with pytest.raises(ValidationError):
            DatasetSpec(**kwargs)


class TestDatasetSpecValidators:
    def test_r2_bucket_blank_raises(self, patch_runtime_io: None) -> None:
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="r2_bucket must not be blank"):
                DatasetSpec(**_valid_spec_kwargs(r2_bucket=blank))

    def test_task_name_blank_raises(self, patch_runtime_io: None) -> None:
        with pytest.raises(ValidationError, match="task_name must not be blank"):
            DatasetSpec(**_valid_spec_kwargs(task_name="   "))

    def test_explicit_r2_prefix_missing_trailing_slash_raises(
        self, patch_runtime_io: None
    ) -> None:
        with pytest.raises(ValidationError, match="r2_prefix must end with"):
            DatasetSpec(**_valid_spec_kwargs(r2_prefix="data/no/slash"))

    def test_split_size_not_multiple_of_batch_per_shard_raises(
        self, patch_runtime_io: None
    ) -> None:
        with pytest.raises(ValidationError, match="not a multiple"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(150, 0, 0)))

    def test_negative_split_size_raises(self, patch_runtime_io: None) -> None:
        with pytest.raises(ValidationError, match="must be non-negative"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(-100, 0, 0)))

    def test_zero_total_split_raises(self, patch_runtime_io: None) -> None:
        with pytest.raises(ValidationError, match="must sum to a positive count"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(0, 0, 0)))

    def test_invalid_output_format_literal_raises(self, patch_runtime_io: None) -> None:
        with pytest.raises(ValidationError):
            DatasetSpec(**_valid_spec_kwargs(output_format="parquet"))


# ---------------------------------------------------------------------------
# DatasetSpec — computed fields
# ---------------------------------------------------------------------------


class TestDatasetSpecComputedFields:
    def test_shards_count_matches_total_size_div_batch(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(400, 100, 100)))
        assert spec.num_shards == 6
        assert len(spec.shards) == 6

    def test_shard_seeds_are_base_plus_shard_id(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(300, 0, 0)))
        assert [s.seed for s in spec.shards] == [42, 43, 44]

    def test_shard_filenames_zero_padded_six_digits(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(300, 0, 0)))
        assert spec.shards[0].filename == "shard-000000.h5"
        assert spec.shards[-1].filename == "shard-000002.h5"

    @pytest.mark.parametrize(("output_format", "ext"), [("hdf5", ".h5"), ("wds", ".tar")])
    def test_shard_filename_extension_matches_output_format(
        self, patch_runtime_io: None, output_format: str, ext: str
    ) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs(output_format=output_format))
        assert all(s.filename.endswith(ext) for s in spec.shards)

    def test_num_params_resolved_from_registry(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.num_params == len(param_specs["surge_simple"])

    def test_unknown_param_spec_name_raises_at_compute(self, patch_runtime_io: None) -> None:
        kwargs = _valid_spec_kwargs()
        kwargs["render"] = {**kwargs["render"], "param_spec_name": "nonexistent_synth"}
        spec = DatasetSpec(**kwargs)
        with pytest.raises(KeyError):
            _ = spec.num_params


# ---------------------------------------------------------------------------
# DatasetSpec — JSON round-trip
# ---------------------------------------------------------------------------


class TestDatasetSpecRoundTrip:
    def test_json_round_trip_preserves_runtime_fields(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs())
        json_str = spec.model_dump_json()
        restored = DatasetSpec.model_validate_json(json_str)

        assert restored.git_sha == spec.git_sha
        assert restored.is_repo_dirty == spec.is_repo_dirty
        assert restored.created_at == spec.created_at
        assert restored.run_id == spec.run_id
        assert restored.r2_prefix == spec.r2_prefix

    def test_json_round_trip_preserves_shards(self, patch_runtime_io: None) -> None:
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=(400, 100, 100)))
        restored = DatasetSpec.model_validate_json(spec.model_dump_json())
        assert restored.shards == spec.shards
        assert restored.num_shards == spec.num_shards

    def test_json_round_trip_works_without_plugin_on_disk(self, patch_runtime_io: None) -> None:
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
    def test_extracts_stem(self) -> None:
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
    import yaml

    path = tmp_path / "ci-smoke-test.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(LEGACY_YAML_KWARGS), sort_keys=False))
    return path


class TestLoadDatasetSpecYaml:
    def test_legacy_yaml_round_trips_into_dataset_spec(
        self, patch_runtime_io: None, legacy_yaml: Path
    ) -> None:
        spec = load_dataset_spec_yaml(legacy_yaml)
        assert spec.task_name == "ci-smoke-test"
        assert spec.output_format == "hdf5"
        assert spec.train_val_test_sizes == (300, 0, 0)
        assert spec.render.plugin_path == "plugins/Surge XT.vst3"
        assert spec.render.batch_per_shard == 100
        assert spec.num_shards == 3

    def test_legacy_yaml_extends_merges_base(self, patch_runtime_io: None, tmp_path: Path) -> None:
        import yaml

        base = tmp_path / "base.yaml"
        base.write_text(yaml.safe_dump(copy.deepcopy(LEGACY_YAML_KWARGS), sort_keys=False))
        child = tmp_path / "child.yaml"
        child.write_text("_extends: base\noutput_format: wds\n")

        spec = load_dataset_spec_yaml(child)
        assert spec.output_format == "wds"
        assert spec.render.plugin_path == "plugins/Surge XT.vst3"
        assert spec.shards[0].filename.endswith(".tar")

    def test_legacy_yaml_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_dataset_spec_yaml(tmp_path / "nonexistent.yaml")

    def test_legacy_yaml_extends_missing_base_raises(self, tmp_path: Path) -> None:
        child = tmp_path / "child.yaml"
        child.write_text("_extends: nonexistent\noutput_format: wds\n")
        with pytest.raises(FileNotFoundError, match="_extends target not found"):
            load_dataset_spec_yaml(child)
