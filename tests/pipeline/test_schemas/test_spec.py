from __future__ import annotations

import plistlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pipeline.schemas.config import DatasetConfig, SplitsConfig
from pipeline.schemas.prefix import DatasetConfigId
from pipeline.schemas.spec import (
    DatasetPipelineSpec,
    extract_renderer_version,
    materialize_spec,
)
from src.data.vst import param_specs

FIXED_NOW = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def patch_materialize_io(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch all I/O for materialize_spec tests.

    Returns plugin_path.
    """
    monkeypatch.setattr("pipeline.schemas.spec._get_git_sha", lambda: "abc123def456")
    monkeypatch.setattr("pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr(
        "pipeline.schemas.spec.datetime",
        type(
            "FakeDatetime",
            (),
            {
                "now": staticmethod(lambda tz: FIXED_NOW),
                "fromisoformat": datetime.fromisoformat,
            },
        )(),
    )
    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')
    return tmp_path / "FakePlugin.vst3"


class TestDatasetPipelineSpec:
    """Behavioral contracts for DatasetPipelineSpec frozen model."""

    def test_pipeline_spec_is_frozen(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """Assigning to a frozen DatasetPipelineSpec field raises ValidationError."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)
        with pytest.raises(ValidationError):
            spec.num_shards = 99  # type: ignore[misc]

    def test_pipeline_spec_rejects_extra_fields(self, patch_materialize_io: Path) -> None:
        """Extra fields on DatasetPipelineSpec raise ValidationError."""
        kwargs: dict[str, Any] = {
            "run_id": "test-run",
            "created_at": FIXED_NOW,
            "code_version": "abc123",
            "is_repo_dirty": False,
            "param_spec": "surge_simple",
            "renderer_version": "1.0.0",
            "output_format": "hdf5",
            "sample_rate": 16000,
            "shard_size": 100,
            "num_shards": 1,
            "base_seed": 42,
            "num_params": 92,
            "splits": SplitsConfig(train=1, val=0, test=0),
            "plugin_path": str(patch_materialize_io),
            "preset_path": "presets/test.vstpreset",
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "sample_batch_size": 32,
            "extra_field": "oops",
        }
        with pytest.raises(ValidationError):
            DatasetPipelineSpec(**kwargs)

    def test_pipeline_spec_output_format_rejects_invalid_literal(
        self, patch_materialize_io: Path
    ) -> None:
        """Invalid output_format literal raises ValidationError."""
        kwargs: dict[str, Any] = {
            "run_id": "test-run",
            "created_at": FIXED_NOW,
            "code_version": "abc123",
            "is_repo_dirty": False,
            "param_spec": "surge_simple",
            "renderer_version": "1.0.0",
            "output_format": "parquet",
            "sample_rate": 16000,
            "shard_size": 100,
            "num_shards": 1,
            "base_seed": 42,
            "num_params": 92,
            "splits": SplitsConfig(train=1, val=0, test=0),
            "plugin_path": str(patch_materialize_io),
            "preset_path": "presets/test.vstpreset",
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "sample_batch_size": 32,
        }
        with pytest.raises(ValidationError):
            DatasetPipelineSpec(**kwargs)

    def test_direct_construction_with_bad_plugin_path_raises_validation_error(self) -> None:
        """Constructing DatasetPipelineSpec with nonexistent plugin_path raises ValidationError."""
        kwargs: dict[str, Any] = {
            "run_id": "test-run",
            "created_at": datetime(2026, 3, 28, tzinfo=timezone.utc),
            "code_version": "abc123",
            "is_repo_dirty": False,
            "param_spec": "surge_simple",
            "renderer_version": "1.0.0",
            "output_format": "hdf5",
            "sample_rate": 16000,
            "shard_size": 100,
            "num_shards": 1,
            "base_seed": 42,
            "num_params": 92,
            "plugin_path": "/nonexistent/plugin.vst3",
            "preset_path": "presets/test.vstpreset",
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "sample_batch_size": 32,
            "splits": SplitsConfig(train=1, val=0, test=0),
        }
        with pytest.raises(ValidationError, match="Plugin path does not exist"):
            DatasetPipelineSpec(**kwargs)


class TestExtractRendererVersion:
    """Platform-specific VST3 plugin version extraction."""

    def test_extracts_version_from_linux_moduleinfo_json(self, tmp_path: Path) -> None:
        """Linux moduleinfo.json with Version key returns the version string."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')
        assert extract_renderer_version(plugin) == "1.3.4"

    def test_extracts_version_from_macos_info_plist(self, tmp_path: Path) -> None:
        """MacOS Info.plist with CFBundleShortVersionString returns the version."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        plist_data = {"CFBundleShortVersionString": "1.3.4"}
        (contents / "Info.plist").write_bytes(plistlib.dumps(plist_data))
        assert extract_renderer_version(plugin) == "1.3.4"

    def test_prefers_moduleinfo_json_when_both_exist(self, tmp_path: Path) -> None:
        """When both version files exist, moduleinfo.json takes precedence."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        (contents / "moduleinfo.json").write_text('{"Version": "2.0.0"}')
        plist_data = {"CFBundleShortVersionString": "1.0.0"}
        (contents / "Info.plist").write_bytes(plistlib.dumps(plist_data))
        assert extract_renderer_version(plugin) == "2.0.0"

    def test_raises_file_not_found_when_no_version_file(self, tmp_path: Path) -> None:
        """Empty Contents directory raises FileNotFoundError."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            extract_renderer_version(plugin)

    def test_raises_key_error_when_version_field_missing(self, tmp_path: Path) -> None:
        """moduleinfo.json without Version key raises KeyError."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        (contents / "moduleinfo.json").write_text('{"Name": "TestPlugin"}')
        with pytest.raises(KeyError):
            extract_renderer_version(plugin)


class TestMaterializeSpec:
    """Behavioral tests for materialize_spec."""

    def test_all_fields_populated(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """Spec has all fields set to expected values."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.run_id == "ci-smoke-test-20260328T120000Z"
        assert spec.created_at == FIXED_NOW
        assert spec.code_version == "abc123def456"
        assert spec.is_repo_dirty is False
        assert spec.renderer_version == "1.3.4"
        assert spec.output_format == "hdf5"
        assert spec.sample_rate == 16000
        assert spec.shard_size == 10000
        assert spec.num_shards == 1
        assert spec.base_seed == 42
        assert spec.param_spec == "surge_simple"
        assert spec.num_params == len(param_specs["surge_simple"])
        assert spec.splits == SplitsConfig(train=1, val=0, test=0)
        assert spec.plugin_path == str(patch_materialize_io)
        assert spec.preset_path == "presets/surge-base.vstpreset"
        assert spec.velocity == 100
        assert spec.signal_duration_seconds == 4.0
        assert spec.min_loudness == -55.0
        assert spec.sample_batch_size == 32

    def test_num_params_resolved_from_registry(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """num_params matches the length of the param_specs registry entry."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.num_params == len(param_specs["surge_simple"])

    def test_json_round_trip_preserves_all_fields(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """JSON serialize then deserialize produces an equal DatasetPipelineSpec."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        json_str = spec.model_dump_json()
        restored = DatasetPipelineSpec.model_validate_json(json_str)
        assert restored == spec

    def test_run_id_format_from_config_id_and_timestamp(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """Run ID combines config_id and UTC timestamp."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.run_id == "ci-smoke-test-20260328T120000Z"

    def test_created_at_is_utc_iso_format(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """created_at is a timezone-aware UTC datetime."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.created_at.tzinfo is not None
        offset = spec.created_at.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0

    def test_code_version_from_git(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """code_version is the mocked git SHA."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.code_version == "abc123def456"

    @pytest.mark.parametrize("dirty_value", [True, False])
    def test_is_repo_dirty_reflects_git_state(
        self,
        patch_materialize_io: Path,
        valid_config_dict: dict,
        monkeypatch: pytest.MonkeyPatch,
        dirty_value: bool,
    ) -> None:
        """is_repo_dirty matches the git dirty state."""
        monkeypatch.setattr("pipeline.schemas.spec._is_repo_dirty", lambda: dirty_value)
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.is_repo_dirty is dirty_value

    def test_renderer_version_from_plugin(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """renderer_version comes from the plugin's moduleinfo.json."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.renderer_version == "1.3.4"

    def test_unknown_param_spec_raises_key_error(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """Unknown param_spec name raises KeyError."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["param_spec"] = "nonexistent_synth"
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")

        with pytest.raises(KeyError):
            materialize_spec(config, config_id)

    def test_missing_plugin_raises_file_not_found(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """Nonexistent plugin_path raises FileNotFoundError."""
        valid_config_dict["plugin_path"] = "/nonexistent/path"
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")

        with pytest.raises(FileNotFoundError):
            materialize_spec(config, config_id)

    def test_wds_output_format_raises_not_implemented(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """WDS output format is not yet supported."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["output_format"] = "wds"
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        with pytest.raises(NotImplementedError, match="wds"):
            materialize_spec(config, config_id)


class TestMaterializeSpecIntegration:
    """Integration test with real I/O, no mocks."""

    def test_materialize_spec_end_to_end_with_real_git_and_fixture_plugin(
        self, valid_config_dict: dict
    ) -> None:
        """Real git + fixture plugin produces a valid spec with expected fields."""
        fixture_plugin = Path(__file__).parent.parent / "fixtures" / "TestPlugin.vst3"
        valid_config_dict["plugin_path"] = str(fixture_plugin)
        valid_config_dict["num_shards"] = 1
        valid_config_dict["splits"] = {"train": 1, "val": 0, "test": 0}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("integration-test")
        spec = materialize_spec(config, config_id)

        assert re.fullmatch(r"[0-9a-f]{40}", spec.code_version)
        assert isinstance(spec.is_repo_dirty, bool)
        assert spec.renderer_version == "1.0.0-test"
        assert spec.created_at.tzinfo is not None
        assert spec.run_id.startswith("integration-test-")
        assert spec.num_shards == 1
        assert spec.num_params > 0
