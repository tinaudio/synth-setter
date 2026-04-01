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
    ShardSpec,
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


class TestShardSpec:
    """Behavioral contracts for ShardSpec frozen model."""

    def test_shard_spec_is_frozen(self) -> None:
        # plumb:req-9c3bfede
        # plumb:req-74aa845b
        # plumb:req-8d71872b
        """Assigning to a frozen ShardSpec field raises ValidationError."""
        shard = ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42)
        with pytest.raises(ValidationError):
            shard.shard_id = 99  # type: ignore[misc]

    def test_shard_spec_rejects_extra_fields(self) -> None:
        """Extra fields on ShardSpec raise ValidationError."""
        with pytest.raises(ValidationError):
            ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42, extra="oops")  # type: ignore[call-arg]


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
            spec.shard_size = 99  # type: ignore[misc]

    def test_pipeline_spec_rejects_extra_fields(self, patch_materialize_io: Path) -> None:
        """Extra fields on DatasetPipelineSpec raise ValidationError."""
        kwargs: dict[str, Any] = {
            "run_id": "test-run",
            "r2_prefix": "data/test/test-run/",
            "created_at": FIXED_NOW,
            "code_version": "abc123",
            "is_repo_dirty": False,
            "param_spec": "surge_simple",
            "renderer_version": "1.0.0",
            "output_format": "hdf5",
            "sample_rate": 16000,
            "shard_size": 100,
            "base_seed": 42,
            "num_params": 92,
            "splits": SplitsConfig(train=1, val=0, test=0),
            "plugin_path": str(patch_materialize_io),
            "preset_path": "presets/test.vstpreset",
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "sample_batch_size": 32,
            "shards": (ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42),),
            "extra_field": "oops",
        }
        with pytest.raises(ValidationError):
            DatasetPipelineSpec(**kwargs)

    def test_pipeline_spec_output_format_rejects_invalid_literal(
        # plumb:req-706a57a8
        # plumb:req-c8dabc9b
        # plumb:req-17055172
        # plumb:req-f1a9ffe7
        self,
        patch_materialize_io: Path,
    ) -> None:
        """Invalid output_format literal raises ValidationError."""
        kwargs: dict[str, Any] = {
            "run_id": "test-run",
            "r2_prefix": "data/test/test-run/",
            "created_at": FIXED_NOW,
            "code_version": "abc123",
            "is_repo_dirty": False,
            "param_spec": "surge_simple",
            "renderer_version": "1.0.0",
            "output_format": "parquet",
            "sample_rate": 16000,
            "shard_size": 100,
            "base_seed": 42,
            "num_params": 92,
            "splits": SplitsConfig(train=1, val=0, test=0),
            "plugin_path": str(patch_materialize_io),
            "preset_path": "presets/test.vstpreset",
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "sample_batch_size": 32,
            "shards": (),
        }
        with pytest.raises(ValidationError):
            DatasetPipelineSpec(**kwargs)

    def test_direct_construction_with_nonexistent_plugin_path_succeeds(self) -> None:
        """Constructing DatasetPipelineSpec with nonexistent plugin_path succeeds.

        Plugin path validation is in materialize_spec(), not the model — so deserialization on
        machines without the plugin works.
        """
        kwargs: dict[str, Any] = {
            "run_id": "test-run",
            "r2_prefix": "data/test/test-run/",
            "created_at": datetime(2026, 3, 28, tzinfo=timezone.utc),
            "code_version": "abc123",
            "is_repo_dirty": False,
            "param_spec": "surge_simple",
            "renderer_version": "1.0.0",
            "output_format": "hdf5",
            "sample_rate": 16000,
            "shard_size": 100,
            "base_seed": 42,
            "num_params": 92,
            "plugin_path": "/nonexistent/plugin.vst3",
            "preset_path": "presets/test.vstpreset",
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "sample_batch_size": 32,
            "splits": SplitsConfig(train=1, val=0, test=0),
            "shards": (ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42),),
        }
        spec = DatasetPipelineSpec(**kwargs)
        assert spec.plugin_path == "/nonexistent/plugin.vst3"

    def test_json_round_trip_without_plugin_on_disk(self) -> None:
        # plumb:req-470fb0bc
        """JSON round-trip works even when plugin_path doesn't exist on disk."""
        kwargs: dict[str, Any] = {
            "run_id": "test-run",
            "r2_prefix": "data/test/test-run/",
            "created_at": datetime(2026, 3, 28, tzinfo=timezone.utc),
            "code_version": "abc123",
            "is_repo_dirty": False,
            "param_spec": "surge_simple",
            "renderer_version": "1.0.0",
            "output_format": "hdf5",
            "sample_rate": 16000,
            "shard_size": 100,
            "base_seed": 42,
            "num_params": 92,
            "plugin_path": "/nonexistent/plugin.vst3",
            "preset_path": "presets/test.vstpreset",
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "sample_batch_size": 32,
            "splits": SplitsConfig(train=1, val=0, test=0),
            "shards": (ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42),),
        }
        spec = DatasetPipelineSpec(**kwargs)
        json_str = spec.model_dump_json()
        restored = DatasetPipelineSpec.model_validate_json(json_str)
        assert restored == spec


class TestExtractRendererVersion:
    """Platform-specific VST3 plugin version extraction."""

    def test_extracts_version_from_linux_moduleinfo_json(self, tmp_path: Path) -> None:
        # plumb:req-2b453be4
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

    def test_raises_when_no_version_file_and_no_loadable_plugin(self, tmp_path: Path) -> None:
        # plumb:req-51993a38
        """Empty Contents directory with no loadable plugin raises an error."""
        plugin = tmp_path / "Plugin.vst3"
        contents = plugin / "Contents"
        contents.mkdir(parents=True)
        with pytest.raises(Exception):  # noqa: B017
            extract_renderer_version(plugin)

    def test_raises_file_not_found_when_plugin_path_does_not_exist(self, tmp_path: Path) -> None:
        """Nonexistent plugin path raises FileNotFoundError with clear message."""
        plugin = tmp_path / "nonexistent.vst3"
        with pytest.raises(FileNotFoundError, match="Plugin path does not exist"):
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
        assert len(spec.shards) == 1
        assert spec.shards[0].shard_id == 0
        assert spec.shards[0].filename == "shard-000000.h5"
        assert spec.shards[0].seed == 42
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
        # plumb:req-6df7c153
        # plumb:req-1e7bbada
        self,
        patch_materialize_io: Path,
        valid_config_dict: dict,
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
        # plumb:req-e9f67970
        self,
        patch_materialize_io: Path,
        valid_config_dict: dict,
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
        # plumb:req-bd3e7a97
        self,
        patch_materialize_io: Path,
        valid_config_dict: dict,
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
        # plumb:req-9c9e072c
        self,
        patch_materialize_io: Path,
        valid_config_dict: dict,
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

    def test_multi_shard_seeds_are_base_plus_shard_id(
        # plumb:req-37250f13
        # plumb:req-455c526a
        # plumb:req-e611852b
        self,
        patch_materialize_io: Path,
        valid_config_dict: dict,
    ) -> None:
        """Each shard seed equals base_seed + shard_id."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        valid_config_dict["num_shards"] = 3
        valid_config_dict["splits"] = {"train": 1, "val": 1, "test": 1}
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert [s.seed for s in spec.shards] == [42, 43, 44]

    def test_filenames_are_zero_padded_six_digits(
        # plumb:req-9f751e55
        # plumb:req-7e6f8ef8
        self,
        patch_materialize_io: Path,
        valid_config_dict: dict,
    ) -> None:
        """Shard filenames use six-digit zero-padded indices."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.shards[0].filename == "shard-000000.h5"
        assert spec.shards[-1].filename == "shard-000047.h5"

    def test_num_shards_property_matches_shards_length(
        self, patch_materialize_io: Path, valid_config_dict: dict
    ) -> None:
        """num_shards property returns len(shards)."""
        valid_config_dict["plugin_path"] = str(patch_materialize_io)
        config = DatasetConfig(**valid_config_dict)
        config_id = DatasetConfigId("ci-smoke-test")
        spec = materialize_spec(config, config_id)

        assert spec.num_shards == len(spec.shards)


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
        assert spec.shards[0].seed == 42
        assert spec.num_params > 0
