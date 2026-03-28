from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.config import DatasetConfig, dataset_config_id_from_path, load_dataset_config


class TestLoadDatasetConfig:
    """Tests for loading DatasetConfig from YAML files."""

    def test_load_dataset_config_valid_yaml_returns_model(self, write_config_yaml):
        """Valid YAML produces a fully populated DatasetConfig."""
        path = write_config_yaml()
        cfg = load_dataset_config(path)

        assert isinstance(cfg, DatasetConfig)
        assert cfg.param_spec == "surge_simple"
        assert cfg.plugin_path == "plugins/Surge XT.vst3"
        assert cfg.output_format == "hdf5"
        assert cfg.sample_rate == 16000
        assert cfg.shard_size == 10000
        assert cfg.num_shards == 48
        assert cfg.base_seed == 42
        assert cfg.splits.train == 44
        assert cfg.splits.val == 2
        assert cfg.splits.test == 2
        assert cfg.preset_path == "presets/surge-base.vstpreset"
        assert cfg.channels == 2
        assert cfg.velocity == 100
        assert cfg.signal_duration_seconds == 4.0
        assert cfg.min_loudness == -55.0
        assert cfg.sample_batch_size == 32

    def test_load_dataset_config_missing_file_raises_file_not_found(self, tmp_path):
        """Missing config path raises FileNotFoundError."""
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_dataset_config(missing)

    def test_load_dataset_config_directory_path_raises_file_not_found(self, tmp_path):
        """Passing a directory (not a file) raises FileNotFoundError, not IsADirectoryError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_dataset_config(tmp_path)

    def test_load_dataset_config_empty_yaml_raises_type_error(self, tmp_path):
        """Empty YAML file raises TypeError mentioning the file path."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(TypeError, match=str(empty)):
            load_dataset_config(empty)

    def test_load_dataset_config_non_mapping_yaml_raises_type_error(self, tmp_path):
        """YAML containing a list (not a mapping) raises TypeError mentioning the file path."""
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n")
        with pytest.raises(TypeError, match=str(tmp_path)):
            load_dataset_config(list_yaml)

    def test_load_dataset_config_invalid_yaml_raises(self, tmp_path):
        """Malformed YAML raises an exception during parsing."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("param_spec: [\ninvalid yaml")
        with pytest.raises(Exception):  # noqa: B017
            load_dataset_config(bad_yaml)


class TestDatasetConfigValidation:
    """Tests for DatasetConfig field validation."""

    def test_dataset_config_splits_must_sum_to_num_shards(self, valid_config_dict):
        """Splits that do not sum to num_shards raise ValidationError."""
        valid_config_dict["splits"] = {"train": 10, "val": 2, "test": 2}
        valid_config_dict["num_shards"] = 48
        with pytest.raises(ValidationError, match="splits sum.*!= num_shards"):
            DatasetConfig(**valid_config_dict)

    def test_dataset_config_output_format_defaults_to_hdf5(self, valid_config_dict):
        """Omitting output_format defaults to hdf5."""
        del valid_config_dict["output_format"]
        cfg = DatasetConfig(**valid_config_dict)
        assert cfg.output_format == "hdf5"

    def test_dataset_config_output_format_rejects_unknown(self, valid_config_dict):
        """Unknown output_format values are rejected."""
        valid_config_dict["output_format"] = "parquet"
        with pytest.raises(ValidationError):
            DatasetConfig(**valid_config_dict)

    def test_dataset_config_rejects_negative_shard_size(self, valid_config_dict):
        """Negative shard_size raises ValidationError."""
        valid_config_dict["shard_size"] = -1
        with pytest.raises(ValidationError, match="shard_size must be positive"):
            DatasetConfig(**valid_config_dict)

    def test_dataset_config_rejects_zero_num_shards(self, valid_config_dict):
        """Zero num_shards raises ValidationError."""
        valid_config_dict["num_shards"] = 0
        valid_config_dict["splits"] = {"train": 0, "val": 0, "test": 0}
        with pytest.raises(ValidationError, match="num_shards must be positive"):
            DatasetConfig(**valid_config_dict)

    def test_dataset_config_strict_rejects_extra_fields(self, valid_config_dict):
        """Extra fields are rejected by the strict model."""
        valid_config_dict["unexpected_field"] = "surprise"
        with pytest.raises(ValidationError):
            DatasetConfig(**valid_config_dict)

    def test_dataset_config_velocity_bounds(self, valid_config_dict):
        """Velocity outside [0, 127] raises; boundary values pass."""
        valid_config_dict["velocity"] = 200
        with pytest.raises(ValidationError, match="velocity must be in"):
            DatasetConfig(**valid_config_dict)

        valid_config_dict["velocity"] = 0
        cfg_zero = DatasetConfig(**valid_config_dict)
        assert cfg_zero.velocity == 0

        valid_config_dict["velocity"] = 127
        cfg_max = DatasetConfig(**valid_config_dict)
        assert cfg_max.velocity == 127


class TestDatasetConfigIdFromPath:
    """Tests for dataset_config_id_from_path."""

    def test_dataset_config_id_from_path_extracts_stem(self):
        """Extracts the filename stem as the config ID."""
        path = Path("configs/dataset/surge-simple-480k-10k.yaml")
        assert dataset_config_id_from_path(path) == "surge-simple-480k-10k"


class TestDatasetConfigRoundTrip:
    """Tests for DatasetConfig serialization round-trip."""

    def test_dataset_config_round_trip_json(self, valid_config_dict):
        """model_dump_json followed by model_validate_json produces an equal model."""
        original = DatasetConfig(**valid_config_dict)
        json_str = original.model_dump_json()
        restored = DatasetConfig.model_validate_json(json_str)
        assert original == restored


class TestConftestFixtureIsolation:
    """Tests verifying that conftest fixtures provide isolated copies."""

    def test_valid_config_dict_fixture_isolates_nested_mutations(self, valid_config_dict):
        """Mutating nested 'splits' in fixture copy must not corrupt the module-level
        VALID_CONFIG."""
        from tests.pipeline.conftest import VALID_CONFIG

        # Mutate the nested splits dict in-place
        valid_config_dict["splits"]["train"] = 9999

        # The module-level constant should be unaffected
        assert VALID_CONFIG["splits"]["train"] == 44, (
            "Shallow copy leaked: mutating fixture's nested dict corrupted VALID_CONFIG"
        )
