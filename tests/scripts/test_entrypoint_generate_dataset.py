"""Tests for scripts/entrypoint_generate_dataset.py — generate_dataset entrypoint helper.

Tests are organized around the PUBLIC typed API:
- build_generate_args(): builds CLI args from a dataset config path
- main(): reads env vars and dispatches to generate_vst_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.entrypoint_generate_dataset import build_generate_args

_COMPLETE_CONFIG = {
    "param_spec": "surge_simple",
    "plugin_path": "plugins/Surge XT.vst3",
    "output_format": "hdf5",
    "sample_rate": 16000,
    "shard_size": 10000,
    "num_shards": 1,
    "base_seed": 42,
    "splits": {"train": 1, "val": 0, "test": 0},
    "preset_path": "presets/surge-base.vstpreset",
    "channels": 2,
    "velocity": 100,
    "signal_duration_seconds": 4.0,
    "min_loudness": -55.0,
    "sample_batch_size": 32,
}


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Write a complete dataset config YAML and return its path."""
    data = {**_COMPLETE_CONFIG, **(overrides or {})}
    config_path = tmp_path / "test-dataset.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return config_path


# ---------------------------------------------------------------------------
# build_generate_args — arg construction
# ---------------------------------------------------------------------------


class TestBuildGenerateArgs:
    """build_generate_args() produces correct CLI arg lists."""

    def test_output_file_uses_shard_id(self, tmp_path: Path) -> None:
        """Output path is {output_dir}/shard-000000.hdf5."""
        config_path = _write_config(tmp_path)
        output_dir = tmp_path / "out"

        args = build_generate_args(config_path, output_dir=output_dir)

        output_file = args[2]
        assert output_file == str(output_dir / "shard-000000.hdf5")

    def test_num_samples_is_shard_size(self, tmp_path: Path) -> None:
        """num_samples equals shard_size (not shard_size * num_shards)."""
        config_path = _write_config(tmp_path)

        args = build_generate_args(config_path, output_dir=tmp_path)

        assert args[3] == "10000"

    def test_config_fields_passed_as_options(self, tmp_path: Path) -> None:
        """All config fields are passed as CLI options."""
        config_path = _write_config(tmp_path)

        args = build_generate_args(config_path, output_dir=tmp_path)

        option_args = {}
        i = 4  # Skip: python, script, output_file, shard_size
        while i < len(args):
            if args[i].startswith("--"):
                option_args[args[i]] = args[i + 1]
                i += 2
            else:
                i += 1

        assert option_args["--plugin_path"] == "plugins/Surge XT.vst3"
        assert option_args["--preset_path"] == "presets/surge-base.vstpreset"
        assert option_args["--sample_rate"] == "16000"
        assert option_args["--channels"] == "2"
        assert option_args["--velocity"] == "100"
        assert option_args["--signal_duration_seconds"] == "4.0"
        assert option_args["--min_loudness"] == "-55.0"
        assert option_args["--param_spec"] == "surge_simple"
        assert option_args["--sample_batch_size"] == "32"

    def test_args_start_with_python_and_script(self, tmp_path: Path) -> None:
        """First arg is the Python executable, second is the generation script."""
        config_path = _write_config(tmp_path)

        args = build_generate_args(config_path, output_dir=tmp_path)

        assert "python" in args[0].lower() or args[0].endswith("/python3.10")
        assert args[1] == "src/data/vst/generate_vst_dataset.py"


# ---------------------------------------------------------------------------
# build_generate_args — output_format validation
# ---------------------------------------------------------------------------


class TestOutputFormatValidation:
    """build_generate_args() rejects non-hdf5 output formats."""

    def test_wds_output_format_raises(self, tmp_path: Path) -> None:
        """output_format 'wds' raises ValueError since only hdf5 is supported."""
        config_path = _write_config(tmp_path, overrides={"output_format": "wds"})

        with pytest.raises(ValueError, match="only supports hdf5"):
            build_generate_args(config_path, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# build_generate_args — multi-shard guard
# ---------------------------------------------------------------------------


class TestMultiShardGuard:
    """build_generate_args() enforces single-shard MVP."""

    def test_num_shards_greater_than_one_raises(self, tmp_path: Path) -> None:
        """num_shards > 1 raises NotImplementedError."""
        config_path = _write_config(
            tmp_path,
            overrides={"num_shards": 48, "splits": {"train": 44, "val": 2, "test": 2}},
        )

        with pytest.raises(NotImplementedError, match="num_shards > 1"):
            build_generate_args(config_path, output_dir=tmp_path)

    def test_num_shards_one_succeeds(self, tmp_path: Path) -> None:
        """num_shards=1 succeeds and passes shard_size as num_samples."""
        config_path = _write_config(tmp_path, overrides={"num_shards": 1})

        args = build_generate_args(config_path, output_dir=tmp_path)

        assert args[3] == str(_COMPLETE_CONFIG["shard_size"])


# ---------------------------------------------------------------------------
# main — env var reading
# ---------------------------------------------------------------------------


class TestMainEnvVars:
    """Main() reads DATASET_CONFIG and OUTPUT_DIR from environment."""

    def test_missing_dataset_config_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing DATASET_CONFIG env var raises KeyError."""
        monkeypatch.delenv("DATASET_CONFIG", raising=False)
        monkeypatch.delenv("OUTPUT_DIR", raising=False)

        from scripts.entrypoint_generate_dataset import main

        with pytest.raises(KeyError, match="DATASET_CONFIG"):
            main()
