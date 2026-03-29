"""Tests for scripts/entrypoint_generate_dataset.py — generate_dataset entrypoint helper.

Tests are organized around the PUBLIC typed API:
- run(): full flow — materialize spec, upload, generate, upload shard
- _build_generate_args(): builds CLI args from a spec + shard + output_dir
- main(): reads env vars and delegates to run()
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.entrypoint_generate_dataset import _build_generate_args, main, run

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


def _make_mock_spec() -> MagicMock:
    """Create a mock DatasetPipelineSpec with all fields needed for arg building."""
    mock_spec = MagicMock()
    mock_spec.run_id = "test-dataset-20260328T180000Z"
    mock_spec.shard_size = 10000
    mock_spec.plugin_path = "plugins/Surge XT.vst3"
    mock_spec.preset_path = "presets/surge-base.vstpreset"
    mock_spec.sample_rate = 16000
    mock_spec.channels = 2
    mock_spec.velocity = 100
    mock_spec.signal_duration_seconds = 4.0
    mock_spec.min_loudness = -55.0
    mock_spec.param_spec = "surge_simple"
    mock_spec.sample_batch_size = 32
    mock_spec.model_dump_json.return_value = '{"run_id": "test"}'

    mock_shard = MagicMock()
    mock_shard.filename = "shard-000000.h5"
    mock_shard.shard_id = 0
    mock_shard.seed = 42
    mock_spec.shards = (mock_shard,)

    return mock_spec


# ---------------------------------------------------------------------------
# run — full flow orchestration
# ---------------------------------------------------------------------------


class TestRun:
    """Run() orchestrates: materialize → upload spec → generate → upload shard."""

    @patch("scripts.entrypoint_generate_dataset.subprocess.check_call")
    @patch("scripts.entrypoint_generate_dataset._rclone_copy")
    @patch("scripts.entrypoint_generate_dataset.materialize_spec")
    def test_writes_spec_json_to_metadata_dir(
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """spec.json is written to metadata_dir with valid JSON."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = _make_mock_spec()

        run(config_path, metadata_dir)

        spec_path = metadata_dir / "spec.json"
        assert spec_path.exists()
        assert spec_path.read_text() == '{"run_id": "test"}'

    @patch("scripts.entrypoint_generate_dataset.subprocess.check_call")
    @patch("scripts.entrypoint_generate_dataset._rclone_copy")
    @patch("scripts.entrypoint_generate_dataset.materialize_spec")
    def test_uploads_spec_to_r2_before_generation(
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Rclone uploads spec.json to R2 before generate_vst_dataset runs."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = _make_mock_spec()

        run(config_path, metadata_dir)

        rclone_calls = mock_rclone.call_args_list
        assert len(rclone_calls) == 2
        spec_upload = rclone_calls[0]
        assert "spec.json" in spec_upload[0][0]
        assert "r2:intermediate-data/" in spec_upload[0][1]

    @patch("scripts.entrypoint_generate_dataset.subprocess.check_call")
    @patch("scripts.entrypoint_generate_dataset._rclone_copy")
    @patch("scripts.entrypoint_generate_dataset.materialize_spec")
    def test_calls_generate_vst_dataset(
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """generate_vst_dataset.py is called as subprocess with spec-derived args."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = _make_mock_spec()

        run(config_path, metadata_dir)

        mock_check_call.assert_called_once()
        args = mock_check_call.call_args[0][0]
        assert "generate_vst_dataset.py" in args[1]
        assert "10000" in args  # shard_size

    @patch("scripts.entrypoint_generate_dataset.subprocess.check_call")
    @patch("scripts.entrypoint_generate_dataset._rclone_copy")
    @patch("scripts.entrypoint_generate_dataset.materialize_spec")
    def test_uploads_shard_to_r2_after_generation(
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Second rclone call uploads the shard to R2 after generation."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = _make_mock_spec()

        run(config_path, metadata_dir)

        rclone_calls = mock_rclone.call_args_list
        assert len(rclone_calls) == 2
        shard_upload = rclone_calls[1]
        assert "shard-000000.h5" in shard_upload[0][0]

    def test_num_shards_greater_than_one_raises(self, tmp_path: Path) -> None:
        """num_shards > 1 raises NotImplementedError."""
        config_path = _write_config(
            tmp_path,
            overrides={"num_shards": 48, "splits": {"train": 44, "val": 2, "test": 2}},
        )
        metadata_dir = tmp_path / "metadata"

        with pytest.raises(NotImplementedError, match="num_shards > 1"):
            run(config_path, metadata_dir)

    def test_wds_output_format_raises(self, tmp_path: Path) -> None:
        """output_format 'wds' raises ValueError."""
        config_path = _write_config(tmp_path, overrides={"output_format": "wds"})
        metadata_dir = tmp_path / "metadata"

        with pytest.raises(ValueError, match="only supports hdf5"):
            run(config_path, metadata_dir)


# ---------------------------------------------------------------------------
# _build_generate_args — arg construction from spec + shard
# ---------------------------------------------------------------------------


class TestBuildGenerateArgs:
    """_build_generate_args() produces correct CLI arg lists from spec + shard."""

    def test_output_file_uses_shard_filename(self, tmp_path: Path) -> None:
        """Output file path is {output_dir}/{shard.filename}."""
        spec = _make_mock_spec()
        shard = spec.shards[0]

        args = _build_generate_args(spec, shard, tmp_path)

        assert args[2] == str(tmp_path / "shard-000000.h5")

    def test_num_samples_is_shard_size(self) -> None:
        """num_samples arg comes from spec.shard_size."""
        spec = _make_mock_spec()
        shard = spec.shards[0]

        args = _build_generate_args(spec, shard, Path("out"))

        assert args[3] == "10000"

    def test_all_spec_fields_passed_as_options(self) -> None:
        """All generation parameters from spec are passed as --key value options."""
        spec = _make_mock_spec()
        shard = spec.shards[0]

        args = _build_generate_args(spec, shard, Path("out"))

        # Parse --key value pairs into a dict
        option_args = {}
        i = 4  # skip: python, script, output_file, shard_size
        while i < len(args):
            if args[i].startswith("--"):
                option_args[args[i].lstrip("-")] = args[i + 1]
                i += 2
            else:
                i += 1

        assert option_args["plugin_path"] == "plugins/Surge XT.vst3"
        assert option_args["preset_path"] == "presets/surge-base.vstpreset"
        assert option_args["sample_rate"] == "16000"
        assert option_args["channels"] == "2"
        assert option_args["velocity"] == "100"
        assert option_args["signal_duration_seconds"] == "4.0"
        assert option_args["min_loudness"] == "-55.0"
        assert option_args["param_spec"] == "surge_simple"
        assert option_args["sample_batch_size"] == "32"

    def test_args_start_with_python_and_script(self) -> None:
        """First arg is the Python executable, second is the generation script."""
        spec = _make_mock_spec()
        shard = spec.shards[0]

        args = _build_generate_args(spec, shard, Path("out"))

        assert "python" in args[0].lower() or args[0].endswith("/python3.10")
        assert args[1] == "src/data/vst/generate_vst_dataset.py"


# ---------------------------------------------------------------------------
# main — env var reading
# ---------------------------------------------------------------------------


class TestMainEnvVars:
    """Main() reads DATASET_CONFIG and RUN_METADATA_DIR from environment."""

    def test_missing_dataset_config_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing DATASET_CONFIG env var raises KeyError."""
        monkeypatch.delenv("DATASET_CONFIG", raising=False)
        monkeypatch.delenv("RUN_METADATA_DIR", raising=False)

        with pytest.raises(KeyError, match="DATASET_CONFIG"):
            main()

    @patch("scripts.entrypoint_generate_dataset.run")
    def test_default_metadata_dir(
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default RUN_METADATA_DIR is /run-metadata when env unset."""
        config_path = _write_config(tmp_path)
        monkeypatch.setenv("DATASET_CONFIG", str(config_path))
        monkeypatch.delenv("RUN_METADATA_DIR", raising=False)

        main()

        mock_run.assert_called_once_with(config_path, Path("/run-metadata"))

    @patch("scripts.entrypoint_generate_dataset.run")
    def test_custom_metadata_dir(
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RUN_METADATA_DIR env var overrides default."""
        config_path = _write_config(tmp_path)
        custom_dir = tmp_path / "custom-meta"
        monkeypatch.setenv("DATASET_CONFIG", str(config_path))
        monkeypatch.setenv("RUN_METADATA_DIR", str(custom_dir))

        main()

        mock_run.assert_called_once_with(config_path, custom_dir)
