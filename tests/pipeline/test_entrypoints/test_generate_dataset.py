"""Tests for pipeline/entrypoints/generate_dataset.py — generate_dataset entrypoint helper.

Tests are organized around the PUBLIC typed API:
- run(): full flow — materialize spec, upload, generate, upload shard
- build_generate_args(): builds CLI args from a spec + shard + output_dir
- main(): reads env vars and delegates to run()
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pipeline.constants import R2_BUCKET
from pipeline.entrypoints.generate_dataset import build_generate_args, main, run

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


@pytest.fixture()
def real_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[return]
    """Create a real DatasetPipelineSpec with mocked I/O."""
    from pipeline.schemas.config import DatasetConfig, SplitsConfig
    from pipeline.schemas.prefix import DatasetConfigId
    from pipeline.schemas.spec import materialize_spec

    monkeypatch.setattr("pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("pipeline.schemas.spec._is_repo_dirty", lambda: False)
    fixed_now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "pipeline.schemas.spec.datetime",
        type(
            "FakeDatetime",
            (),
            {
                "now": staticmethod(lambda tz: fixed_now),
                "fromisoformat": datetime.fromisoformat,
            },
        )(),
    )

    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')

    config = DatasetConfig(
        param_spec="surge_simple",
        plugin_path=str(tmp_path / "FakePlugin.vst3"),
        output_format="hdf5",
        sample_rate=16000,
        shard_size=10000,
        num_shards=1,
        base_seed=42,
        splits=SplitsConfig(train=1, val=0, test=0),
        preset_path="presets/surge-base.vstpreset",
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-55.0,
        sample_batch_size=32,
    )
    return materialize_spec(config, DatasetConfigId("test-dataset"))


# ---------------------------------------------------------------------------
# run — full flow orchestration
# ---------------------------------------------------------------------------


class TestRun:
    """Run() orchestrates: materialize → upload spec → generate → upload shard."""

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    @patch("pipeline.entrypoints.generate_dataset.materialize_spec")
    def test_writes_spec_json_to_metadata_dir(
        # plumb:req-5bd551a7
        # plumb:req-470fb0bc
        # plumb:req-3e671363
        # plumb:req-4cdfd71a
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        real_spec: object,
    ) -> None:
        """input_spec.json is written to metadata_dir as valid JSON with a run_id field."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = real_spec

        run(config_path, metadata_dir)

        spec_path = metadata_dir / "input_spec.json"
        assert spec_path.exists()
        data = json.loads(spec_path.read_text())
        assert "run_id" in data

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    @patch("pipeline.entrypoints.generate_dataset.materialize_spec")
    def test_uploads_spec_to_r2_before_generation(
        # plumb:req-6e69d1c2
        # plumb:req-d0135b99
        # plumb:req-c0a8e86f
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        real_spec: object,
    ) -> None:
        """Rclone uploads input_spec.json to R2 before generate_vst_dataset runs."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = real_spec

        # Track call order across both mocks using a shared parent mock
        manager = MagicMock()
        manager.attach_mock(mock_rclone, "rclone")
        manager.attach_mock(mock_check_call, "check_call")

        run(config_path, metadata_dir)

        rclone_calls = mock_rclone.call_args_list
        assert len(rclone_calls) == 2
        spec_upload = rclone_calls[0]
        assert "input_spec.json" in spec_upload[0][0]
        assert f"r2:{R2_BUCKET}/" in spec_upload[0][1]

        # Ordering: spec upload must appear before check_call in the shared call log
        call_names = [c[0] for c in manager.mock_calls]
        assert call_names.index("rclone") < call_names.index("check_call")

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    @patch("pipeline.entrypoints.generate_dataset.materialize_spec")
    def test_calls_generate_vst_dataset(
        # plumb:req-1cb78576
        # plumb:req-eee4f671
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        real_spec: object,
    ) -> None:
        """generate_vst_dataset.py is called as subprocess with spec-derived args."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = real_spec

        run(config_path, metadata_dir)

        mock_check_call.assert_called_once()
        args = mock_check_call.call_args[0][0]
        assert "generate_vst_dataset.py" in args[1]
        assert "10000" in args  # shard_size from real_spec.shard_size

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    @patch("pipeline.entrypoints.generate_dataset.materialize_spec")
    def test_uploads_shard_to_r2_after_generation(
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        real_spec: object,
    ) -> None:
        """Second rclone call uploads the shard to R2 after generation."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = real_spec

        run(config_path, metadata_dir)

        rclone_calls = mock_rclone.call_args_list
        assert len(rclone_calls) == 2
        shard_upload = rclone_calls[1]
        assert "shard-000000.h5" in shard_upload[0][0]

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    @patch("pipeline.entrypoints.generate_dataset.materialize_spec")
    def test_subprocess_failure_propagates(
        # plumb:req-51993a38
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        real_spec: object,
    ) -> None:
        """CalledProcessError from generate_vst_dataset propagates to caller."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = real_spec
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "generate_vst_dataset.py")

        with pytest.raises(subprocess.CalledProcessError):
            run(config_path, metadata_dir)

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    @patch("pipeline.entrypoints.generate_dataset.materialize_spec")
    def test_rclone_failure_propagates(
        self,
        mock_materialize: MagicMock,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        real_spec: object,
    ) -> None:
        """CalledProcessError from rclone propagates to caller."""
        config_path = _write_config(tmp_path)
        metadata_dir = tmp_path / "metadata"
        mock_materialize.return_value = real_spec
        mock_rclone.side_effect = subprocess.CalledProcessError(1, "rclone")

        with pytest.raises(subprocess.CalledProcessError):
            run(config_path, metadata_dir)

    def test_num_shards_greater_than_one_raises(self, tmp_path: Path) -> None:
        # plumb:req-bbe52a4d
        # plumb:req-c52244a4
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
# build_generate_args — arg construction from spec + shard
# ---------------------------------------------------------------------------


class TestBuildGenerateArgs:
    """build_generate_args() produces correct CLI arg lists from spec + shard."""

    def test_output_file_uses_shard_filename(self, real_spec: object, tmp_path: Path) -> None:
        # plumb:req-9f751e55
        """Output file path is {output_dir}/{shard.filename}."""
        shard = real_spec.shards[0]  # type: ignore[union-attr]

        args = build_generate_args(real_spec, shard, tmp_path)  # type: ignore[arg-type]

        assert args[2] == str(tmp_path / "shard-000000.h5")

    def test_num_samples_is_shard_size(self, real_spec: object) -> None:
        # plumb:req-e2d52f55
        """num_samples arg comes from spec.shard_size."""
        shard = real_spec.shards[0]  # type: ignore[union-attr]

        args = build_generate_args(real_spec, shard, Path("out"))  # type: ignore[arg-type]

        assert args[3] == "10000"

    def test_all_spec_fields_passed_as_options(self, real_spec: object) -> None:
        # plumb:req-2b453be4
        # plumb:req-37250f13
        """All generation parameters from spec are passed as --key value options."""
        shard = real_spec.shards[0]  # type: ignore[union-attr]

        args = build_generate_args(real_spec, shard, Path("out"))  # type: ignore[arg-type]

        # Parse --key value pairs into a dict (skip: python, script, output_file, shard_size)
        option_keys = set()
        i = 4
        while i < len(args):
            if args[i].startswith("--"):
                option_keys.add(args[i].lstrip("-"))
                i += 2
            else:
                i += 1

        expected_keys = {
            "plugin_path",
            "preset_path",
            "sample_rate",
            "channels",
            "velocity",
            "signal_duration_seconds",
            "min_loudness",
            "param_spec",
            "sample_batch_size",
        }
        assert expected_keys <= option_keys

    def test_args_start_with_python_and_script(self, real_spec: object) -> None:
        """First arg is the Python executable, second is the generation script."""
        shard = real_spec.shards[0]  # type: ignore[union-attr]

        args = build_generate_args(real_spec, shard, Path("out"))  # type: ignore[arg-type]

        assert "python" in args[0].lower() or args[0].endswith("/python3.10")
        assert args[1] == "src/data/vst/generate_vst_dataset.py"


# ---------------------------------------------------------------------------
# main — env var reading
# ---------------------------------------------------------------------------


class TestMainEnvVars:
    """Main() reads DATASET_CONFIG and RUN_METADATA_DIR from environment."""

    def test_missing_dataset_config_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # plumb:req-a7360ffe
        # plumb:req-d1239d1d
        # plumb:req-34cac29a
        """Missing DATASET_CONFIG env var raises KeyError."""
        monkeypatch.delenv("DATASET_CONFIG", raising=False)
        monkeypatch.delenv("RUN_METADATA_DIR", raising=False)

        with pytest.raises(KeyError, match="DATASET_CONFIG"):
            main()

    @patch("pipeline.entrypoints.generate_dataset.run")
    def test_default_metadata_dir(
        # plumb:req-84ebde7c
        # plumb:req-34509493
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default RUN_METADATA_DIR is /run-metadata when env unset."""
        config_path = _write_config(tmp_path)
        monkeypatch.setenv("DATASET_CONFIG", str(config_path))
        monkeypatch.delenv("RUN_METADATA_DIR", raising=False)

        main()

        mock_run.assert_called_once_with(config_path, Path("/run-metadata"))

    @patch("pipeline.entrypoints.generate_dataset.run")
    def test_custom_metadata_dir(
        # plumb:req-6569ee0e
        self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RUN_METADATA_DIR env var overrides default."""
        config_path = _write_config(tmp_path)
        custom_dir = tmp_path / "custom-meta"
        monkeypatch.setenv("DATASET_CONFIG", str(config_path))
        monkeypatch.setenv("RUN_METADATA_DIR", str(custom_dir))

        main()

        mock_run.assert_called_once_with(config_path, custom_dir)
