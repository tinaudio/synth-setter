"""Unit tests for scripts/run_dataset_pipeline.py.

The pipeline generates HDF5 splits via subprocess calls to
generate_vst_dataset.py and get_dataset_stats.py, then uploads via a
DatasetUploader. These tests mock the subprocess calls and use
LocalFakeUploader so they run without VST plugins, R2 credentials, or Docker.

To run:
    pytest tests/test_run_dataset_pipeline.py -v
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import h5py
import hdf5plugin
import numpy as np
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.run_dataset_pipeline import (
    PARAM_SPEC_TO_DATA_CONFIG,
    _write_metadata,
    run_pipeline,
)
from src.data.uploader import LocalFakeUploader, RcloneUploader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_h5(path: Path, n_samples: int = 5, n_params: int = 92) -> None:
    """Write a minimal HDF5 file that satisfies SurgeXTDataset expectations."""
    with h5py.File(path, "w") as f:
        audio = f.create_dataset(
            "audio",
            shape=(n_samples, 2, 44100 * 4),
            dtype=np.float16,
            compression=hdf5plugin.Blosc2(),
        )
        audio.attrs["velocity"] = 100
        audio.attrs["signal_duration_seconds"] = 4.0
        audio.attrs["sample_rate"] = 44100.0
        audio.attrs["channels"] = 2
        audio.attrs["min_loudness"] = -55.0

        f.create_dataset(
            "mel_spec",
            shape=(n_samples, 2, 128, 401),
            dtype=np.float32,
            compression=hdf5plugin.Blosc2(),
        )
        f.create_dataset(
            "param_array",
            shape=(n_samples, n_params),
            dtype=np.float32,
            compression=hdf5plugin.Blosc2(),
        )


def _make_fake_stats(output_dir: Path) -> None:
    """Write a minimal stats.npz next to train.h5 (SurgeXTDataset convention)."""
    np.savez(output_dir / "stats.npz", mean=np.zeros((2, 128, 401)), std=np.ones((2, 128, 401)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_subprocess(tmp_path):
    """Mock subprocess.run to create fake HDF5 outputs instead of running VST code."""

    call_log = []

    def _fake_run(cmd, check=True):
        call_log.append(cmd)
        # generate_vst_dataset.py calls: [..., data_file, num_samples, ...]
        # The data_file is at index 2 (after headless wrapper + "python" + script)
        if "generate_vst_dataset.py" in " ".join(cmd):
            # cmd pattern: [wrapper, python, script, data_file, num_samples, ...]
            # Find the .h5 argument
            h5_path = None
            for arg in cmd:
                if arg.endswith(".h5"):
                    h5_path = Path(arg)
                    break
            if h5_path is not None:
                h5_path.parent.mkdir(parents=True, exist_ok=True)
                _make_fake_h5(h5_path)
        elif "get_dataset_stats.py" in " ".join(cmd):
            # cmd: [python, script, train_h5_path]
            train_h5 = Path(cmd[-1])
            _make_fake_stats(train_h5.parent)

    with patch("scripts.run_dataset_pipeline.subprocess.run", side_effect=_fake_run):
        yield call_log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLocalFakeUploader:
    """Tests for LocalFakeUploader — local-filesystem stand-in for the R2 uploader."""

    def test_upload_copies_files(self, tmp_path):
        """Files in src directory are copied to dest_root/remote_path."""
        src = tmp_path / "source"
        src.mkdir()
        (src / "train.h5").write_bytes(b"fake-h5-data")
        (src / "metadata.json").write_text('{"key": "value"}')

        dest_root = tmp_path / "fake_r2"
        uploader = LocalFakeUploader(dest_root)
        uploader.upload(src, "runs/surge_simple/abc1234")

        dest = dest_root / "runs/surge_simple/abc1234"
        assert (dest / "train.h5").exists()
        assert (dest / "metadata.json").exists()
        assert (dest / "train.h5").read_bytes() == b"fake-h5-data"

    def test_upload_creates_nested_dirs(self, tmp_path):
        """Deeply nested remote_path directories are created automatically."""
        src = tmp_path / "source"
        src.mkdir()
        (src / "file.txt").write_text("hello")

        dest_root = tmp_path / "fake_r2"
        uploader = LocalFakeUploader(dest_root)
        uploader.upload(src, "deep/nested/path")

        assert (dest_root / "deep/nested/path/file.txt").exists()

    def test_upload_twice_overwrites(self, tmp_path):
        """Re-uploading to the same path overwrites existing files with new content."""
        src = tmp_path / "source"
        src.mkdir()
        (src / "file.txt").write_text("v1")

        dest_root = tmp_path / "fake_r2"
        uploader = LocalFakeUploader(dest_root)
        uploader.upload(src, "runs/x")

        (src / "file.txt").write_text("v2")
        uploader.upload(src, "runs/x")

        assert (dest_root / "runs/x/file.txt").read_text() == "v2"


class TestWriteMetadata:
    """Tests for _write_metadata — verifies the metadata.json schema and field values."""

    def test_metadata_fields(self, tmp_path):
        """Core fields (param_spec, git_sha, splits, r2_prefix, generated_at) are present."""
        meta_path = _write_metadata(
            output_dir=tmp_path,
            param_spec="surge_simple",
            train_samples=100,
            val_samples=10,
            test_samples=10,
            r2_prefix="runs/surge_simple/abc1234",
            git_sha="abc1234",
        )
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["param_spec"] == "surge_simple"
        assert meta["git_sha"] == "abc1234"
        assert meta["splits"] == {"train": 100, "val": 10, "test": 10}
        assert meta["r2_prefix"] == "runs/surge_simple/abc1234"
        assert "generated_at" in meta

    def test_metadata_no_upload(self, tmp_path):
        """r2_prefix is null in metadata when no upload destination is given."""
        meta_path = _write_metadata(
            output_dir=tmp_path,
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            r2_prefix=None,
            git_sha="unknown",
        )
        meta = json.loads(meta_path.read_text())
        assert meta["r2_prefix"] is None

    def test_metadata_generation_block(self, tmp_path):
        """The 'generation' block records audio rendering parameters."""
        meta_path = _write_metadata(
            output_dir=tmp_path,
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            r2_prefix=None,
            git_sha="unknown",
            sample_rate=44100.0,
            channels=2,
            velocity=100,
            signal_duration_seconds=4.0,
            min_loudness=-55.0,
            sample_batch_size=32,
        )
        meta = json.loads(meta_path.read_text())
        assert "generation" in meta
        assert meta["generation"]["sample_rate"] == 44100.0
        assert meta["generation"]["channels"] == 2

    def test_metadata_param_spec_num_params(self, tmp_path):
        """param_spec_num_params reflects the correct parameter count for each spec."""
        for spec, expected_n in [("surge_simple", 92), ("surge_xt", 189)]:
            meta_path = _write_metadata(
                output_dir=tmp_path,
                param_spec=spec,
                train_samples=5,
                val_samples=2,
                test_samples=2,
                r2_prefix=None,
                git_sha="unknown",
            )
            meta = json.loads(meta_path.read_text())
            assert meta["param_spec_num_params"] == expected_n

    def test_metadata_git_provenance_baked(self, tmp_path):
        """git_ref_source='baked' and git_dirty=False are recorded for production images."""
        meta_path = _write_metadata(
            output_dir=tmp_path,
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            r2_prefix=None,
            git_sha="prod-sha",
            git_ref_source="baked",
            git_dirty=False,
        )
        meta = json.loads(meta_path.read_text())
        assert meta["git_ref_source"] == "baked"
        assert meta["git_dirty"] is False

    def test_metadata_git_provenance_local_dirty(self, tmp_path):
        """git_ref_source='local' and git_dirty=True are recorded for dev-live runs with edits."""
        meta_path = _write_metadata(
            output_dir=tmp_path,
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            r2_prefix=None,
            git_sha="dev-sha",
            git_ref_source="local",
            git_dirty=True,
        )
        meta = json.loads(meta_path.read_text())
        assert meta["git_ref_source"] == "local"
        assert meta["git_dirty"] is True

    def test_metadata_git_provenance_unknown_defaults(self, tmp_path):
        """git_ref_source defaults to 'unknown' and git_dirty to None when not supplied."""
        meta_path = _write_metadata(
            output_dir=tmp_path,
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            r2_prefix=None,
            git_sha="unknown",
        )
        meta = json.loads(meta_path.read_text())
        assert meta["git_ref_source"] == "unknown"
        assert meta["git_dirty"] is None


class TestParamSpecValidation:
    """Tests for param_spec validation — accepted values and rejection of unknown specs."""

    def test_valid_param_specs_accepted(self, tmp_path, fake_subprocess):
        """All entries in PARAM_SPEC_TO_DATA_CONFIG run without error."""
        for spec in PARAM_SPEC_TO_DATA_CONFIG:
            output_dir = tmp_path / spec
            run_pipeline(
                param_spec=spec,
                train_samples=5,
                val_samples=2,
                test_samples=2,
                output_dir=output_dir,
                uploader=None,
            )

    def test_invalid_param_spec_exits(self, tmp_path, fake_subprocess):
        """An unrecognized param_spec causes a SystemExit."""
        with pytest.raises(SystemExit):
            run_pipeline(
                param_spec="not_a_real_spec",
                train_samples=5,
                val_samples=2,
                test_samples=2,
                output_dir=tmp_path / "out",
                uploader=None,
            )

    def test_param_spec_to_data_config_mapping(self):
        """Verify the param_spec -> Hydra data config name mapping for known specs."""
        assert PARAM_SPEC_TO_DATA_CONFIG["surge_simple"] == "surge_simple"
        assert PARAM_SPEC_TO_DATA_CONFIG["surge_xt"] == "surge"


class TestRunPipeline:
    """Integration tests for run_pipeline with mocked subprocess and LocalFakeUploader."""

    def test_generates_all_splits(self, tmp_path, fake_subprocess):
        """run_pipeline creates train.h5, val.h5, and test.h5 in the output directory."""
        output_dir = tmp_path / "dataset"
        run_pipeline(
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            output_dir=output_dir,
            uploader=None,
        )

        assert (output_dir / "train.h5").exists()
        assert (output_dir / "val.h5").exists()
        assert (output_dir / "test.h5").exists()

    def test_generates_stats_and_metadata(self, tmp_path, fake_subprocess):
        """run_pipeline produces stats.npz and metadata.json with correct fields."""
        output_dir = tmp_path / "dataset"
        run_pipeline(
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            output_dir=output_dir,
            uploader=None,
        )

        assert (output_dir / "stats.npz").exists()
        assert (output_dir / "metadata.json").exists()
        meta = json.loads((output_dir / "metadata.json").read_text())
        assert meta["param_spec"] == "surge_simple"
        assert meta["splits"]["train"] == 5

    def test_uploads_with_fake_uploader(self, tmp_path, fake_subprocess):
        """All generated files are present in the fake R2 destination after upload."""
        output_dir = tmp_path / "dataset"
        fake_r2 = tmp_path / "fake_r2"
        uploader = LocalFakeUploader(fake_r2)
        r2_prefix = "runs/surge_simple/testsha"

        run_pipeline(
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            output_dir=output_dir,
            uploader=uploader,
            r2_prefix=r2_prefix,
            git_sha="testsha",
        )

        uploaded = list((fake_r2 / r2_prefix).iterdir())
        uploaded_names = {f.name for f in uploaded}
        assert "train.h5" in uploaded_names
        assert "val.h5" in uploaded_names
        assert "test.h5" in uploaded_names
        assert "stats.npz" in uploaded_names
        assert "metadata.json" in uploaded_names

    def test_skips_upload_when_no_uploader(self, tmp_path, fake_subprocess):
        """No files are written to the fake R2 root when uploader=None."""
        output_dir = tmp_path / "dataset"
        fake_r2 = tmp_path / "fake_r2"

        run_pipeline(
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            output_dir=output_dir,
            uploader=None,
            r2_prefix="runs/surge_simple/sha",
        )

        # Fake R2 directory should not exist — nothing was uploaded
        assert not fake_r2.exists()

    def test_metadata_records_git_sha(self, tmp_path, fake_subprocess):
        """The git_sha passed to run_pipeline is recorded in metadata.json."""
        output_dir = tmp_path / "dataset"
        run_pipeline(
            param_spec="surge_simple",
            train_samples=5,
            val_samples=2,
            test_samples=2,
            output_dir=output_dir,
            uploader=None,
            git_sha="test_sha",
        )

        meta = json.loads((output_dir / "metadata.json").read_text())
        assert meta["git_sha"] == "test_sha"


class TestRcloneUploader:
    """Tests for RcloneUploader — verifies the rclone command arguments."""

    def test_upload_command_structure(self, tmp_path):
        """Rclone copy is invoked with the correct remote destination path."""
        with patch("src.data.uploader.subprocess.run") as mock_run:
            uploader = RcloneUploader(bucket="my-bucket")
            uploader.upload(tmp_path, "runs/surge_simple/your-commit-sha")
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "rclone"
            assert cmd[1] == "copy"
            assert "r2:my-bucket/runs/surge_simple/your-commit-sha" in cmd

    def test_upload_includes_checksum(self, tmp_path):
        """--checksum flag is always passed to rclone for integrity verification."""
        with patch("src.data.uploader.subprocess.run") as mock_run:
            uploader = RcloneUploader(bucket="my-bucket")
            uploader.upload(tmp_path, "runs/surge_simple/your-commit-sha")
            cmd = mock_run.call_args[0][0]
            assert "--checksum" in cmd

    def test_upload_includes_progress(self, tmp_path):
        """--progress flag is always passed to rclone for transfer visibility."""
        with patch("src.data.uploader.subprocess.run") as mock_run:
            uploader = RcloneUploader(bucket="my-bucket")
            uploader.upload(tmp_path, "runs/surge_simple/your-commit-sha")
            cmd = mock_run.call_args[0][0]
            assert "--progress" in cmd

    def test_dry_run_flag(self, tmp_path):
        """--dry-run is appended to rclone when dry_run=True."""
        with patch("src.data.uploader.subprocess.run") as mock_run:
            uploader = RcloneUploader(bucket="my-bucket", dry_run=True)
            uploader.upload(tmp_path, "runs/surge_simple/your-commit-sha")
            cmd = mock_run.call_args[0][0]
            assert "--dry-run" in cmd

    def test_no_dry_run_flag_by_default(self, tmp_path):
        """--dry-run is absent from rclone command when dry_run=False (the default)."""
        with patch("src.data.uploader.subprocess.run") as mock_run:
            uploader = RcloneUploader(bucket="my-bucket")
            uploader.upload(tmp_path, "runs/surge_simple/your-commit-sha")
            cmd = mock_run.call_args[0][0]
            assert "--dry-run" not in cmd

    def test_custom_remote_name(self, tmp_path):
        """A custom rclone_remote name is used as the destination prefix instead of 'r2'."""
        with patch("src.data.uploader.subprocess.run") as mock_run:
            uploader = RcloneUploader(bucket="my-bucket", rclone_remote="cf")
            uploader.upload(tmp_path, "runs/x")
            cmd = mock_run.call_args[0][0]
            assert "cf:my-bucket/runs/x" in cmd
