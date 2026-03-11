"""Unit tests for scripts/finalize_shards.py.

Tests verify:
  1. finalize_shards() — downloads shards, reshards, computes stats, uploads
  2. _rclone_download() — rclone command construction
  3. CLI wiring

To run:
    pytest tests/test_finalize_shards.py -v
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.finalize_shards import _rclone_download, finalize_shards
from src.data.uploader import LocalFakeUploader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUDIO_TAIL = (2, 1000)
_MEL_TAIL = (2, 16, 41)
_PARAM_WIDTH = 8


def _create_test_shards(
    directory: Path,
    num_shards: int,
    samples_per_shard: int = 10,
) -> list[Path]:
    """Create shard-*.h5 files with deterministic data.

    Returns shard paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    offset = 0
    for i in range(num_shards):
        n = samples_per_shard
        audio = np.full((n, *_AUDIO_TAIL), fill_value=offset, dtype=np.float32)
        mel = np.full((n, *_MEL_TAIL), fill_value=offset + 0.5, dtype=np.float32)
        param = np.full((n, _PARAM_WIDTH), fill_value=offset + 0.25, dtype=np.float32)
        for j in range(n):
            audio[j, 0, 0] = offset + j
            mel[j, 0, 0, 0] = offset + j
            param[j, 0] = offset + j

        path = directory / f"shard-{i:04d}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("audio", data=audio)
            f.create_dataset("mel_spec", data=mel)
            f.create_dataset("param_array", data=param)
        paths.append(path)
        offset += n
    return paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def shard_source(tmp_path):
    """Pre-populated shard source directory with 5 shards."""
    source = tmp_path / "r2_shards"
    _create_test_shards(source, num_shards=5, samples_per_shard=10)
    return source


@pytest.fixture()
def make_download_fn(shard_source):
    """Returns a fake download_fn that copies from shard_source to local_dir."""

    def _download_fn(remote_path: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for f in shard_source.iterdir():
            if f.is_file():
                shutil.copy2(f, local_dir / f.name)

    return _download_fn


@pytest.fixture()
def noop_download_fn():
    """A download_fn that does nothing (simulates empty R2 path)."""

    def _download_fn(remote_path: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)

    return _download_fn


@pytest.fixture()
def fake_stats_subprocess():
    """Mock subprocess.run to create fake stats.npz instead of running dask."""
    call_log = []

    def _fake_run(cmd, check=True):
        call_log.append(cmd)
        if "get_dataset_stats.py" in " ".join(cmd):
            train_h5 = Path(cmd[-1])
            np.savez(
                train_h5.parent / "stats.npz",
                mean=np.zeros(_MEL_TAIL),
                std=np.ones(_MEL_TAIL),
            )

    with patch("scripts.finalize_shards.subprocess.run", side_effect=_fake_run):
        yield call_log


# ---------------------------------------------------------------------------
# Tests for finalize_shards()
# ---------------------------------------------------------------------------


class TestFinalizeShards:
    """Tests for finalize_shards() — the core download-reshard-stats-upload flow."""

    def test_creates_train_val_test_splits(
        self, tmp_path, make_download_fn, fake_stats_subprocess
    ):
        """5 shards with val=1, test=1 produces train.h5, val.h5, test.h5."""
        output_dir = tmp_path / "dataset"
        finalize_shards(
            output_dir=output_dir,
            download_fn=make_download_fn,
            r2_prefix="runs/batch42",
            val_shards=1,
            test_shards=1,
        )
        assert (output_dir / "train.h5").exists()
        assert (output_dir / "val.h5").exists()
        assert (output_dir / "test.h5").exists()

    def test_split_sample_counts(self, tmp_path, make_download_fn, fake_stats_subprocess):
        """Train gets 3 shards (30 samples), val and test each get 1 shard (10)."""
        output_dir = tmp_path / "dataset"
        finalize_shards(
            output_dir=output_dir,
            download_fn=make_download_fn,
            r2_prefix="runs/batch42",
            val_shards=1,
            test_shards=1,
        )
        # 5 shards × 10 samples each: train=3 shards (30), val=1 (10), test=1 (10)
        with h5py.File(output_dir / "train.h5", "r") as f:
            assert f["audio"].shape[0] == 30
        with h5py.File(output_dir / "val.h5", "r") as f:
            assert f["audio"].shape[0] == 10
        with h5py.File(output_dir / "test.h5", "r") as f:
            assert f["audio"].shape[0] == 10

    def test_computes_stats(self, tmp_path, make_download_fn, fake_stats_subprocess):
        """stats.npz is created in output_dir after pipeline runs."""
        output_dir = tmp_path / "dataset"
        finalize_shards(
            output_dir=output_dir,
            download_fn=make_download_fn,
            r2_prefix="runs/batch42",
            val_shards=1,
            test_shards=1,
        )
        assert (output_dir / "stats.npz").exists()

    def test_stats_called_with_train_h5(self, tmp_path, make_download_fn, fake_stats_subprocess):
        """get_dataset_stats.py is invoked with the correct train.h5 path."""
        output_dir = tmp_path / "dataset"
        finalize_shards(
            output_dir=output_dir,
            download_fn=make_download_fn,
            r2_prefix="runs/batch42",
            val_shards=1,
            test_shards=1,
        )
        stats_calls = [c for c in fake_stats_subprocess if "get_dataset_stats.py" in " ".join(c)]
        assert len(stats_calls) == 1
        assert stats_calls[0][-1] == str(output_dir / "train.h5")

    def test_uploads_results(self, tmp_path, make_download_fn, fake_stats_subprocess):
        """Fake R2 destination contains train.h5, val.h5, test.h5, stats.npz."""
        output_dir = tmp_path / "dataset"
        fake_r2 = tmp_path / "fake_r2"
        uploader = LocalFakeUploader(fake_r2)
        finalize_shards(
            output_dir=output_dir,
            download_fn=make_download_fn,
            r2_prefix="runs/batch42",
            val_shards=1,
            test_shards=1,
            uploader=uploader,
        )
        dest = fake_r2 / "runs/batch42"
        uploaded_names = {f.name for f in dest.iterdir()}
        assert "train.h5" in uploaded_names
        assert "val.h5" in uploaded_names
        assert "test.h5" in uploaded_names
        assert "stats.npz" in uploaded_names

    def test_skips_upload_when_no_uploader(
        self, tmp_path, make_download_fn, fake_stats_subprocess
    ):
        """No files written to fake R2 root when uploader=None."""
        output_dir = tmp_path / "dataset"
        fake_r2 = tmp_path / "fake_r2"
        finalize_shards(
            output_dir=output_dir,
            download_fn=make_download_fn,
            r2_prefix="runs/batch42",
            val_shards=1,
            test_shards=1,
            uploader=None,
        )
        assert not fake_r2.exists()

    def test_no_shards_exits(self, tmp_path, noop_download_fn, fake_stats_subprocess):
        """Empty shard directory causes SystemExit."""
        output_dir = tmp_path / "dataset"
        with pytest.raises(SystemExit):
            finalize_shards(
                output_dir=output_dir,
                download_fn=noop_download_fn,
                r2_prefix="runs/batch42",
                val_shards=1,
                test_shards=1,
            )

    def test_not_enough_shards_exits(self, tmp_path, fake_stats_subprocess):
        """2 shards but val=1, test=2 → not enough for train."""
        source = tmp_path / "small_source"
        _create_test_shards(source, num_shards=2, samples_per_shard=10)

        def _download_fn(remote_path: str, local_dir: Path) -> None:
            local_dir.mkdir(parents=True, exist_ok=True)
            for f in source.iterdir():
                if f.is_file():
                    shutil.copy2(f, local_dir / f.name)

        output_dir = tmp_path / "dataset"
        with pytest.raises(SystemExit):
            finalize_shards(
                output_dir=output_dir,
                download_fn=_download_fn,
                r2_prefix="runs/batch42",
                val_shards=1,
                test_shards=2,
            )


# ---------------------------------------------------------------------------
# Tests for _rclone_download()
# ---------------------------------------------------------------------------


class TestRcloneDownload:
    """Tests for _rclone_download() — rclone command construction."""

    def test_download_command_structure(self, tmp_path):
        """Rclone copy is invoked with correct source, dest, and --checksum."""
        with patch("scripts.finalize_shards.subprocess.run") as mock_run:
            _rclone_download("runs/batch42/shards", tmp_path / "shards", "my-bucket")
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "rclone"
            assert cmd[1] == "copy"
            assert "r2:my-bucket/runs/batch42/shards" in cmd
            assert str(tmp_path / "shards") in cmd
            assert "--checksum" in cmd

    def test_custom_remote_name(self, tmp_path):
        """Custom rclone_remote name appears in the command."""
        with patch("scripts.finalize_shards.subprocess.run") as mock_run:
            _rclone_download(
                "runs/batch42/shards", tmp_path / "shards", "my-bucket", rclone_remote="cf"
            )
            cmd = mock_run.call_args[0][0]
            assert "cf:my-bucket/runs/batch42/shards" in cmd
