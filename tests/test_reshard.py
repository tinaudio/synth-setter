"""Unit tests for scripts/reshard_data_dynamic_shard.py.

Tests verify behavior of:
  1. reshard_split() — creates virtual HDF5 datasets from shard files
  2. CLI split assignment — distributes shards into train/val/test splits

To run:
    pytest tests/test_reshard.py -v
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.reshard_data_dynamic_shard import reshard_split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use small shapes to keep tests fast. The exact dimensions don't matter —
# what matters is that reshard_split correctly concatenates along axis 0.
_AUDIO_TAIL = (2, 1000)  # (channels, samples) — tiny audio
_MEL_TAIL = (2, 16, 41)  # (channels, mels, frames) — tiny mel
_PARAM_WIDTH = 8  # number of synth parameters


def _create_test_shards(
    directory: Path,
    num_shards: int,
    samples_per_shard: int | list[int],
) -> tuple[list[Path], dict[str, np.ndarray]]:
    """Create shard-*.h5 files with deterministic data.

    Args:
        directory: Where to write shard files.
        num_shards: Number of shards to create.
        samples_per_shard: Samples per shard — int for uniform, list for varying.

    Returns:
        (shard_paths, expected) where expected maps dataset name to the
        concatenated array across all shards in order.
    """
    if isinstance(samples_per_shard, int):
        sizes = [samples_per_shard] * num_shards
    else:
        assert len(samples_per_shard) == num_shards
        sizes = samples_per_shard

    all_audio, all_mel, all_param = [], [], []
    paths = []

    offset = 0
    for i, n in enumerate(sizes):
        # Deterministic data: each sample's first element encodes its global index.
        audio = np.full((n, *_AUDIO_TAIL), fill_value=offset, dtype=np.float32)
        mel = np.full((n, *_MEL_TAIL), fill_value=offset + 0.5, dtype=np.float32)
        param = np.full((n, _PARAM_WIDTH), fill_value=offset + 0.25, dtype=np.float32)

        # Make each sample uniquely identifiable by its global index.
        for j in range(n):
            audio[j, 0, 0] = offset + j
            mel[j, 0, 0, 0] = offset + j
            param[j, 0] = offset + j

        shard_name = f"shard-{i:04d}.h5"  # noqa: E231
        path = directory / shard_name
        with h5py.File(path, "w") as f:
            f.create_dataset("audio", data=audio)
            f.create_dataset("mel_spec", data=mel)
            f.create_dataset("param_array", data=param)

        paths.append(path)
        all_audio.append(audio)
        all_mel.append(mel)
        all_param.append(param)
        offset += n

    expected = {
        "audio": np.concatenate(all_audio, axis=0),
        "mel_spec": np.concatenate(all_mel, axis=0),
        "param_array": np.concatenate(all_param, axis=0),
    }
    return paths, expected


# ---------------------------------------------------------------------------
# Tests for reshard_split()
# ---------------------------------------------------------------------------


class TestReshardSplit:
    """Tests for reshard_split — the core virtual-dataset builder."""

    def test_single_shard(self, tmp_path):
        """A single shard produces a virtual dataset matching the original."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        paths, expected = _create_test_shards(shard_dir, num_shards=1, samples_per_shard=10)

        out = tmp_path / "out.h5"
        reshard_split(paths, out)

        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f["audio"][:], expected["audio"])
            np.testing.assert_array_equal(f["mel_spec"][:], expected["mel_spec"])
            np.testing.assert_array_equal(f["param_array"][:], expected["param_array"])

    def test_multiple_shards_same_size(self, tmp_path):
        """Three equal-sized shards concatenate correctly with right boundary data."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        paths, expected = _create_test_shards(shard_dir, num_shards=3, samples_per_shard=100)

        out = tmp_path / "out.h5"
        reshard_split(paths, out)

        with h5py.File(out, "r") as f:
            assert f["audio"].shape[0] == 300
            # Verify boundary samples (last of shard 0, first of shard 1)
            assert f["audio"][99, 0, 0] == 99
            assert f["audio"][100, 0, 0] == 100
            # Verify last sample
            assert f["audio"][299, 0, 0] == 299
            # Full array equality
            np.testing.assert_array_equal(f["param_array"][:], expected["param_array"])

    def test_multiple_shards_varying_size(self, tmp_path):
        """Shards of 50, 100, 75 samples produce correct total and offsets."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        paths, expected = _create_test_shards(
            shard_dir, num_shards=3, samples_per_shard=[50, 100, 75]
        )

        out = tmp_path / "out.h5"
        reshard_split(paths, out)

        with h5py.File(out, "r") as f:
            assert f["audio"].shape[0] == 225
            # Boundary: end of shard 0 / start of shard 1
            assert f["audio"][49, 0, 0] == 49
            assert f["audio"][50, 0, 0] == 50
            # Boundary: end of shard 1 / start of shard 2
            assert f["audio"][149, 0, 0] == 149
            assert f["audio"][150, 0, 0] == 150
            np.testing.assert_array_equal(f["mel_spec"][:], expected["mel_spec"])

    def test_empty_shard_list_raises(self, tmp_path):
        """An empty shard list raises ValueError."""
        out = tmp_path / "out.h5"
        with pytest.raises(ValueError, match="[Nn]o shard"):
            reshard_split([], out)

    def test_output_is_virtual_dataset(self, tmp_path):
        """Output datasets are HDF5 virtual datasets, not copies."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        paths, _ = _create_test_shards(shard_dir, num_shards=2, samples_per_shard=5)

        out = tmp_path / "out.h5"
        reshard_split(paths, out)

        with h5py.File(out, "r") as f:
            assert f["audio"].is_virtual
            assert f["mel_spec"].is_virtual
            assert f["param_array"].is_virtual

    def test_uses_relative_paths(self, tmp_path):
        """Virtual sources use relative paths so the dataset is portable."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        paths, expected = _create_test_shards(shard_dir, num_shards=2, samples_per_shard=5)

        out = tmp_path / "out.h5"
        reshard_split(paths, out)

        # Move the entire directory to a new location and verify it still works.
        new_root = tmp_path / "moved"
        new_root.mkdir()
        import shutil

        shutil.copytree(shard_dir, new_root / "shards")
        shutil.copy2(out, new_root / "out.h5")

        with h5py.File(new_root / "out.h5", "r") as f:
            np.testing.assert_array_equal(f["audio"][:], expected["audio"])


# ---------------------------------------------------------------------------
# Tests for CLI / split assignment
# ---------------------------------------------------------------------------


class TestReshardCLI:
    """Tests for the CLI wrapper — split assignment and shard counting."""

    def _setup_shards_dir(self, tmp_path, num_shards, samples_per_shard=10):
        """Create dataset_root/shards/ with test shards.

        Returns dataset_root.
        """
        dataset_root = tmp_path / "dataset"
        shard_dir = dataset_root / "shards"
        shard_dir.mkdir(parents=True)
        _create_test_shards(shard_dir, num_shards, samples_per_shard)
        return dataset_root

    def test_remainder_train_shards(self, tmp_path):
        """With 12 shards and --val-shards 1 --test-shards 1, train gets 10."""
        from click.testing import CliRunner

        from scripts.reshard_data_dynamic_shard import main

        dataset_root = self._setup_shards_dir(tmp_path, num_shards=12)
        runner = CliRunner()
        result = runner.invoke(
            main, [str(dataset_root), "--val-shards", "1", "--test-shards", "1"]
        )

        assert result.exit_code == 0, result.output
        with h5py.File(dataset_root / "train.h5", "r") as f:
            assert f["audio"].shape[0] == 10 * 10  # 10 shards × 10 samples each
        with h5py.File(dataset_root / "val.h5", "r") as f:
            assert f["audio"].shape[0] == 1 * 10
        with h5py.File(dataset_root / "test.h5", "r") as f:
            assert f["audio"].shape[0] == 1 * 10

    def test_explicit_train_shards(self, tmp_path):
        """Explicit --train-shards 8 --val-shards 2 --test-shards 2 with 12 shards."""
        from click.testing import CliRunner

        from scripts.reshard_data_dynamic_shard import main

        dataset_root = self._setup_shards_dir(tmp_path, num_shards=12)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(dataset_root),
                "--train-shards",
                "8",
                "--val-shards",
                "2",
                "--test-shards",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        with h5py.File(dataset_root / "train.h5", "r") as f:
            assert f["audio"].shape[0] == 8 * 10
        with h5py.File(dataset_root / "val.h5", "r") as f:
            assert f["audio"].shape[0] == 2 * 10
        with h5py.File(dataset_root / "test.h5", "r") as f:
            assert f["audio"].shape[0] == 2 * 10

    def test_shard_count_mismatch_exits(self, tmp_path):
        """When explicit shard counts exceed total shards, CLI exits with error."""
        from click.testing import CliRunner

        from scripts.reshard_data_dynamic_shard import main

        dataset_root = self._setup_shards_dir(tmp_path, num_shards=12)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(dataset_root),
                "--train-shards",
                "5",
                "--val-shards",
                "5",
                "--test-shards",
                "5",
            ],
        )

        assert result.exit_code != 0

    def test_reads_from_shards_subdir(self, tmp_path):
        """Shards are read from dataset_root/shards/, outputs go to dataset_root/."""
        from click.testing import CliRunner

        from scripts.reshard_data_dynamic_shard import main

        dataset_root = self._setup_shards_dir(tmp_path, num_shards=3)
        runner = CliRunner()
        result = runner.invoke(
            main, [str(dataset_root), "--val-shards", "1", "--test-shards", "1"]
        )

        assert result.exit_code == 0, result.output
        assert (dataset_root / "train.h5").exists()
        assert (dataset_root / "val.h5").exists()
        assert (dataset_root / "test.h5").exists()
        # Shards should still be in the shards/ subdir
        assert list((dataset_root / "shards").glob("shard-*.h5"))
