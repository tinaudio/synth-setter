"""Tests for pipeline.stats module."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pytest

from pipeline.stats import DatasetStats, compute_stats_from_hdf5, load_stats, save_stats


def _make_hdf5(path: Path, data: np.ndarray, dataset_name: str = "mel_spec") -> Path:
    """Create an HDF5 file with a single dataset."""
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=data)
    return path


class TestComputeStats:
    """Tests for compute_stats_from_hdf5."""

    def test_compute_stats_single_file_correct_mean(self, tmp_path: Path) -> None:
        """All-ones input yields all-ones mean."""
        data = np.ones((10, 2, 3), dtype=np.float32)
        hdf5_path = _make_hdf5(tmp_path / "ones.h5", data)

        stats = compute_stats_from_hdf5([hdf5_path])

        np.testing.assert_allclose(stats.mean, np.ones((2, 3)))

    def test_compute_stats_single_file_correct_std(self, tmp_path: Path) -> None:
        """Random data yields std matching numpy reference."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4, 5)).astype(np.float32)
        hdf5_path = _make_hdf5(tmp_path / "randn.h5", data)

        stats = compute_stats_from_hdf5([hdf5_path])

        expected_mean = data.mean(axis=0)
        expected_std = data.std(axis=0)
        np.testing.assert_allclose(stats.mean, expected_mean, atol=1e-5)
        np.testing.assert_allclose(stats.std, expected_std, atol=1e-5)

    def test_compute_stats_multiple_files_aggregated(self, tmp_path: Path) -> None:
        """Zeros and twos across two files yield mean=1 and std=1."""
        zeros = np.zeros((5, 2, 3), dtype=np.float32)
        twos = np.full((5, 2, 3), 2.0, dtype=np.float32)
        path1 = _make_hdf5(tmp_path / "zeros.h5", zeros)
        path2 = _make_hdf5(tmp_path / "twos.h5", twos)

        stats = compute_stats_from_hdf5([path1, path2])

        np.testing.assert_allclose(stats.mean, np.ones((2, 3)))
        np.testing.assert_allclose(stats.std, np.ones((2, 3)))

    def test_compute_stats_empty_paths_raises_value_error(self) -> None:
        """Empty path list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_stats_from_hdf5([])

    def test_compute_stats_missing_dataset_raises_key_error(self, tmp_path: Path) -> None:
        """Missing dataset name raises KeyError."""
        data = np.ones((5, 2, 3), dtype=np.float32)
        hdf5_path = _make_hdf5(tmp_path / "audio.h5", data, dataset_name="audio")

        with pytest.raises(KeyError, match="mel_spec"):
            compute_stats_from_hdf5([hdf5_path])

    def test_compute_stats_custom_dataset_name(self, tmp_path: Path) -> None:
        """Custom dataset_name reads from the correct dataset."""
        data = np.ones((5, 2, 3), dtype=np.float32) * 3.0
        hdf5_path = _make_hdf5(tmp_path / "custom.h5", data, dataset_name="audio")

        stats = compute_stats_from_hdf5([hdf5_path], dataset_name="audio")

        np.testing.assert_allclose(stats.mean, np.full((2, 3), 3.0))

    def test_compute_stats_numerically_stable(self, tmp_path: Path) -> None:
        """Mix of 1e6 and 1e-6 values produces no NaN or Inf."""
        large = np.full((5, 2, 3), 1e6, dtype=np.float64)
        small = np.full((5, 2, 3), 1e-6, dtype=np.float64)
        path1 = _make_hdf5(tmp_path / "large.h5", large)
        path2 = _make_hdf5(tmp_path / "small.h5", small)

        stats = compute_stats_from_hdf5([path1, path2])

        assert not np.any(np.isnan(stats.mean))
        assert not np.any(np.isnan(stats.std))
        assert not np.any(np.isinf(stats.mean))
        assert not np.any(np.isinf(stats.std))

    def test_compute_stats_frozen_dataclass(self, tmp_path: Path) -> None:
        """DatasetStats is immutable (frozen dataclass)."""
        data = np.ones((5, 2, 3), dtype=np.float32)
        hdf5_path = _make_hdf5(tmp_path / "frozen.h5", data)

        stats = compute_stats_from_hdf5([hdf5_path])

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            stats.mean = np.zeros((2, 3))  # type: ignore[misc]


class TestSaveLoadStats:
    """Tests for save_stats and load_stats."""

    def test_save_stats_creates_npz(self, tmp_path: Path) -> None:
        """save_stats creates a file at the given path."""
        stats = DatasetStats(mean=np.array([1.0, 2.0]), std=np.array([0.5, 0.3]))
        output_path = tmp_path / "stats.npz"

        result = save_stats(stats, output_path)

        assert result.exists()
        assert result == output_path

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Save then load preserves mean and std."""
        original = DatasetStats(
            mean=np.array([1.0, 2.0, 3.0]),
            std=np.array([0.1, 0.2, 0.3]),
        )
        output_path = tmp_path / "stats.npz"

        save_stats(original, output_path)
        loaded = load_stats(output_path)

        np.testing.assert_allclose(loaded.mean, original.mean)
        np.testing.assert_allclose(loaded.std, original.std)

    def test_load_stats_missing_file_raises(self, tmp_path: Path) -> None:
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_stats(tmp_path / "nonexistent.npz")

    def test_load_stats_missing_keys_raises(self, tmp_path: Path) -> None:
        """Npz without 'mean' key raises KeyError."""
        bad_path = tmp_path / "bad_stats.npz"
        np.savez(bad_path, something_else=np.array([1.0]))

        with pytest.raises(KeyError, match="mean"):
            load_stats(bad_path)
