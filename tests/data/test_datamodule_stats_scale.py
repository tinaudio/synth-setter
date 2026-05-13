"""Tests for `self.scale` precomputation in audio/surge datamodules (#998).

Both datamodules previously stored ``self.std`` and divided by it inside
``__getitem__``; a ``std==0`` bin (produced by a small or silence-heavy
dataset) caused ``inf``/``nan`` outputs. The fix precomputes
``self.scale = where(std > 0, 1/std, 0)`` at load time, turning degenerate bins
into a constant-zero mask instead of a divide-by-zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.audio_datamodule import AudioFolderDataset
from synth_setter.data.surge_datamodule import SurgeXTDataset


def _write_stats_dir(dirpath: Path, mean: np.ndarray, std: np.ndarray) -> Path:
    """Persist ``stats.npz`` inside ``dirpath`` for both datamodule paths.

    ``AudioFolderDataset`` takes the stats file path directly, while
    ``SurgeXTDataset`` derives ``<dir>/stats.npz`` from the dataset file path.
    Writing into a directory lets both tests target the same fixture.

    :param dirpath: Directory to create (recursively) and write into.
    :param mean: Per-bin mean array stored under the ``mean`` key.
    :param std: Per-bin standard deviation stored under the ``std`` key.

    :returns: The path of the written ``stats.npz`` file.
    :rtype: Path
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    stats_path = dirpath / "stats.npz"
    np.savez(stats_path, mean=mean, std=std)
    return stats_path


@pytest.fixture()
def healthy_stats(tmp_path: Path) -> Path:
    """A ``stats.npz`` with all-positive ``std`` (no degenerate bins).

    :param tmp_path: Per-test temporary directory injected by pytest.

    :returns: Path to ``<tmp_path>/healthy/stats.npz``.
    :rtype: Path
    """
    mean = np.array([0.0, 1.0, 2.0, 3.0])
    std = np.array([0.5, 1.0, 2.0, 4.0])
    return _write_stats_dir(tmp_path / "healthy", mean, std)


@pytest.fixture()
def degenerate_stats(tmp_path: Path) -> Path:
    """A ``stats.npz`` containing two ``std==0`` (degenerate) bins.

    Bins 1 and 3 are degenerate; bins 0 and 2 are healthy.

    :param tmp_path: Per-test temporary directory injected by pytest.

    :returns: Path to ``<tmp_path>/degenerate/stats.npz``.
    :rtype: Path
    """
    mean = np.array([0.0, 1.0, 2.0, 3.0])
    std = np.array([0.5, 0.0, 2.0, 0.0])
    return _write_stats_dir(tmp_path / "degenerate", mean, std)


class TestAudioFolderDataset:
    """Pin ``AudioFolderDataset``'s ``scale`` precomputation contract."""

    def test_no_stats_file_leaves_mean_and_scale_none(self, tmp_path: Path) -> None:
        """No ``reference_stats_file`` → both ``mean`` and ``scale`` stay ``None``.

        :param tmp_path: Per-test temporary directory injected by pytest.
        """
        dataset = AudioFolderDataset(root=str(tmp_path))

        assert dataset.mean is None
        assert dataset.scale is None

    def test_healthy_stats_sets_scale_to_reciprocal_of_std(
        self, tmp_path: Path, healthy_stats: Path
    ) -> None:
        """For all-positive ``std``, ``scale`` is element-wise ``1/std``.

        :param tmp_path: Per-test temporary directory injected by pytest.
        :param healthy_stats: Path to a healthy ``stats.npz`` fixture.
        """
        dataset = AudioFolderDataset(
            root=str(tmp_path), reference_stats_file=str(healthy_stats)
        )

        np.testing.assert_allclose(dataset.scale, np.array([2.0, 1.0, 0.5, 0.25]))

    def test_degenerate_stats_sets_scale_to_zero_for_zero_std_bins(
        self, tmp_path: Path, degenerate_stats: Path
    ) -> None:
        """``std==0`` bins map to ``scale==0`` (not inf); healthy bins are unchanged.

        :param tmp_path: Per-test temporary directory injected by pytest.
        :param degenerate_stats: Path to a degenerate ``stats.npz`` fixture.
        """
        dataset = AudioFolderDataset(
            root=str(tmp_path), reference_stats_file=str(degenerate_stats)
        )

        np.testing.assert_allclose(dataset.scale, np.array([2.0, 0.0, 0.5, 0.0]))
        assert np.isfinite(dataset.scale).all()

    def test_degenerate_stats_zeroes_masked_bins_in_normalized_output(
        self, tmp_path: Path, degenerate_stats: Path
    ) -> None:
        """``(spec - mean) * scale`` is finite everywhere and zero at masked bins.

        :param tmp_path: Per-test temporary directory injected by pytest.
        :param degenerate_stats: Path to a degenerate ``stats.npz`` fixture.
        """
        dataset = AudioFolderDataset(
            root=str(tmp_path), reference_stats_file=str(degenerate_stats)
        )

        spec = np.array([10.0, 999.0, 4.0, 999.0])
        normalized = (spec - dataset.mean) * dataset.scale

        assert np.isfinite(normalized).all()
        assert normalized[1] == 0.0
        assert normalized[3] == 0.0
        np.testing.assert_allclose(normalized[0], (10.0 - 0.0) * 2.0)
        np.testing.assert_allclose(normalized[2], (4.0 - 2.0) * 0.5)


class TestSurgeXTDataset:
    """Pin ``SurgeXTDataset``'s ``scale`` precomputation contract."""

    def test_no_stats_file_raises(self, tmp_path: Path) -> None:
        """A missing ``stats.npz`` next to the dataset file surfaces as FileNotFoundError.

        :param tmp_path: Per-test temporary directory injected by pytest.
        """
        ds = SurgeXTDataset(dataset_file=tmp_path / "nope.h5", batch_size=1, fake=True)

        with pytest.raises(FileNotFoundError, match="statistics file"):
            ds._load_dataset_statistics(tmp_path / "nope.h5")

    def test_healthy_stats_sets_scale_to_reciprocal_of_std(
        self, tmp_path: Path, healthy_stats: Path
    ) -> None:
        """For all-positive ``std``, ``scale`` is element-wise ``1/std``.

        :param tmp_path: Per-test temporary directory injected by pytest.
        :param healthy_stats: Path to a healthy ``stats.npz`` fixture.
        """
        ds = SurgeXTDataset(dataset_file=tmp_path / "train.h5", batch_size=1, fake=True)

        ds._load_dataset_statistics(healthy_stats.parent / "train.h5")

        np.testing.assert_allclose(ds.scale, np.array([2.0, 1.0, 0.5, 0.25]))

    def test_degenerate_stats_sets_scale_to_zero_for_zero_std_bins(
        self, tmp_path: Path, degenerate_stats: Path
    ) -> None:
        """``std==0`` bins map to ``scale==0`` (not inf); healthy bins are unchanged.

        :param tmp_path: Per-test temporary directory injected by pytest.
        :param degenerate_stats: Path to a degenerate ``stats.npz`` fixture.
        """
        ds = SurgeXTDataset(dataset_file=tmp_path / "train.h5", batch_size=1, fake=True)

        ds._load_dataset_statistics(degenerate_stats.parent / "train.h5")

        np.testing.assert_allclose(ds.scale, np.array([2.0, 0.0, 0.5, 0.0]))
        assert np.isfinite(ds.scale).all()

    def test_degenerate_stats_zeroes_masked_bins_in_normalized_output(
        self, tmp_path: Path, degenerate_stats: Path
    ) -> None:
        """``(mel_spec - mean) * scale`` is finite everywhere and zero at masked bins.

        :param tmp_path: Per-test temporary directory injected by pytest.
        :param degenerate_stats: Path to a degenerate ``stats.npz`` fixture.
        """
        ds = SurgeXTDataset(dataset_file=tmp_path / "train.h5", batch_size=1, fake=True)

        ds._load_dataset_statistics(degenerate_stats.parent / "train.h5")

        mel_spec = np.array([10.0, 999.0, 4.0, 999.0])
        normalized = (mel_spec - ds.mean) * ds.scale

        assert np.isfinite(normalized).all()
        assert normalized[1] == 0.0
        assert normalized[3] == 0.0
