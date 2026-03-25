from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Normalization statistics for a dataset."""

    mean: npt.NDArray[np.floating]
    std: npt.NDArray[np.floating]


def compute_stats_from_hdf5(
    hdf5_paths: list[Path],
    dataset_name: str = "mel_spec",
) -> DatasetStats:
    """Compute mean and std across samples in one or more HDF5 files.

    Uses Welford's online algorithm for numerical stability and O(1) memory. Iterates sample-by-
    sample across all files.

    :param hdf5_paths: Paths to HDF5 files containing the dataset.
    :param dataset_name: Name of the dataset inside each HDF5 file.
    :returns: DatasetStats with mean and std arrays.
    :raises ValueError: If hdf5_paths is empty.
    :raises KeyError: If dataset_name not found in an HDF5 file.
    """
    if not hdf5_paths:
        raise ValueError("hdf5_paths must not be empty")

    count = 0
    mean: npt.NDArray[np.float64] | None = None
    m2: npt.NDArray[np.float64] | None = None

    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            if dataset_name not in f:
                raise KeyError(
                    f"Dataset '{dataset_name}' not found in {path}. Available: {list(f.keys())}"
                )
            ds = f[dataset_name]
            if not isinstance(ds, h5py.Dataset):
                raise TypeError(f"Expected h5py.Dataset, got {type(ds)} for '{dataset_name}'")
            for i in range(ds.shape[0]):
                sample = ds[i].astype(np.float64)
                count += 1
                if mean is None or m2 is None:
                    mean = np.zeros_like(sample, dtype=np.float64)
                    m2 = np.zeros_like(sample, dtype=np.float64)
                delta = sample - mean
                mean += delta / count
                delta2 = sample - mean
                m2 += delta * delta2

    if count == 0 or mean is None or m2 is None:
        raise ValueError("No samples found in provided HDF5 files")

    variance = m2 / count
    std = np.sqrt(variance)
    return DatasetStats(mean=mean, std=std)


def save_stats(stats: DatasetStats, output_path: Path) -> Path:
    """Save DatasetStats to .npz with 'mean' and 'std' keys.

    :param stats: The statistics to save.
    :param output_path: Destination file path.
    :returns: The output path.
    """
    np.savez(output_path, mean=stats.mean, std=stats.std)
    return output_path


def load_stats(stats_path: Path) -> DatasetStats:
    """Load DatasetStats from .npz file.

    :param stats_path: Path to the .npz file.
    :returns: Loaded DatasetStats.
    :raises FileNotFoundError: If stats_path does not exist.
    :raises KeyError: If 'mean' or 'std' keys are missing.
    """
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    data = np.load(stats_path)
    if "mean" not in data:
        raise KeyError("'mean' key not found in stats file")
    if "std" not in data:
        raise KeyError("'std' key not found in stats file")
    return DatasetStats(mean=data["mean"], std=data["std"])
