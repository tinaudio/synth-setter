import argparse
import logging
import os

import dask.array as da
import h5py
import numpy as np
import rootutils
from dask.distributed import Client, progress

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from synth_setter.data.audio_datamodule import AudioFolderDataset
from synth_setter.data.surge_datamodule import SurgeXTDataset

logger = logging.getLogger(__name__)


def _check_degenerate_bins(std: np.ndarray, mask_degenerate: bool) -> None:
    """Raise on zero-variance bins unless masking is enabled.

    A zero entry in ``std`` means the corresponding mel bin was constant across
    the entire dataset, which propagates as inf/nan through downstream
    ``(x - mean) / std`` normalization. By default this aborts so the upstream
    cause (silence-dominated data, mel filterbank above Nyquist, dataset too
    small) is surfaced; with ``mask_degenerate=True`` the caller opts in to
    persisting ``std=0`` and letting the datamodule treat it as a mask.

    :param std: Per-bin standard deviation array. Zero entries indicate
        constant bins.
    :param mask_degenerate: If True, log a warning and return. If False, raise.

    :raises ValueError: When ``mask_degenerate`` is False and any entry of
        ``std`` is zero. The message lists the degenerate bin indices.
    """
    degenerate = np.where(std == 0)[0]
    if degenerate.size == 0:
        return
    if not mask_degenerate:
        raise ValueError(
            f"Found {degenerate.size} mel bin(s) with zero variance across the "
            f"dataset: indices {degenerate.tolist()}. This usually indicates an "
            f"upstream problem (silence-dominated data, mel filterbank above "
            f"Nyquist, or a dataset too small to vary these bins). Rerun with "
            f"--mask-degenerate-bins to mask these bins instead of failing."
        )
    logger.warning(
        "Masking %d degenerate mel bin(s) with zero variance: %s",
        degenerate.size,
        degenerate.tolist(),
    )


def get_stats_hdf5(filename, mask_degenerate: bool = False):
    dataset_name = "mel_spec"

    num_workers = 4

    print("Starting client...")
    client = Client(n_workers=num_workers, threads_per_worker=8)
    # Create a dask array that references the HDF5 dataset
    # "chunks=" controls the chunk size in memory
    print("Creating dask array...")
    darray = da.from_array(
        h5py.File(filename, "r")[dataset_name],
        chunks="auto",  # You can tune this chunk size
    )

    print("Computing mean and std...")
    mean_task = darray.mean(axis=0)
    std_task = darray.std(axis=0)

    print("Persisting tasks...")
    futures = [mean_task.persist(), std_task.persist()]

    print("Displaying progress...")
    progress(futures)

    print("Gathering results...")
    mean_val, std_val = client.gather(futures)

    print("Mean:", mean_val)
    print("std:", std_val)

    mean = mean_val.compute()
    std = std_val.compute()

    _check_degenerate_bins(std, mask_degenerate)

    print("Saving to file...")
    out_file = SurgeXTDataset.get_stats_file_path(filename)
    np.savez(out_file, mean=mean, std=std)


def update(existing, new):
    count, mean, M2 = existing
    count += 1
    delta = new - mean
    mean += delta / count
    delta2 = new - mean
    M2 += delta * delta2
    return count, mean, M2


def finalize(existing, mask_degenerate: bool = False):
    count, mean, M2 = existing
    variance = M2 / count if count > 1 else 0
    std = np.sqrt(variance)
    _check_degenerate_bins(std, mask_degenerate)
    return mean, std


def get_stats_directory(directory, mask_degenerate: bool = False):
    dataset = AudioFolderDataset(directory)
    out_file = AudioFolderDataset.get_stats_file_path(directory)

    existing = (0, 0, 0)
    # we run Welford's online algorithm
    for i in range(len(dataset)):
        x = dataset[i]["mel_spec"]
        existing = update(existing, x)

        if i % 10 == 0:
            logger.info("Processed %d files...", i + 1)

    mean, std = finalize(existing, mask_degenerate=mask_degenerate)

    logger.info("Saving to %s", str(out_file))

    np.savez(out_file, mean=mean, std=std)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean/std statistics over a Surge XT HDF5 file or an audio "
            "folder and write the result to a sibling stats.npz."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Path to a .h5 file (Dask path) or a directory of audio files "
            "(streaming Welford path)."
        ),
    )
    parser.add_argument(
        "--mask-degenerate-bins",
        action="store_true",
        help=(
            "If set, mel bins with zero variance across the dataset are masked "
            "(their normalization scale is set to 0 by downstream consumers) "
            "rather than raising an error. Default is to raise so degenerate "
            "bins are surfaced explicitly."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if os.path.splitext(args.input)[-1] == ".h5":
        get_stats_hdf5(args.input, mask_degenerate=args.mask_degenerate_bins)
    else:
        get_stats_directory(args.input, mask_degenerate=args.mask_degenerate_bins)
