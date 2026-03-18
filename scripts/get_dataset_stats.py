import os
import sys

import dask.array as da
import h5py
import numpy as np
import rootutils
from dask.distributed import Client, progress
from loguru import logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.audio_datamodule import AudioFolderDataset
from src.data.surge_datamodule import SurgeXTDataset


def get_stats_hdf5(filename):
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

    print("Saving to file...")
    out_file = SurgeXTDataset.get_stats_file_path(filename)
    mean = mean_val.compute()
    std = std_val.compute()
    np.savez(out_file, mean=mean, std=std)


def update(existing, new):
    count, mean, M2 = existing
    count += 1
    delta = new - mean
    mean += delta / count
    delta2 = new - mean
    M2 += delta * delta2
    return count, mean, M2


def finalize(existing):
    count, mean, M2 = existing
    variance = M2 / count if count > 1 else 0
    return mean, np.sqrt(variance)


def get_stats_directory(directory):
    dataset = AudioFolderDataset(directory)
    out_file = AudioFolderDataset.get_stats_file_path(directory)

    existing = (0, 0, 0)
    # we run Welford's online algorithm
    for i in range(len(dataset)):
        x = dataset[i]["mel_spec"]
        existing = update(existing, x)

        if i % 10 == 0:
            logger.info(f"Processed {i + 1} files...")

    mean, std = finalize(existing)

    logger.info(f"Saving to {str(out_file)}")

    np.savez(out_file, mean=mean, std=std)


if __name__ == "__main__":
    # filename = "/data/scratch/acw585/surge/train.hdf5"
    filename = sys.argv[1]

    if os.path.splitext(filename)[-1] == ".h5":
        get_stats_hdf5(filename)
    else:
        get_stats_directory(filename)
