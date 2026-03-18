#!/usr/bin/env python
import multiprocessing
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

import click
import h5py
import numpy as np
from einops import rearrange
from loguru import logger
from music2latent import EncoderDecoder  # Your model: adjust as needed.
from tqdm import tqdm


def get_shard_id(shard_path: Path) -> int:
    return int(shard_path.stem.split("-")[1])


def reader_process(shard_path: Path, batch_size: int, read_queue: Queue, batch_indices: list[int]):
    """Opens the HDF5 file in read-only SWMR mode and reads only the batches whose indices are in
    `batch_indices` from the "audio" dataset.

    Each read batch is put on the read_queue as a tuple: (start_index, end_index, audio_data).
    When finished, a None is put on the queue as a sentinel.
    """
    with h5py.File(str(shard_path), "r", libver="latest", swmr=True) as f:
        num_samples = f["audio"].shape[0]
        for i in batch_indices:
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            audio = f["audio"][start:end]
            read_queue.put((start, end, audio))
    # Signal that this reader is done.
    read_queue.put(None)


def writer_process(shard_path: Path, write_queue: Queue):
    """Opens the HDF5 file in read/write SWMR mode and writes processed batches into the
    "music2latent" dataset.

    Each item in the write_queue is expected to be a tuple:
    (start_index, end_index, processed_data). When finished (i.e. when receiving None),
    the process exits.
    """
    logger.info(f"Opening for writing to {shard_path}")

    os.system(f"h5clear -s {str(shard_path)}")
    with h5py.File(str(shard_path), "r+", libver="latest") as f:
        f.swmr_mode = True
        while True:
            item = write_queue.get()
            if item is None:
                break
            start, end, m2l_out_np = item
            f["music2latent"][start:end] = m2l_out_np
            f.flush()  # Flush so changes are visible in SWMR mode.


def process_shard(shard_path: Path, batch_size: int, m2l: EncoderDecoder, num_readers: int):
    """For a given shard, first pre-create the output dataset if needed.

    Then spawn multiple reader processes and one writer process. The main (GPU) process pulls
    batches from the read_queue, processes them, and pushes the results onto the write_queue.
    """
    # First, create (or verify) the output dataset.
    # first we run `h5clear -s file`
    os.system(f"h5clear -s {str(shard_path)}")
    with h5py.File(str(shard_path), "r+", libver="latest") as f:
        f.swmr_mode = True
        num_samples = f["audio"].shape[0]
        if "music2latent" not in f:
            # Adjust the shape and dtype as needed.
            f.create_dataset("music2latent", shape=(num_samples, 128, 42), dtype=np.float32)
        f.flush()

    f.close()

    # Compute the number of batches.
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Create multiprocessing queues.
    read_queue = Queue(maxsize=10)
    write_queue = Queue(maxsize=10)

    # Partition the batch indices among the reader processes.
    all_batch_indices = list(range(num_batches))
    # Use round-robin partitioning.
    split_batches = [all_batch_indices[i::num_readers] for i in range(num_readers)]

    # Start the writer process.
    writer_proc = Process(target=writer_process, args=(shard_path, write_queue))
    writer_proc.start()
    time.sleep(5)

    # Start the reader processes.
    reader_procs = []
    for sub_batch_indices in split_batches:
        proc = Process(
            target=reader_process,
            args=(shard_path, batch_size, read_queue, sub_batch_indices),
        )
        proc.start()
        reader_procs.append(proc)

    # Main loop: get batches from the read_queue, process on GPU, send to writer.
    finished_readers = 0
    pbar = tqdm(total=num_batches, desc=f"Processing {shard_path.name}")

    while finished_readers < num_readers:
        item = read_queue.get()
        if item is None:
            finished_readers += 1
            continue

        start, end, audio = item

        # Rearrange audio (assuming original shape: (batch, channels, time)).
        audio_reshaped = rearrange(audio, "b c t -> (b c) t")
        # Process the batch on the GPU.
        m2l_out = m2l.encode(audio_reshaped, max_batch_size=audio_reshaped.shape[0])
        # Determine the original batch size and channel count.
        n = audio.shape[0]  # Might be less than batch_size for the last batch.
        c = audio.shape[1]
        # Rearrange the output to (n, (c*d), t) if the model output has shape ((n*c), d, t).
        m2l_out = rearrange(m2l_out, "(n c) d t -> n (c d) t", n=n, c=c)
        # Convert to a NumPy array (if m2l_out is e.g. a torch.Tensor).
        m2l_out_np = m2l_out.cpu().numpy()

        # Send processed batch to the writer process.
        write_queue.put((start, end, m2l_out_np))
        pbar.update(1)
    pbar.close()

    # Signal the writer process that processing is complete.
    write_queue.put(None)

    # Wait for all processes to finish.
    for proc in reader_procs:
        proc.join()
    writer_proc.join()


@click.command()
@click.argument("data_dir", type=str)
@click.option("--batch-size", "-c", type=int, default=1024, help="Batch size for processing.")
@click.option("--shard-range", "-r", type=int, nargs=2, default=None, help="Optional shard range.")
@click.option("--shard", "-s", type=int, default=None, help="Optional shard index.")
@click.option(
    "--num-readers",
    "-n",
    type=int,
    default=4,
    help="Number of concurrent reader processes.",
)
def main(
    data_dir: str,
    batch_size: int,
    shard_range: tuple[int, int] | None,
    shard: int | None,
    num_readers: int,
):
    data_dir = Path(data_dir)
    data_shards = list(data_dir.glob("shard-*.h5"))

    if shard_range is not None and shard is not None:
        raise ValueError("Cannot specify both --shard-range and --shard.")

    if shard_range is not None:
        data_shards = [ds for ds in data_shards if get_shard_id(ds) in range(*shard_range)]
    if shard is not None:
        data_shards = [ds for ds in data_shards if get_shard_id(ds) == shard]

    if not data_shards:
        click.echo("No valid data shards found.")
        return

    data_shards.sort(key=get_shard_id)

    # Create the model instance (holds GPU inference code).
    m2l = EncoderDecoder()

    for data_shard in data_shards:
        logger.info(f"Starting processing for shard: {data_shard.name}")
        process_shard(data_shard, batch_size, m2l, num_readers)
        logger.info(f"Finished processing shard: {data_shard.name}")


if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # On some platforms (e.g. Windows), the multiprocessing start method must be guarded.
    multiprocessing.set_start_method("spawn", force=True)
    main()
