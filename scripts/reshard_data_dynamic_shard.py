"""Reshard a directory of HDF5 shard files into train/val/test virtual datasets.

Reads shard-*.h5 files from <dataset_root>/shards/, assigns them to splits,
and creates virtual HDF5 datasets (train.h5, val.h5, test.h5) in dataset_root.

Virtual datasets contain no data — they reference the underlying shard files
using relative paths, so the output is portable as long as the directory
structure is preserved.

Usage:
    python scripts/reshard_data_dynamic_shard.py data/surge_simple \\
        --val-shards 1 --test-shards 1

    # Explicit train count (default: remainder after val + test)
    python scripts/reshard_data_dynamic_shard.py data/surge_simple \\
        --train-shards 10 --val-shards 1 --test-shards 1
"""

import sys
from pathlib import Path

import click
import h5py
import numpy as np


def reshard_split(shard_files: list[Path], output_file: Path) -> None:
    """Create a virtual HDF5 dataset by concatenating shard files along axis 0.

    Args:
        shard_files: Ordered list of shard HDF5 file paths. Each must contain
            datasets named 'audio', 'mel_spec', and 'param_array'.
        output_file: Path for the output virtual HDF5 file.

    Raises:
        ValueError: If shard_files is empty.
    """
    if not shard_files:
        raise ValueError("No shard files provided")

    output_dir = output_file.parent

    shard_sizes = []
    for f in shard_files:
        with h5py.File(f, "r") as h:
            shard_sizes.append(h["audio"].shape[0])
    split_len = sum(shard_sizes)

    with h5py.File(shard_files[0], "r") as f:
        audio_shape = f["audio"].shape[1:]
        mel_shape = f["mel_spec"].shape[1:]
        param_shape = f["param_array"].shape[1:]

    vl_audio = h5py.VirtualLayout(shape=(split_len, *audio_shape), dtype=np.float32)
    vl_mel = h5py.VirtualLayout(shape=(split_len, *mel_shape), dtype=np.float32)
    vl_param = h5py.VirtualLayout(shape=(split_len, *param_shape), dtype=np.float32)

    offset = 0
    for file, size in zip(shard_files, shard_sizes):
        # Use path relative to the output file's directory for portability.
        rel_path = file.resolve().relative_to(output_dir.resolve())
        rel_str = str(rel_path)

        vs_audio = h5py.VirtualSource(
            rel_str, "audio", dtype=np.float32, shape=(size, *audio_shape)
        )
        vs_mel = h5py.VirtualSource(
            rel_str, "mel_spec", dtype=np.float32, shape=(size, *mel_shape)
        )
        vs_param = h5py.VirtualSource(
            rel_str, "param_array", dtype=np.float32, shape=(size, *param_shape)
        )

        vl_audio[offset : offset + size] = vs_audio
        vl_mel[offset : offset + size] = vs_mel
        vl_param[offset : offset + size] = vs_param
        offset += size

    with h5py.File(output_file, "w") as f:
        f.create_virtual_dataset("audio", vl_audio)
        f.create_virtual_dataset("mel_spec", vl_mel)
        f.create_virtual_dataset("param_array", vl_param)


@click.command()
@click.argument("dataset_root", type=str)
@click.option(
    "--train-shards",
    "-t",
    type=int,
    default=None,
    help="Train shard count (default: remainder after val + test).",
)
@click.option(
    "--val-shards", "-v", type=int, default=1, show_default=True, help="Validation shard count."
)
@click.option(
    "--test-shards", "-e", type=int, default=1, show_default=True, help="Test shard count."
)
def main(
    dataset_root: str,
    train_shards: int | None,
    val_shards: int,
    test_shards: int,
):
    dataset_root = Path(dataset_root)
    shard_dir = dataset_root / "shards"
    files = sorted(shard_dir.glob("shard-*.h5"))
    total = len(files)

    if total == 0:
        print("ERROR: No shard-*.h5 files found in", shard_dir, file=sys.stderr)
        sys.exit(1)

    if train_shards is None:
        train_shards = total - val_shards - test_shards

    if train_shards + val_shards + test_shards != total:
        print(
            f"ERROR: shard count mismatch: train({train_shards}) + val({val_shards}) "
            f"+ test({test_shards}) = {train_shards + val_shards + test_shards}, "
            f"but found {total} shards.",
            file=sys.stderr,
        )
        sys.exit(1)

    if train_shards < 0:
        print(
            f"ERROR: not enough shards for val({val_shards}) + test({test_shards}) "
            f"— only {total} shards available.",
            file=sys.stderr,
        )
        sys.exit(1)

    splits = {
        "train": files[:train_shards],
        "val": files[train_shards : train_shards + val_shards],
        "test": files[train_shards + val_shards :],
    }

    for split_name, split_files in splits.items():
        if not split_files:
            print(f"[reshard] skipping {split_name} (0 shards)")
            continue
        out_path = dataset_root / f"{split_name}.h5"
        n_samples = sum(h5py.File(f, "r")["audio"].shape[0] for f in split_files)
        print(
            f"[reshard] {split_name}: {len(split_files)} shards, {n_samples} samples -> {out_path}"
        )
        reshard_split(split_files, out_path)


if __name__ == "__main__":
    main()
