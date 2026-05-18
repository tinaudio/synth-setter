"""Reshard per-shard HDF5 files into train/val/test virtual-dataset files.

Split sizes are sourced from the ``DatasetSpec`` the shards were generated
against — never from CLI flags — so the resharded layout cannot silently
drift from the spec on R2.
"""

from __future__ import annotations

from pathlib import Path

import click
import h5py
import numpy as np

from synth_setter.cli.generate_dataset import load_spec_from_uri
from synth_setter.pipeline.schemas.spec import DatasetSpec

SHARD_GLOB = "shard-*.h5"
_SPLIT_LABELS: tuple[str, str, str] = ("train", "val", "test")
_VIRTUAL_DATASETS: tuple[str, ...] = ("audio", "mel_spec", "param_array")


def split_shard_counts(spec: DatasetSpec) -> tuple[int, int, int]:
    """Return ``(train, val, test)`` shard counts derived from ``spec``.

    ``DatasetSpec`` already enforces that each ``train_val_test_sizes`` entry
    is a multiple of ``render.samples_per_shard``, so the integer division is
    exact by construction.

    :param spec: The dataset spec the shards were generated against.
    :returns: A 3-tuple of shard counts in ``(train, val, test)`` order.
    :rtype: tuple[int, int, int]
    """
    samples_per_shard = spec.render.samples_per_shard
    train, val, test = spec.train_val_test_sizes
    return (
        train // samples_per_shard,
        val // samples_per_shard,
        test // samples_per_shard,
    )


def assign_shards_to_splits(
    shard_paths: list[Path], spec: DatasetSpec
) -> dict[str, list[Path]]:
    """Slice ``shard_paths`` into ``{train, val, test}`` per ``spec``.

    :param shard_paths: Sorted list of shard files under ``dataset_root``.
    :param spec: The dataset spec defining the expected split sizes.
    :returns: A dict mapping ``"train"``/``"val"``/``"test"`` to its shard files.
    :rtype: dict[str, list[Path]]
    :raises ValueError: If ``len(shard_paths)`` disagrees with the spec's
        expected total — that drift is exactly what consuming the spec is meant
        to catch, so we fail loud rather than silently slice.
    """
    counts = split_shard_counts(spec)
    expected_total = sum(counts)
    observed_total = len(shard_paths)
    if observed_total != expected_total:
        raise ValueError(
            "Shard count mismatch: spec expected "
            f"{expected_total} shards (train={counts[0]}, val={counts[1]}, "
            f"test={counts[2]}), observed {observed_total} on disk."
        )

    train_n, val_n, _ = counts
    return {
        "train": shard_paths[:train_n],
        "val": shard_paths[train_n : train_n + val_n],
        "test": shard_paths[train_n + val_n :],
    }


def _write_split_virtual_file(
    split_file: Path, shard_files: list[Path], samples_per_shard: int
) -> None:
    """Concatenate ``shard_files`` into one HDF5 virtual-dataset file.

    :param split_file: Destination ``.h5`` path for the virtual dataset.
    :param shard_files: Source shard files contributing to this split.
    :param samples_per_shard: Rows per source shard (from ``spec.render``).
    """
    with h5py.File(shard_files[0], "r") as f:
        per_dataset_tail_shape = {name: f[name].shape[1:] for name in _VIRTUAL_DATASETS}

    split_len = len(shard_files) * samples_per_shard
    layouts = {
        name: h5py.VirtualLayout(shape=(split_len, *tail), dtype=np.float32)
        for name, tail in per_dataset_tail_shape.items()
    }

    for i, file in enumerate(shard_files):
        start = i * samples_per_shard
        end = start + samples_per_shard
        for name, tail in per_dataset_tail_shape.items():
            source = h5py.VirtualSource(
                file, name, dtype=np.float32, shape=(samples_per_shard, *tail)
            )
            layouts[name][start:end] = source

    with h5py.File(split_file, "w") as f:
        for name, layout in layouts.items():
            f.create_virtual_dataset(name, layout)


@click.command()
@click.argument("dataset_root", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--spec",
    "spec_uri",
    required=True,
    type=str,
    help=(
        "Local path to a JSON-serialized DatasetSpec, or an `r2://bucket/key` URI "
        "(downloaded via rclone — RCLONE_CONFIG_R2_* env vars must be set)."
    ),
)
def main(dataset_root: Path, spec_uri: str) -> None:
    """Reshard per-shard HDF5 files under ``dataset_root`` into train/val/test.

    :param dataset_root: Directory containing the ``shard-NNNNNN.h5`` files.
    :param spec_uri: Local path or ``r2://`` URI to the dataset spec.
    """
    spec = load_spec_from_uri(spec_uri)
    shard_files = sorted(dataset_root.glob(SHARD_GLOB))
    splits = assign_shards_to_splits(shard_files, spec)

    samples_per_shard = spec.render.samples_per_shard
    for split_label in _SPLIT_LABELS:
        files = splits[split_label]
        click.echo(f"{split_label}: {len(files)} shards")
        split_file = dataset_root / f"{split_label}.h5"
        _write_split_virtual_file(split_file, files, samples_per_shard)


if __name__ == "__main__":
    main()
