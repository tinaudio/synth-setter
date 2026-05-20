"""Reshard a directory of HDF5 shards into train/val/test virtual datasets.

Both the split sizes *and* the exact shard filenames come from
``<dataset_root>/input_spec.json`` (written once by the launcher at
``@hydra.main`` and uploaded alongside the shards). The CLI reads
``spec.shards`` in order and slices into ``{train, val, test}.h5``
according to ``spec.train_val_test_sizes // spec.render.samples_per_shard``;
``spec.render.samples_per_shard`` is the single source of truth, so
there is no shard-size override flag to drift against it.
"""

from pathlib import Path

import click
import h5py
import numpy as np

from synth_setter.cli.generate_dataset import load_spec_from_uri
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME


@click.command()
@click.argument("dataset_root", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--spec",
    "spec_uri",
    type=str,
    default=None,
    help=(
        "Local path or ``r2://bucket/key`` URI of the materialized DatasetSpec JSON "
        "(``r2://`` is downloaded via rclone — ``RCLONE_CONFIG_R2_*`` env vars must be set). "
        f"Defaults to ``<dataset_root>/{INPUT_SPEC_FILENAME}``."
    ),
)
def main(dataset_root: Path, spec_uri: str | None) -> None:
    """Split shards under ``dataset_root`` into ``{train,val,test}.h5`` virtual datasets.

    :param dataset_root: Directory containing the shard files named by ``spec.shards``.
    :param spec_uri: Optional local path or ``r2://`` URI for the DatasetSpec;
        defaults to ``<dataset_root>/input_spec.json``.
    :raises click.ClickException: If any ``spec.train_val_test_sizes`` entry is not
        evenly divisible by ``spec.render.samples_per_shard``.
    """
    spec = load_spec_from_uri(spec_uri or str(dataset_root / INPUT_SPEC_FILENAME))

    shard_size = spec.render.samples_per_shard
    bad = [sz for sz in spec.train_val_test_sizes if sz % shard_size != 0]
    if bad:
        # DatasetSpec._split_sizes_must_be_multiples_of_samples_per_shard normally
        # rejects this at parse time; this guard catches the case where a stale
        # spec from R2 predates that validator.
        raise click.ClickException(
            f"spec.train_val_test_sizes={list(spec.train_val_test_sizes)} contains "
            f"sizes not divisible by spec.render.samples_per_shard={shard_size}: "
            f"{bad}"
        )

    files = [dataset_root / s.filename for s in spec.shards]
    train_n, val_n, _ = (sz // shard_size for sz in spec.train_val_test_sizes)
    splits = {
        "train": files[:train_n],
        "val": files[train_n : train_n + val_n],
        "test": files[train_n + val_n :],
    }

    for split, files in splits.items():
        if not files:
            continue
        click.echo(f"{split}: {len(files)} shards")
        split_len = len(files) * shard_size

        with h5py.File(files[0], "r") as f:
            audio_shape = f["audio"].shape[1:]
            mel_shape = f["mel_spec"].shape[1:]
            param_shape = f["param_array"].shape[1:]

        vl_audio = h5py.VirtualLayout(shape=(split_len, *audio_shape), dtype=np.float32)
        vl_mel = h5py.VirtualLayout(shape=(split_len, *mel_shape), dtype=np.float32)
        vl_param = h5py.VirtualLayout(shape=(split_len, *param_shape), dtype=np.float32)

        for i, file in enumerate(files):
            vs_audio = h5py.VirtualSource(
                file, "audio", dtype=np.float32, shape=(shard_size, *audio_shape)
            )
            vs_mel = h5py.VirtualSource(
                file, "mel_spec", dtype=np.float32, shape=(shard_size, *mel_shape)
            )
            vs_param = h5py.VirtualSource(
                file, "param_array", dtype=np.float32, shape=(shard_size, *param_shape)
            )

            range_start = i * shard_size
            range_end = (i + 1) * shard_size

            vl_audio[range_start:range_end, :, :] = vs_audio
            vl_mel[range_start:range_end, :, :, :] = vs_mel
            vl_param[range_start:range_end, :] = vs_param

        with h5py.File(dataset_root / f"{split}.h5", "w") as f:
            f.create_virtual_dataset("audio", vl_audio)
            f.create_virtual_dataset("mel_spec", vl_mel)
            f.create_virtual_dataset("param_array", vl_param)


if __name__ == "__main__":
    main()
