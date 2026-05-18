"""Reshard a directory of HDF5 shards into train/val/test virtual datasets.

Both the split sizes *and* the exact shard filenames come from
``<dataset_root>/input_spec.json`` (written once by the launcher at
``@hydra.main`` and uploaded alongside the shards). Per-split shard
counts are ``size // render.samples_per_shard``, and the source files
are exactly ``spec.shards`` in order — no glob, so a stale or extra
``shard-NNNNNN.h5`` next to the dataset cannot silently displace a
canonical one.
"""

from pathlib import Path

import click
import h5py
import numpy as np

from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.schemas.spec import DatasetSpec

_SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def _load_spec(spec_path: Path) -> DatasetSpec:
    """Read a JSON-serialized DatasetSpec from disk.

    :param spec_path: Filesystem path to the ``input_spec.json`` produced by
        the launcher.
    :returns: The validated :class:`DatasetSpec`.
    :rtype: DatasetSpec
    """
    return DatasetSpec.model_validate_json(spec_path.read_text())


def _build_virtual_split(
    split_files: list[Path],
    split_path: Path,
    shard_size: int,
) -> None:
    """Write a single split's virtual-dataset HDF5 file pointing at ``split_files``.

    :param split_files: Ordered list of source shard files for this split.
    :param split_path: Destination path for the ``{split}.h5`` virtual dataset.
    :param shard_size: Number of samples per source shard.
    """
    split_len = len(split_files) * shard_size

    with h5py.File(split_files[0], "r") as f:
        audio_shape = f["audio"].shape[1:]
        mel_shape = f["mel_spec"].shape[1:]
        param_shape = f["param_array"].shape[1:]

    vl_audio = h5py.VirtualLayout(shape=(split_len, *audio_shape), dtype=np.float32)
    vl_mel = h5py.VirtualLayout(shape=(split_len, *mel_shape), dtype=np.float32)
    vl_param = h5py.VirtualLayout(shape=(split_len, *param_shape), dtype=np.float32)

    for i, file in enumerate(split_files):
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

    with h5py.File(split_path, "w") as f:
        f.create_virtual_dataset("audio", vl_audio)
        f.create_virtual_dataset("mel_spec", vl_mel)
        f.create_virtual_dataset("param_array", vl_param)


@click.command()
@click.argument("dataset_root", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        f"Path to the materialized DatasetSpec JSON. "
        f"Defaults to ``<dataset_root>/{INPUT_SPEC_FILENAME}``."
    ),
)
@click.option(
    "--shard-size",
    "-s",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Override the per-shard sample count. Must match "
        "``spec.render.samples_per_shard`` exactly; mismatches abort. "
        "Defaults to the value from the spec."
    ),
)
def main(
    dataset_root: Path,
    spec_path: Path | None,
    shard_size: int | None,
) -> None:
    """Split ``shard-*.h5`` files into ``{train,val,test}.h5`` virtual datasets.

    :param dataset_root: Directory containing the ``shard-*.h5`` files to combine.
    :param spec_path: Optional explicit path to the DatasetSpec JSON; defaults
        to ``<dataset_root>/input_spec.json``.
    :param shard_size: Optional override for the per-shard sample count;
        must equal ``spec.render.samples_per_shard``.
    :raises click.ClickException: If the spec disagrees with the requested
        ``--shard-size``.
    """
    resolved_spec_path = spec_path if spec_path is not None else dataset_root / INPUT_SPEC_FILENAME
    spec = _load_spec(resolved_spec_path)

    sps = spec.render.samples_per_shard
    if shard_size is not None and shard_size != sps:
        raise click.ClickException(
            f"--shard-size={shard_size} disagrees with "
            f"spec.render.samples_per_shard={sps}; remove the override or "
            f"regenerate the spec"
        )

    files = [dataset_root / shard.filename for shard in spec.shards]

    cursor = 0
    for split_name, split_size in zip(_SPLIT_NAMES, spec.train_val_test_sizes, strict=True):
        n_shards = split_size // sps
        if n_shards == 0:
            continue
        split_files = files[cursor : cursor + n_shards]
        cursor += n_shards
        _build_virtual_split(split_files, dataset_root / f"{split_name}.h5", sps)
        click.echo(f"{split_name}: wrote {split_size} samples across {n_shards} shards")


if __name__ == "__main__":
    main()
