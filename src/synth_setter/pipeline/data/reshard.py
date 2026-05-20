"""Reshard a directory of HDF5 shards into ``{train,val,test}.h5`` virtual datasets.

Splits and shard filenames are read from ``<dataset_root>/input_spec.json``;
``spec.render.samples_per_shard`` is the single source of truth for shard size.
Outputs are staged under ``<dataset_root>/.tmp-<split>.h5`` and renamed into
place after ``create_virtual_dataset`` succeeds, so a mid-loop failure on
shard N never leaves a partially-populated ``<split>.h5`` next to the inputs.
"""

import os
from pathlib import Path

import click
import h5py
import numpy as np

from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import load_spec_from_uri

# Datasets every shard must carry. Worker validation enforces this, but
# reshard re-checks at the trust boundary so a drifted or hand-edited shard
# surfaces a clean ClickException instead of a bare KeyError.
_REQUIRED_DATASETS: tuple[str, ...] = ("audio", "mel_spec", "param_array")


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
def main(dataset_root: Path, spec_uri: str | None) -> None:  # noqa: DOC502
    """Split shards under ``dataset_root`` into ``{train,val,test}.h5`` virtual datasets.

    :param dataset_root: Directory containing the shard files named by ``spec.shards``.
    :param spec_uri: Optional local path or ``r2://`` URI for the DatasetSpec;
        defaults to ``<dataset_root>/input_spec.json``.
    :raises click.ClickException: If the spec cannot be located, parsed, or
        declares a non-HDF5 ``output_format``; if a shard named in
        ``spec.shards`` is missing from ``dataset_root``; or if a shard's
        on-disk shape/dtype disagrees with the spec.
    """
    spec = _load_spec(spec_uri, dataset_root)
    shard_paths = [dataset_root / s.filename for s in spec.shards]
    _check_shards_present(shard_paths)
    _check_shard_contracts(shard_paths, spec.render.samples_per_shard)

    shard_size = spec.render.samples_per_shard
    train_n, val_n, test_n = (sz // shard_size for sz in spec.train_val_test_sizes)
    splits = {
        "train": shard_paths[:train_n],
        "val": shard_paths[train_n : train_n + val_n],
        "test": shard_paths[train_n + val_n : train_n + val_n + test_n],
    }
    for split, split_paths in splits.items():
        if not split_paths:
            continue
        click.echo(f"{split}: {len(split_paths)} shards")
        _write_split(dataset_root, split, split_paths, shard_size)


def _load_spec(spec_uri: str | None, dataset_root: Path) -> DatasetSpec:
    """Resolve ``spec_uri`` (or the default under ``dataset_root``) and load it.

    :param spec_uri: Operator-supplied URI, or ``None`` to use the default.
    :param dataset_root: Directory used to compute the default ``input_spec.json`` path.
    :returns: The parsed spec.
    :raises click.ClickException: With a user-oriented message when the spec
        is missing, malformed, or declares ``output_format!="hdf5"``.
    """
    resolved_uri = spec_uri or str(dataset_root / INPUT_SPEC_FILENAME)
    try:
        spec = load_spec_from_uri(resolved_uri)
    except FileNotFoundError as exc:
        raise click.ClickException(f"DatasetSpec not found at {resolved_uri}: {exc}") from exc
    except ValueError as exc:
        # Unsupported scheme from ``read_spec_text`` and ``pydantic.ValidationError``
        # (a ValueError subclass) for malformed / stale specs both land here.
        raise click.ClickException(f"DatasetSpec at {resolved_uri} is invalid: {exc}") from exc

    if spec.output_format != "hdf5":
        raise click.ClickException(
            f"reshard only supports output_format='hdf5'; spec at {resolved_uri} "
            f"declares output_format={spec.output_format!r}."
        )
    return spec


def _check_shards_present(shard_paths: list[Path]) -> None:
    """Fail loud before any output handle opens if a spec-named shard is missing.

    :param shard_paths: All shards reshard will source from, in spec order.
    :raises click.ClickException: Listing each missing path.
    """
    missing = [p for p in shard_paths if not p.is_file()]
    if missing:
        formatted = "\n  ".join(str(p) for p in missing)
        raise click.ClickException(
            f"{len(missing)} shard(s) named by ``spec.shards`` are missing under "
            f"dataset_root:\n  {formatted}"
        )


def _check_shard_contracts(shard_paths: list[Path], samples_per_shard: int) -> None:
    """Assert every shard has the expected datasets, dtype, and shape.

    Catches a drifted or partial worker upload before reshard wires the file
    into a ``VirtualSource`` (which would otherwise either silently return
    fill values or surface as a low-signal h5py error mid-run).

    :param shard_paths: Spec-ordered list of shard files.
    :param samples_per_shard: Required leading-axis length for every dataset.
    :raises click.ClickException: With the offending shard, key, and observed
        shape/dtype.
    """
    expected_tails: dict[str, tuple[int, ...]] = {}
    for shard in shard_paths:
        with h5py.File(shard, "r") as f:
            for key in _REQUIRED_DATASETS:
                if key not in f:
                    raise click.ClickException(
                        f"shard {shard} is missing required dataset {key!r}; "
                        f"present: {sorted(f.keys())}."
                    )
                dataset = f[key]
                if not isinstance(dataset, h5py.Dataset):
                    raise click.ClickException(
                        f"shard {shard}: key {key!r} is a {type(dataset).__name__}, not a Dataset."
                    )
                if dataset.dtype != np.float32:
                    raise click.ClickException(
                        f"shard {shard}: dataset {key!r} has dtype {dataset.dtype}, "
                        f"expected np.float32."
                    )
                if dataset.shape[0] != samples_per_shard:
                    raise click.ClickException(
                        f"shard {shard}: dataset {key!r} has {dataset.shape[0]} rows, "
                        f"expected samples_per_shard={samples_per_shard}."
                    )
                tail = tuple(dataset.shape[1:])
                expected_tail = expected_tails.setdefault(key, tail)
                if tail != expected_tail:
                    raise click.ClickException(
                        f"shard {shard}: dataset {key!r} trailing shape {tail} disagrees "
                        f"with first shard's {expected_tail}."
                    )


def _write_split(  # noqa: DOC501,DOC503
    dataset_root: Path,
    split: str,
    split_paths: list[Path],
    shard_size: int,
) -> None:
    """Assemble one split as a virtual dataset, staged then atomically renamed.

    The staging file is deleted on any exception (including KeyboardInterrupt)
    so a partial ``.tmp-*.h5`` never lingers; the original exception is
    re-raised unchanged.

    :param dataset_root: Destination directory; output lands at ``<dataset_root>/<split>.h5``.
    :param split: Split name (``train``/``val``/``test``); becomes the output filename stem.
    :param split_paths: Shard files concatenated in order into the virtual layout.
    :param shard_size: Per-shard row count; identical for every shard by the
        :func:`_check_shard_contracts` invariant.
    """
    split_len = len(split_paths) * shard_size
    with h5py.File(split_paths[0], "r") as f:
        # Trailing shape is identical across shards by ``_check_shard_contracts``,
        # so reading it once off the first shard is safe.
        tails: dict[str, tuple[int, ...]] = {
            key: tuple(f[key].shape[1:]) for key in _REQUIRED_DATASETS
        }

    layouts = {
        key: h5py.VirtualLayout(shape=(split_len, *tail), dtype=np.float32)
        for key, tail in tails.items()
    }
    for i, shard_path in enumerate(split_paths):
        range_start = i * shard_size
        range_end = range_start + shard_size
        for key, tail in tails.items():
            source = h5py.VirtualSource(
                shard_path, key, dtype=np.float32, shape=(shard_size, *tail)
            )
            layouts[key][range_start:range_end] = source

    final_path = dataset_root / f"{split}.h5"
    staging_path = dataset_root / f".tmp-{split}.h5"
    try:
        with h5py.File(staging_path, "w") as f:
            for key, layout in layouts.items():
                f.create_virtual_dataset(key, layout)
        os.replace(staging_path, final_path)
    except BaseException:
        staging_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
