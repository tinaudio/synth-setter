"""Reshard a directory of HDF5 shards into ``{train,val,test}.h5`` virtual datasets.

Splits and shard filenames are read from ``<dataset_root>/input_spec.json``;
``spec.render.samples_per_shard`` is the single source of truth for shard size.
All split outputs are staged under ``<dataset_root>/.tmp-<split>.h5`` and only
renamed into place after every split's ``create_virtual_dataset`` succeeds, so
a failure on any shard leaves no partial ``{train,val,test}.h5`` next to the
inputs.
"""

from pathlib import Path

import click
import h5py

from synth_setter.data.vst.shapes import DATASET_FIELD_DTYPES
from synth_setter.pipeline.ci.validate_shard import (
    check_shard_contracts,
    check_shard_ids_match_spec_order,
    check_shards_present,
)
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import load_spec_from_uri

# Output filenames, in the order ``main`` writes them.
_SPLITS: tuple[str, ...] = ("train", "val", "test")

# Prefix for staging files under ``dataset_root``. Cleanup paths and the
# ``TestReshardAtomicWrite`` assertions both depend on this exact string.
_STAGING_PREFIX = ".tmp-"


def reshard_dataset(dataset_root: Path, spec_uri: str | None = None) -> None:  # noqa: DOC502
    """Split shards under ``dataset_root`` into ``{train,val,test}.h5`` virtual datasets.

    Pure-function form of the reshard operation; importable from other
    pipeline stages (notably ``cli.finalize_dataset.finalize_hdf5``) without
    invoking the Click wrapper's ``.callback`` indirection. The :func:`main`
    command below is a thin Click adapter that delegates to this function.

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
    check_shards_present(shard_paths)
    check_shard_ids_match_spec_order(spec)
    tails = check_shard_contracts(shard_paths, spec.render.samples_per_shard)
    shard_size = spec.render.samples_per_shard

    splits = _build_splits(shard_paths, spec.train_val_test_sizes, shard_size)
    _stage_and_commit_splits(dataset_root, splits, shard_size, tails)


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
    """Click adapter that delegates to :func:`reshard_dataset`.

    :param dataset_root: Directory containing the shard files named by ``spec.shards``.
    :param spec_uri: Optional local path or ``r2://`` URI for the DatasetSpec;
        defaults to ``<dataset_root>/input_spec.json``.
    :raises click.ClickException: Propagated from :func:`reshard_dataset`.
    """
    reshard_dataset(dataset_root, spec_uri)


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


def _build_splits(
    shard_paths: list[Path],
    train_val_test_sizes: tuple[int, int, int],
    shard_size: int,
) -> dict[str, list[Path]]:
    """Map each split name to the slice of ``shard_paths`` it owns.

    :param shard_paths: Spec-ordered list of every shard file.
    :param train_val_test_sizes: Spec-declared row counts per split.
    :param shard_size: ``samples_per_shard``; each size must be a multiple.
    :returns: ``{split_name: [shard_path, ...]}``; empty lists for zero-sized splits.
    """
    counts = tuple(sz // shard_size for sz in train_val_test_sizes)
    starts = (0, counts[0], counts[0] + counts[1])
    return {
        name: shard_paths[start : start + count]
        for name, start, count in zip(_SPLITS, starts, counts, strict=True)
    }


def _stage_and_commit_splits(  # noqa: DOC501,DOC503
    dataset_root: Path,
    splits: dict[str, list[Path]],
    shard_size: int,
    tails: dict[str, tuple[int, ...]],
) -> None:
    """Stage every non-empty split under ``.tmp-<split>.h5``, then atomically rename.

    Across-splits atomicity: every staging write succeeds before any rename
    runs, and any failure during *either* phase unlinks every staging file
    plus every final output already renamed in this call before re-raising.
    So no ``{train,val,test}.h5`` ever appears unless every output file was
    successfully created. Exceptions from :mod:`h5py` (during staging) and
    :meth:`pathlib.Path.replace` (during rename) propagate raw — only the
    cleanup is added on top.

    Empty splits are pruned to match the spec: any stale ``{name}.h5`` or
    ``.tmp-{name}.h5`` left from a previous run with a non-zero size is
    unlinked up front, so the dataset_root after this call always reflects
    the spec's current split sizes (non-empty splits' outputs are
    overwritten by the rename phase, so they don't need pre-cleanup).

    :param dataset_root: Directory that holds both the input shards and the outputs.
    :param splits: Per-split shard lists from :func:`_build_splits`.
    :param shard_size: ``samples_per_shard`` for every shard.
    :param tails: Per-dataset trailing shape from :func:`check_shard_contracts`.
    """
    for name, paths in splits.items():
        if not paths:
            (dataset_root / f"{name}.h5").unlink(missing_ok=True)
            (dataset_root / f"{_STAGING_PREFIX}{name}.h5").unlink(missing_ok=True)
    nonempty = [(name, paths) for name, paths in splits.items() if paths]
    staging_paths = [dataset_root / f"{_STAGING_PREFIX}{name}.h5" for name, _ in nonempty]
    final_paths = [dataset_root / f"{name}.h5" for name, _ in nonempty]
    renamed: list[Path] = []
    try:
        for (name, paths), staging in zip(nonempty, staging_paths, strict=True):
            click.echo(f"{name}: {len(paths)} shards")
            _write_split(staging, paths, shard_size, tails)
        for staging, final in zip(staging_paths, final_paths, strict=True):
            staging.replace(final)
            renamed.append(final)
    except BaseException:
        for staging in staging_paths:
            staging.unlink(missing_ok=True)
        for final in renamed:
            final.unlink(missing_ok=True)
        raise


def _write_split(
    staging_path: Path,
    split_paths: list[Path],
    shard_size: int,
    tails: dict[str, tuple[int, ...]],
) -> None:
    """Materialize one split's virtual dataset into ``staging_path``.

    VDS source references are written as relative filenames (the shard
    basename, not the absolute path) so the resulting ``{split}.h5`` resolves
    against any directory that holds sibling shards — required because the
    file is uploaded to R2 from a temp dir and later read from a different
    local cache by training.

    :param staging_path: ``.tmp-<split>.h5`` destination; the caller renames
        it into place after every split's stage succeeds.
    :param split_paths: Shard files concatenated in order into the virtual layout.
    :param shard_size: Per-shard row count; identical for every shard by the
        :func:`check_shard_contracts` invariant.
    :param tails: Per-dataset trailing shape from :func:`check_shard_contracts`.
    """
    split_len = len(split_paths) * shard_size
    layouts = {
        key: h5py.VirtualLayout(shape=(split_len, *tail), dtype=DATASET_FIELD_DTYPES[key])
        for key, tail in tails.items()
    }
    for i, shard_path in enumerate(split_paths):
        range_start = i * shard_size
        range_end = range_start + shard_size
        for key, tail in tails.items():
            source = h5py.VirtualSource(
                shard_path.name,
                key,
                dtype=DATASET_FIELD_DTYPES[key],
                shape=(shard_size, *tail),
            )
            layouts[key][range_start:range_end] = source

    with h5py.File(staging_path, "w") as f:
        for key, layout in layouts.items():
            f.create_virtual_dataset(key, layout)


if __name__ == "__main__":
    main()
