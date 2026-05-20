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
import numpy as np

from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import load_spec_from_uri

# Re-checked at the trust boundary so a drifted shard surfaces a clean
# ClickException instead of a bare KeyError mid-VirtualSource wiring.
_REQUIRED_DATASETS: tuple[str, ...] = ("audio", "mel_spec", "param_array")

# Output filenames, in the order ``main`` writes them.
_SPLITS: tuple[str, ...] = ("train", "val", "test")

# Prefix for staging files under ``dataset_root``. Cleanup paths and the
# ``TestReshardAtomicWrite`` assertions both depend on this exact string.
_STAGING_PREFIX = ".tmp-"

_FLOAT32 = np.dtype("float32")


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
    _check_shard_ids_match_spec_order(spec)
    tails = _check_shard_contracts(shard_paths, spec.render.samples_per_shard)
    shard_size = spec.render.samples_per_shard

    splits = _build_splits(shard_paths, spec.train_val_test_sizes, shard_size)
    _stage_and_commit_splits(dataset_root, splits, shard_size, tails)


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


def _check_shard_ids_match_spec_order(spec: DatasetSpec) -> None:
    """Catch a tampered spec whose ``shards[i].shard_id`` no longer equals ``i``.

    :param spec: Loaded ``DatasetSpec``.
    :raises click.ClickException: If any shard's ``shard_id`` disagrees with its index.
    """
    for index, shard in enumerate(spec.shards):
        if shard.shard_id != index:
            raise click.ClickException(
                f"spec.shards[{index}].shard_id={shard.shard_id} disagrees with its "
                f"position; spec.shards must be in shard_id order."
            )


def _check_shard_contracts(
    shard_paths: list[Path],
    samples_per_shard: int,
) -> dict[str, tuple[int, ...]]:
    """Validate every shard's structure and return the per-dataset trailing shape.

    Catches a drifted or partial worker upload before reshard wires the file
    into a ``VirtualSource`` (which would otherwise either silently return
    fill values or surface as a low-signal h5py error mid-run). The returned
    tails are reused by :func:`_write_split` so the first shard isn't reopened.

    :param shard_paths: Spec-ordered list of shard files.
    :param samples_per_shard: Required leading-axis length for every dataset.
    :returns: Trailing shape for each required dataset key.
    :raises click.ClickException: With the offending shard, key, and observed
        value (shape, dtype, or row count).
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
                node = f[key]
                if not isinstance(node, h5py.Dataset):
                    raise click.ClickException(
                        f"shard {shard}: key {key!r} is a {type(node).__name__}, not a Dataset."
                    )
                if node.dtype != _FLOAT32:
                    raise click.ClickException(
                        f"shard {shard}: dataset {key!r} has dtype {node.dtype}, "
                        f"expected np.float32."
                    )
                if node.shape[0] != samples_per_shard:
                    raise click.ClickException(
                        f"shard {shard}: dataset {key!r} has {node.shape[0]} rows, "
                        f"expected samples_per_shard={samples_per_shard}."
                    )
                tail = tuple(node.shape[1:])
                expected_tail = expected_tails.setdefault(key, tail)
                if tail != expected_tail:
                    raise click.ClickException(
                        f"shard {shard}: dataset {key!r} trailing shape {tail} disagrees "
                        f"with first shard's {expected_tail}."
                    )
    return expected_tails


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


def _stage_and_commit_splits(  # noqa: DOC503
    dataset_root: Path,
    splits: dict[str, list[Path]],
    shard_size: int,
    tails: dict[str, tuple[int, ...]],
) -> None:
    """Stage every non-empty split under ``.tmp-<split>.h5``, then atomically rename.

    Across-splits atomicity: every staging write succeeds before any rename;
    any in-flight failure unlinks every staging file before re-raising. So no
    ``{train,val,test}.h5`` ever appears unless every output file was
    successfully created.

    :param dataset_root: Directory that holds both the input shards and the outputs.
    :param splits: Per-split shard lists from :func:`_build_splits`.
    :param shard_size: ``samples_per_shard`` for every shard.
    :param tails: Per-dataset trailing shape from :func:`_check_shard_contracts`.
    :raises click.ClickException: Propagated from h5py via ``main``'s contract;
        listed for pydoclint completeness only — the staging cleanup happens
        for *any* exception (including ``KeyboardInterrupt``).
    """
    nonempty = [(name, paths) for name, paths in splits.items() if paths]
    staging_paths = [dataset_root / f"{_STAGING_PREFIX}{name}.h5" for name, _ in nonempty]
    try:
        for (name, paths), staging in zip(nonempty, staging_paths, strict=True):
            click.echo(f"{name}: {len(paths)} shards")
            _write_split(staging, paths, shard_size, tails)
        for (name, _), staging in zip(nonempty, staging_paths, strict=True):
            staging.replace(dataset_root / f"{name}.h5")
    except BaseException:
        # Any failure between the first staging open and the last rename leaves
        # zero ``.tmp-*.h5`` files behind; previously-renamed splits cannot land
        # because every rename happens after every stage succeeds.
        for staging in staging_paths:
            staging.unlink(missing_ok=True)
        raise


def _write_split(
    staging_path: Path,
    split_paths: list[Path],
    shard_size: int,
    tails: dict[str, tuple[int, ...]],
) -> None:
    """Materialize one split's virtual dataset into ``staging_path``.

    :param staging_path: ``.tmp-<split>.h5`` destination; the caller renames
        it into place after every split's stage succeeds.
    :param split_paths: Shard files concatenated in order into the virtual layout.
    :param shard_size: Per-shard row count; identical for every shard by the
        :func:`_check_shard_contracts` invariant.
    :param tails: Per-dataset trailing shape from :func:`_check_shard_contracts`.
    """
    split_len = len(split_paths) * shard_size
    layouts = {
        key: h5py.VirtualLayout(shape=(split_len, *tail), dtype=_FLOAT32)
        for key, tail in tails.items()
    }
    for i, shard_path in enumerate(split_paths):
        range_start = i * shard_size
        range_end = range_start + shard_size
        for key, tail in tails.items():
            source = h5py.VirtualSource(shard_path, key, dtype=_FLOAT32, shape=(shard_size, *tail))
            layouts[key][range_start:range_end] = source

    with h5py.File(staging_path, "w") as f:
        for key, layout in layouts.items():
            f.create_virtual_dataset(key, layout)


if __name__ == "__main__":
    main()
