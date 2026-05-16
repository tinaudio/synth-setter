import argparse
import io
import logging
import re
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import dask.array as da
import h5py
import numpy as np
import rootutils
from dask.distributed import Client, progress

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from synth_setter.data.audio_datamodule import AudioFolderDataset
from synth_setter.data.surge_datamodule import SurgeXTDataset
from synth_setter.data.vst.shapes import MEL_SPEC_FIELD

logger = logging.getLogger(__name__)

_SHARD_GLOB = "shard-*.tar"
_MEL_SPEC_MEMBER_RE = re.compile(rf"^\d{{8}}\.{re.escape(MEL_SPEC_FIELD)}\.npy$")

# Cap the listed degenerate indices in error/warning messages so a fully
# degenerate dataset (e.g. count==1 on a 3-D mel spec, ~100k elements) doesn't
# emit a megabyte-scale message. Remaining count is summarised as "+N more".
_MAX_DEGENERATE_INDEX_PREVIEW = 20


class _DegenerateBinsFound(NamedTuple):
    # Bundle of degenerate-bin info shared between ``_check_degenerate_bins`` and
    # ``_fix_degenerate_bins`` — see ``_locate_degenerate_bins`` for population.
    std: np.ndarray
    mask: np.ndarray
    n_degenerate: int
    preview: list
    overflow_suffix: str


def _locate_degenerate_bins(std: np.ndarray) -> _DegenerateBinsFound | None:
    """Find and format degenerate (``std==0``) positions for error/warning rendering.

    Internal helper shared between ``_check_degenerate_bins`` (which raises on the
    result) and ``_fix_degenerate_bins`` (which substitutes ``std=1.0`` and warns).
    Always raises on 0-d std: when ``finalize()`` reduces to a scalar variance
    (Welford state with <=1 samples), there is no per-bin shape to either check
    or mask.

    :param std: Array-like per-bin standard deviation. Cast through ``np.asarray``
        so torch tensors (returned by datamodule ``__getitem__``) and other
        array-likes go through numpy's ``argwhere`` rather than their own
        framework's ``nonzero`` delegate.

    :returns: ``None`` if no bins are degenerate; otherwise a
        :class:`_DegenerateBinsFound` carrying the canonicalized array, boolean
        mask, count, truncated preview, and ``"; +N more"`` overflow suffix.
    :rtype: _DegenerateBinsFound | None

    :raises ValueError: When ``std`` is 0-d (count<=1 datasets); a generic
        degenerate-bin report makes no sense in that case.
    """
    std = np.asarray(std)
    if std.ndim == 0:
        raise ValueError(
            "stats reduce to a scalar (likely a dataset with <=1 samples); "
            "cannot compute per-bin std. Need at least 2 samples."
        )
    mask = std == 0
    n_degenerate = int(mask.sum())
    if n_degenerate == 0:
        return None
    # Slice the index array before ``.tolist()`` so a fully-degenerate ~100k-
    # element mel doesn't allocate a 100k-tuple Python list just to print 20.
    # 1-D std (unit tests, simple flattened layouts) yields bin indices;
    # N-D std (real Surge mel: (channels, mels, frames); audio: (mels,
    # frames)) yields one coordinate tuple per degenerate element — first-
    # axis-only indexing would collapse element coordinates to channel/row
    # indices and lose the bin location.
    if std.ndim == 1:
        preview = np.flatnonzero(mask)[:_MAX_DEGENERATE_INDEX_PREVIEW].tolist()
    else:
        preview = np.argwhere(mask)[:_MAX_DEGENERATE_INDEX_PREVIEW].tolist()
    overflow = n_degenerate - _MAX_DEGENERATE_INDEX_PREVIEW
    suffix = f"; +{overflow} more" if overflow > 0 else ""
    return _DegenerateBinsFound(std, mask, n_degenerate, preview, suffix)


def _check_degenerate_bins(std: np.ndarray) -> None:
    """Raise if any entry of ``std`` is zero (or ``std`` is 0-d from a <=1-sample dataset).

    Pure check: does not mutate the input. Used by the default
    (``mask_degenerate=False``) path; pair with :func:`_fix_degenerate_bins` for
    the opt-in masking path.

    :param std: Per-bin standard deviation array.

    :raises ValueError: When ``std`` is 0-d, or any entry is zero. The message
        lists the degenerate bin indices (truncated past
        ``_MAX_DEGENERATE_INDEX_PREVIEW``).
    """
    found = _locate_degenerate_bins(std)
    if found is None:
        return
    raise ValueError(
        f"Found {found.n_degenerate} mel bin(s) with zero variance across the "
        f"dataset (std shape {found.std.shape}; indices "
        f"{found.preview}{found.overflow_suffix}). This usually indicates an "
        f"upstream problem (silence-dominated data, mel filterbank above "
        f"Nyquist, or a dataset too small to vary these bins). Rerun with "
        f"--mask-degenerate-bins to mask these bins instead of failing."
    )


def _fix_degenerate_bins(std: np.ndarray) -> np.ndarray:
    """Substitute ``std=1.0`` at degenerate positions and return the patched array.

    Pairs with :func:`_check_degenerate_bins` for the ``--mask-degenerate-bins``
    path. Because Welford's ``mean`` converges to the constant value for any bin
    that was constant during stat collection, downstream ``(spec - mean) / std``
    at training/eval time yields ``(constant - constant) / 1.0 = 0`` — equivalent
    to a constant-zero mask on in-distribution data, with no datamodule changes.

    Raises on 0-d ``std`` (count<=1 datasets) via :func:`_locate_degenerate_bins`,
    since substituting a scalar makes no sense.

    :param std: Per-bin standard deviation array.

    :returns: A new array of the same dtype with degenerate positions set to
        ``1.0`` (in the input's dtype). Returns the canonicalized input
        unchanged when no bins are degenerate.
    :rtype: np.ndarray
    """
    found = _locate_degenerate_bins(std)
    if found is None:
        return np.asarray(std)
    logger.warning(
        "Masking %d degenerate mel bin(s) with std=1.0 (std shape %s; indices %s%s).",
        found.n_degenerate,
        found.std.shape,
        found.preview,
        found.overflow_suffix,
    )
    # Preserve std's dtype: ``np.where(mask, 1.0, std)`` would promote
    # float32 → float64 from the Python literal and silently inflate
    # stats.npz on disk + change downstream dtypes.
    out = found.std.copy()
    out[found.mask] = found.std.dtype.type(1)
    return out


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

    print("Saving to file...")
    out_file = SurgeXTDataset.get_stats_file_path(filename)
    mean = mean_val.compute()
    std = std_val.compute()
    if mask_degenerate:
        std = _fix_degenerate_bins(std)
    else:
        _check_degenerate_bins(std)
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
    if mask_degenerate:
        std = _fix_degenerate_bins(std)
    else:
        _check_degenerate_bins(std)
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
            logger.info(f"Processed {i + 1} files...")

    mean, std = finalize(existing, mask_degenerate=mask_degenerate)

    logger.info(f"Saving to {str(out_file)}")

    np.savez(out_file, mean=mean, std=std)


def _iter_mel_batches(shard_path: Path) -> Iterator[np.ndarray]:
    """Yield each ``mel_spec.npy`` member's array from one shard, in name-sorted order.

    Mirrors ``validate_shard``'s raw-``tarfile`` idiom so the stats path
    pulls in no extra runtime deps (no ``webdataset``) and surfaces tar
    parse errors at the same layer as the validator. Ignores the writer's
    ``metadata.json`` sentinel and other per-row fields.

    Iterates sorted ``TarInfo`` objects (not member names) and extracts
    by the ``TarInfo`` itself so a malformed archive with duplicate
    matching names cannot trick ``extractfile`` into returning the same
    payload twice (tar name lookup resolves to the *last* matching entry
    regardless of which one we're iterating).

    :param shard_path: Filesystem path to one ``shard-*.tar``.
    :yields: One ``(rows, *inner)`` mel array per matched tar member.
    :ytype: np.ndarray
    :raises ValueError: When a matched ``*.mel_spec.npy`` member is not a
        regular file (directory, symlink, hardlink, device, etc.). Hard-
        and symlinks pointing at another archive member would satisfy
        ``extractfile != None``, so the check uses ``TarInfo.isfile()``
        which only returns ``True`` for ``REGTYPE``/``AREGTYPE``. Treated
        as a malformed shard so the per-shard guard in
        :func:`get_stats_wds` cannot be defeated by a mix of readable and
        unreadable matched members on the same shard.
    """
    with tarfile.open(shard_path, mode="r:") as tar:
        matched = sorted(
            (m for m in tar.getmembers() if _MEL_SPEC_MEMBER_RE.match(m.name)),
            key=lambda m: m.name,
        )
        for member in matched:
            if not member.isfile():
                raise ValueError(
                    f"shard {shard_path.name}: matched member {member.name!r} is "
                    f"not a regular file (TarInfo.isfile() is False; rejects "
                    f"directories, symlinks, hardlinks, and devices); treat as "
                    f"malformed shard rather than silently skip"
                )
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(
                    f"shard {shard_path.name}: matched member {member.name!r} "
                    f"could not be extracted (tarfile.extractfile returned None)"
                )
            yield np.load(io.BytesIO(extracted.read()))


def get_stats_wds(directory: str | Path, mask_degenerate: bool = False) -> None:
    """Compute mel-spec mean/std across all ``shard-*.tar`` in ``directory`` and write sibling
    ``stats.npz``.

    Streams Welford's algorithm row-by-row over each shard's pre-computed
    ``mel_spec`` arrays — no mel recompute, no full-dataset load.

    :param directory: Path to a directory containing ``shard-*.tar`` shards.
    :param mask_degenerate: If ``True``, mel bins with zero variance are
        masked to ``std=1.0`` instead of raising — see the matching flag
        on :func:`get_stats_hdf5` / :func:`get_stats_directory` for the
        downstream rationale.
    :raises FileNotFoundError: When ``directory`` contains no shards.
    :raises ValueError: When a matched shard has zero readable
        ``mel_spec.npy`` members — raised eagerly so partial stats are
        never written silently for a truncated/malformed shard.
    :returns: ``None``. Writes ``stats.npz`` to ``directory / "stats.npz"``.
    :rtype: None
    """
    directory = Path(directory)
    shard_paths = sorted(directory.glob(_SHARD_GLOB))
    if not shard_paths:
        raise FileNotFoundError(f"no {_SHARD_GLOB} files in {directory}")
    out_file = directory / "stats.npz"

    existing = (0, 0, 0)
    for shard_path in shard_paths:
        logger.info(f"Processing {shard_path.name}...")
        shard_rows = 0
        for mel_batch in _iter_mel_batches(shard_path):
            for row in mel_batch:
                existing = update(existing, row)
                shard_rows += 1
        if shard_rows == 0:
            raise ValueError(
                f"shard {shard_path.name} contained no readable "
                f"'*.{MEL_SPEC_FIELD}.npy' members; aborting so partial stats "
                f"are never written silently for a truncated/malformed shard"
            )

    mean, std = finalize(existing, mask_degenerate=mask_degenerate)

    logger.info(f"Saving to {out_file}")
    np.savez(out_file, mean=mean, std=std)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean/std statistics over a Surge XT HDF5 file, a "
            "directory of WebDataset shard-*.tar files, or an audio folder, "
            "and write the result to a sibling stats.npz."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Path to a .h5 file (Dask path), a directory containing "
            "shard-*.tar shards (streaming Welford path), or a directory of "
            "audio files (streaming Welford path)."
        ),
    )
    parser.add_argument(
        "--mask-degenerate-bins",
        action="store_true",
        help=(
            "If set, mel bins with zero variance across the dataset are masked "
            "by substituting std=1.0 at those positions. Because the mean for a "
            "constant bin converges to that constant value, downstream "
            "(spec - mean) / std then yields 0 on the training distribution. "
            "Default is to raise so degenerate bins are surfaced explicitly."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and dispatch to the matching stats entrypoint.

    Dispatch order: ``.h5`` suffix → :func:`get_stats_hdf5`; directory
    containing ``shard-*.tar`` → :func:`get_stats_wds`; everything else →
    :func:`get_stats_directory` (audio folder).

    :param argv: Argument list forwarded to ``argparse``. ``None`` uses
        ``sys.argv[1:]`` — the standard CLI behavior.
    :returns: ``None``.
    :rtype: None
    """
    args = _parse_args(argv)

    # Without this the stdlib root logger stays at WARNING and the per-file
    # progress + "Saving to..." messages emitted by get_stats_*() are silently
    # dropped, leaving the operator staring at a blank terminal during long
    # runs over thousands of files.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    if input_path.suffix == ".h5":
        get_stats_hdf5(args.input, mask_degenerate=args.mask_degenerate_bins)
    elif input_path.is_dir() and any(input_path.glob(_SHARD_GLOB)):
        get_stats_wds(args.input, mask_degenerate=args.mask_degenerate_bins)
    else:
        get_stats_directory(args.input, mask_degenerate=args.mask_degenerate_bins)


if __name__ == "__main__":
    main()
