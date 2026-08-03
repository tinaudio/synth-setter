"""Stream dataset-level mel and cached-conditioning normalization statistics."""

import argparse
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa

from synth_setter.conditioning import EmbeddingNormalization
from synth_setter.data.audio_datamodule import AudioFolderDataset
from synth_setter.data.vst.shapes import MEL_SPEC_FIELD
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import conditioning_stats_filename

logger = logging.getLogger(__name__)

type WelfordValue = np.ndarray | int
type WelfordState = tuple[int, WelfordValue, WelfordValue]

# Lance shards are dataset directories, not single files.
_SHARD_GLOB = "shard-*.lance"

# Cap the listed degenerate indices in error/warning messages so a fully
# degenerate dataset (e.g. count==1 on a 3-D mel spec, ~100k elements) doesn't
# emit a megabyte-scale message. Remaining count is summarised as "+N more".
_MAX_DEGENERATE_INDEX_PREVIEW = 20
_CONDITIONING_DEGENERATE_STD = 1e-6


class _DegenerateBinsFound(NamedTuple):
    # Bundle of degenerate-bin info shared between ``_check_degenerate_bins`` and
    # ``_fix_degenerate_bins`` — see ``_locate_degenerate_bins`` for population.
    std: np.ndarray
    mask: np.ndarray
    n_degenerate: int
    preview: list
    overflow_suffix: str


def _locate_degenerate_bins(
    std: np.ndarray, *, threshold: float = 0.0
) -> _DegenerateBinsFound | None:
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
    :param threshold: Positive values below this standard deviation are degenerate;
        zero preserves the legacy exact-zero rule.

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
    mask = std < threshold if threshold > 0 else std == 0
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


def _fix_degenerate_bins(std: np.ndarray, *, threshold: float = 0.0) -> np.ndarray:
    """Substitute ``std=1.0`` at degenerate positions and return the patched array.

    Pairs with :func:`_check_degenerate_bins` for the ``--mask-degenerate-bins``
    path. Because Welford's ``mean`` converges to the constant value for any bin
    that was constant during stat collection, downstream ``(spec - mean) / std``
    at training/eval time yields ``(constant - constant) / 1.0 = 0`` — equivalent
    to a constant-zero mask on in-distribution data, with no datamodule changes.

    Raises on 0-d ``std`` (count<=1 datasets) via :func:`_locate_degenerate_bins`,
    since substituting a scalar makes no sense.

    :param std: Per-bin standard deviation array.
    :param threshold: Positive values below this standard deviation are degenerate;
        zero preserves the legacy exact-zero rule.

    :returns: A new array of the same dtype with degenerate positions set to
        ``1.0`` (in the input's dtype). Returns the canonicalized input
        unchanged when no bins are degenerate.
    :rtype: np.ndarray
    """
    found = _locate_degenerate_bins(std, threshold=threshold)
    if found is None:
        return np.asarray(std)
    logger.warning(
        "Masking %d degenerate bin(s) with std=1.0 (std shape %s; indices %s%s).",
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


def update(existing: WelfordState, new: np.ndarray) -> WelfordState:
    """Fold one observation into a Welford state.

    :param existing: Current ``(count, mean, M2)`` state.
    :param new: One array-shaped observation.
    :returns: Updated Welford state.
    """
    count, mean, M2 = existing
    count += 1
    delta = new - mean
    mean += delta / count
    delta2 = new - mean
    M2 += delta * delta2
    return count, mean, M2


def merge_welford(existing: WelfordState, other: WelfordState) -> WelfordState:
    """Merge two Welford states (Chan et al. parallel combine).

    Lets finalize reduce per-attempt ``(count, mean, m2)`` shard sidecars into
    one dataset-level state without touching any rows. The zero state
    ``(0, 0, 0)`` is the identity, so it seeds a fold.

    :param existing: Welford state ``(count, mean, M2)`` accumulated so far.
    :param other: Welford state to fold in.
    :returns: Combined Welford state over both inputs' rows.
    Both non-identity states must carry identically shaped mean and M2 arrays.
    """
    count_a, mean_a, m2_a = existing
    count_b, mean_b, m2_b = other
    count = count_a + count_b
    if count == 0:
        return existing
    delta = mean_b - mean_a
    mean = mean_a + delta * (count_b / count)
    m2 = m2_a + m2_b + delta * delta * (count_a * count_b / count)
    return count, mean, m2


def finalize(
    existing: WelfordState,
    mask_degenerate: bool = False,
    *,
    degenerate_threshold: float = 0.0,
    allow_single_observation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Welford state into population mean and standard deviation.

    :param existing: Completed ``(count, mean, M2)`` state.
    :param mask_degenerate: Whether to substitute unit std for constant bins.
    :param degenerate_threshold: Positive standard deviations below this value are
        degenerate when masking is enabled.
    :param allow_single_observation: Preserve the mean array's shape with zero variance
        when only one observation exists.
    :returns: Population ``(mean, std)`` arrays.
    """
    count, mean, M2 = existing
    variance = (
        M2 / count
        if count > 1
        else np.zeros_like(mean) if allow_single_observation else 0
    )
    std = np.sqrt(variance)
    if mask_degenerate:
        std = _fix_degenerate_bins(std, threshold=degenerate_threshold)
    else:
        _check_degenerate_bins(std)
    return mean, std


def get_stats_directory(
    directory: str | Path, mask_degenerate: bool = False
) -> None:
    """Write mel normalization statistics for an audio directory.

    :param directory: Audio folder consumed by :class:`AudioFolderDataset`.
    :param mask_degenerate: Whether to substitute unit std for constant bins.
    """
    dataset = AudioFolderDataset(directory)
    out_file = AudioFolderDataset.get_stats_file_path(directory)

    existing = (0, 0, 0)
    # we run Welford's online algorithm
    for i in range(len(dataset)):
        x = dataset[i]["mel"]
        existing = update(existing, x)

        if i % 10 == 0:
            logger.info(f"Processed {i + 1} files...")

    mean, std = finalize(existing, mask_degenerate=mask_degenerate)

    logger.info(f"Saving to {str(out_file)}")

    np.savez(out_file, mean=mean, std=std)


def fold_lance_shard_into_welford(
    existing: WelfordState,
    shard_uri: str | Path,
    *,
    storage_options: dict[str, str] | None = None,
) -> WelfordState:
    """Fold every Lance ``mel_spec`` row from one shard into Welford state.

    :param existing: Welford state ``(count, mean, M2)`` before this shard.
    :param shard_uri: One ``shard-*.lance`` dataset (local path or ``s3://`` URI).
    :param storage_options: Object-store config for a cloud ``shard_uri``; ``None`` local.
    :returns: Updated Welford state after every readable mel row was folded.
    :raises ValueError: The shard carried no readable ``mel_spec`` rows.
    """
    from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows

    shard_rows = 0
    for row in iter_lance_column_rows(shard_uri, MEL_SPEC_FIELD, storage_options=storage_options):
        existing = update(existing, row.astype(np.float32, copy=False))
        shard_rows += 1
    if shard_rows == 0:
        raise ValueError(
            f"shard {Path(str(shard_uri)).name} contained no readable {MEL_SPEC_FIELD!r} "
            "rows; aborting so partial stats are never written silently"
        )
    return existing


def stream_stats_lance(
    shard_uris: Iterable[str | Path],
    mask_degenerate: bool = False,
    *,
    storage_options: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream Welford mean/std across an iterable of ``shard-*.lance`` datasets.

    :param shard_uris: Iterable of Lance shard datasets in fold order (local
        paths or ``s3://`` URIs).
    :param mask_degenerate: See :func:`get_stats_lance`.
    :param storage_options: Object-store config for cloud ``shard_uris``; ``None`` local.
    :returns: ``(mean, std)`` arrays as produced by :func:`finalize`.
    :raises FileNotFoundError: ``shard_uris`` yielded zero entries.
    """
    existing: WelfordState = (0, 0, 0)
    folded_any = False
    for shard_uri in shard_uris:
        logger.info("Processing %s...", Path(str(shard_uri)).name)
        existing = fold_lance_shard_into_welford(
            existing, shard_uri, storage_options=storage_options
        )
        folded_any = True
    if not folded_any:
        raise FileNotFoundError("stream_stats_lance received no shard URIs")
    return finalize(existing, mask_degenerate=mask_degenerate)


def _conditioning_observations(
    values: np.ndarray,
    *,
    input_shape: tuple[int, ...],
    normalization: EmbeddingNormalization,
) -> np.ndarray:
    """Arrange one Lance batch as observations for its configured affine.

    :param values: Stored values shaped ``(rows, *input_shape)``.
    :param input_shape: Fixed per-row conditioning shape.
    :param normalization: Per-channel or shared-global strategy.
    :returns: Float64 observations shaped ``(N, D)`` or ``(N, 1)``.
    :raises ValueError: Values have the wrong shape, rank, or non-finite entries.
    """
    if values.shape[1:] != input_shape:
        raise ValueError(
            f"conditioning batch has shape {values.shape[1:]}, expected {input_shape}"
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"conditioning values must be floating-point, got {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError("conditioning values contain non-finite entries")
    if normalization == "global":
        return values.reshape(-1, 1).astype(np.float64, copy=False)
    if normalization != "per_channel":
        raise ValueError(f"cannot compute conditioning statistics for {normalization!r}")
    if len(input_shape) == 1:
        observations = values
    elif len(input_shape) == 2:
        observations = values.transpose(0, 2, 1).reshape(-1, input_shape[0])
    else:
        raise ValueError(
            "per-channel conditioning statistics require vector [D] or sequence [D, T] values"
        )
    return observations.astype(np.float64, copy=False)


def _conditioning_array_to_numpy(
    array: pa.Array, input_shape: tuple[int, ...]
) -> np.ndarray:
    """Decode fixed-shape-tensor or fixed-size-list conditioning rows.

    :param array: Arrow batch column from Lance.
    :param input_shape: Registered row shape.
    :returns: Dense rows shaped ``(batch, *input_shape)``.
    :raises ValueError: Values are null or use unsupported storage.
    """
    current = array.storage if isinstance(array, pa.ExtensionArray) else array
    while pa.types.is_fixed_size_list(current.type):
        if current.null_count:
            raise ValueError("conditioning values contain null entries")
        current = current.flatten()
    if current.null_count:
        raise ValueError("conditioning values contain null entries")

    to_tensor = getattr(array, "to_numpy_ndarray", None)
    if callable(to_tensor):
        return np.asarray(to_tensor())
    if not pa.types.is_fixed_size_list(array.type):
        raise ValueError(f"unsupported conditioning Arrow type {array.type}")
    values = current.to_numpy(zero_copy_only=False)
    return np.asarray(values).reshape(len(array), *input_shape)


def stream_conditioning_stats_lance(
    dataset_uri: str | Path,
    *,
    column: str,
    input_shape: tuple[int, ...],
    normalization: EmbeddingNormalization,
    storage_options: dict[str, str] | None = None,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream cached-conditioning affine statistics from one Lance dataset.

    Per-channel sequence statistics aggregate each ``[D]`` channel over rows
    and time. Global statistics aggregate every stored scalar into one affine.

    :param dataset_uri: Finalized Lance split containing the cached column.
    :param column: Fixed-shape cached-conditioning column.
    :param input_shape: Stored per-row shape, either ``[D]`` or ``[D, T]``.
    :param normalization: ``per_channel`` or ``global``.
    :param storage_options: Object-store options for a cloud dataset.
    :param batch_size: Lance scan batch size in rows.
    :returns: Float32 ``(mean, std)`` arrays; global arrays have shape ``(1,)``.
    :raises ValueError: The dataset is empty or violates the numeric/shape contract.
    """
    import lance

    dataset_target = str(dataset_uri)
    dataset_storage_options = storage_options
    if r2_io.is_r2_uri(dataset_target):
        dataset_target, dataset_storage_options = r2_io.lance_target(dataset_target)
    dataset = lance.dataset(dataset_target, storage_options=dataset_storage_options)
    state: WelfordState = (0, 0, 0)
    for batch in dataset.to_batches(columns=[column], batch_size=batch_size):
        values = _conditioning_array_to_numpy(batch.column(0), input_shape)
        observations = _conditioning_observations(
            values,
            input_shape=input_shape,
            normalization=normalization,
        )
        batch_mean = observations.mean(axis=0)
        centered = observations - batch_mean
        batch_state: WelfordState = (
            len(observations),
            batch_mean,
            np.sum(centered * centered, axis=0),
        )
        state = merge_welford(state, batch_state)
    if state[0] == 0:
        raise ValueError(f"conditioning column {column!r} contained no rows")
    mean, std = finalize(
        state,
        mask_degenerate=True,
        degenerate_threshold=_CONDITIONING_DEGENERATE_STD,
        allow_single_observation=True,
    )
    return mean.astype(np.float32), std.astype(np.float32)


def _assert_conditioning_stats_match(
    stats_file: Path, mean: np.ndarray, std: np.ndarray
) -> None:
    """Reject an existing artifact whose affine differs from this computation.

    :param stats_file: Existing column-specific archive.
    :param mean: Newly computed mean.
    :param std: Newly computed standard deviation.
    :raises FileExistsError: Existing and new affine arrays differ.
    """
    with np.load(stats_file) as existing:
        matches = (
            set(existing.files) == {"mean", "std"}
            and np.array_equal(existing["mean"], mean)
            and np.array_equal(existing["std"], std)
        )
    if not matches:
        raise FileExistsError(f"new conditioning statistics conflict with existing {stats_file}")


def _write_conditioning_stats(  # noqa: DOC502
    stats_file: Path, mean: np.ndarray, std: np.ndarray
) -> None:
    """Atomically publish one column's immutable statistics artifact.

    :param stats_file: Column-specific archive beside finalized split datasets.
    :param mean: Affine mean array.
    :param std: Affine standard-deviation array.
    :raises FileExistsError: An existing artifact contains a different affine.
    """
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    if stats_file.exists():
        _assert_conditioning_stats_match(stats_file, mean, std)
        return
    with tempfile.NamedTemporaryFile(
        dir=stats_file.parent,
        prefix=f".{stats_file.name}.",
        suffix=".npz",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez(temporary_path, mean=mean, std=std)
        try:
            os.link(temporary_path, stats_file)
        except FileExistsError:
            _assert_conditioning_stats_match(stats_file, mean, std)
    finally:
        temporary_path.unlink(missing_ok=True)


def get_conditioning_stats_lance(
    dataset_uri: str | Path,
    *,
    column: str,
    input_shape: tuple[int, ...],
    normalization: EmbeddingNormalization,
    output_directory: str | Path | None = None,
    storage_options: dict[str, str] | None = None,
    batch_size: int = 16,
) -> Path | str:
    """Write one column's streaming statistics to its immutable archive.

    :param dataset_uri: Finalized Lance split containing the cached column.
    :param column: Fixed-shape cached-conditioning column.
    :param input_shape: Stored per-row shape.
    :param normalization: ``per_channel`` or ``global``.
    :param output_directory: Local archive directory; defaults beside local or R2 input.
    :param storage_options: Object-store options for a cloud dataset.
    :param batch_size: Lance scan batch size in rows.
    :returns: Local path or canonical R2 URI of the column-specific archive.
    """
    mean, std = stream_conditioning_stats_lance(
        dataset_uri,
        column=column,
        input_shape=input_shape,
        normalization=normalization,
        storage_options=storage_options,
        batch_size=batch_size,
    )
    filename = conditioning_stats_filename(column)
    dataset_uri_string = str(dataset_uri)
    if output_directory is not None:
        stats_file = Path(output_directory) / filename
        _write_conditioning_stats(stats_file, mean, std)
        logger.info("Saving conditioning statistics to %s", stats_file)
        return stats_file
    if not r2_io.is_r2_uri(dataset_uri_string):
        stats_file = Path(dataset_uri).parent / filename
        _write_conditioning_stats(stats_file, mean, std)
        logger.info("Saving conditioning statistics to %s", stats_file)
        return stats_file

    destination = f"{dataset_uri_string.rsplit('/', 1)[0]}/{filename}"
    with tempfile.TemporaryDirectory(prefix="conditioning-stats-") as temporary:
        stats_file = Path(temporary) / filename
        _write_conditioning_stats(stats_file, mean, std)
        if r2_io.object_size(destination) is not None:
            existing_file = Path(temporary) / f"existing-{filename}"
            r2_io.download_to_path(destination, existing_file)
            _assert_conditioning_stats_match(existing_file, mean, std)
            return destination
        r2_io.upload_to_uri_immutable(stats_file, destination)
    logger.info("Saving conditioning statistics to %s", destination)
    return destination


def get_stats_lance(directory: str | Path, mask_degenerate: bool = False) -> None:
    """Compute mel-spec mean/std over ``shard-*.lance`` shards and write a sibling ``stats.npz``.

    Streams Welford's algorithm row-by-row over each shard's pre-computed
    ``mel_spec`` column — no mel recompute, no full-dataset load.

    :param directory: Path to a directory containing ``shard-*.lance`` datasets.
    :param mask_degenerate: If ``True``, mel bins with zero variance are
        masked to ``std=1.0`` instead of raising — see the matching flag
        on :func:`get_stats_directory` for the downstream rationale.
    :raises FileNotFoundError: When ``directory`` contains no shards.
    :returns: ``None``. Writes ``stats.npz`` to ``directory / "stats.npz"``.
    """
    directory = Path(directory)
    shard_paths = sorted(directory.glob(_SHARD_GLOB))
    if not shard_paths:
        raise FileNotFoundError(f"no {_SHARD_GLOB} files in {directory}")
    mean, std = stream_stats_lance(shard_paths, mask_degenerate=mask_degenerate)
    out_file = directory / "stats.npz"
    logger.info("Saving to %s", out_file)
    np.savez(out_file, mean=mean, std=std)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean/std statistics over a directory of Lance shard-*.lance "
            "datasets or an audio folder, and write the result to a sibling stats.npz."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Path to a directory containing shard-*.lance datasets (streaming "
            "Welford path), or a directory of audio files (streaming Welford path)."
        ),
    )
    parser.add_argument(
        "--conditioning-column",
        help="Cached Lance column for column-specific conditioning statistics.",
    )
    parser.add_argument(
        "--conditioning-shape",
        nargs="+",
        type=int,
        help="Fixed conditioning row shape, e.g. --conditioning-shape 256 44.",
    )
    parser.add_argument(
        "--conditioning-normalization",
        choices=("per_channel", "global"),
        help="Conditioning affine to compute.",
    )
    parser.add_argument(
        "--conditioning-batch-size",
        type=int,
        default=16,
        help="Lance rows scanned per conditioning-statistics batch.",
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

    Dispatch order: a directory containing ``shard-*.lance`` datasets →
    :func:`get_stats_lance`; everything else → :func:`get_stats_directory`
    (audio folder).

    :param argv: Argument list forwarded to ``argparse``. ``None`` uses
        ``sys.argv[1:]`` — the standard CLI behavior.
    :returns: ``None``.
    :rtype: None
    :raises ValueError: Conditioning-stat flags are incomplete.
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

    if args.conditioning_column is not None:
        if args.conditioning_shape is None or args.conditioning_normalization is None:
            raise ValueError(
                "--conditioning-column requires --conditioning-shape and "
                "--conditioning-normalization"
            )
        get_conditioning_stats_lance(
            args.input,
            column=args.conditioning_column,
            input_shape=tuple(args.conditioning_shape),
            normalization=args.conditioning_normalization,
            batch_size=args.conditioning_batch_size,
        )
        return

    input_path = Path(args.input)
    if input_path.is_dir() and any(input_path.glob(_SHARD_GLOB)):
        get_stats_lance(args.input, mask_degenerate=args.mask_degenerate_bins)
    else:
        get_stats_directory(args.input, mask_degenerate=args.mask_degenerate_bins)


if __name__ == "__main__":
    main()
