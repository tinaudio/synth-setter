"""Permute rendered ``pred.wav`` across sample dirs to probe render-order effects.

Reassigns each ``sample_*/pred.wav`` to a different sample dir while leaving
``target.wav`` and ``params.csv`` in place, so recomputed metrics score each
target against a pred rendered from identical params but a different render-order
position. Only meaningful when every sample dir renders identical params, so the
shuffle is gated on that invariant — see #489.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

_PRED_FILENAME = "pred.wav"
_PARAMS_FILENAME = "params.csv"
_SNAPSHOT_SUFFIX = ".shuffle-src"


def _sample_dirs(audio_dir: Path) -> list[Path]:
    """Find the sample dirs holding both a ``pred.wav`` and a ``params.csv``.

    :param audio_dir: Directory whose ``sample_*`` children are candidates.
    :returns: Matches sorted by path, so the permutation is stable across filesystems.
    """
    return sorted(
        d
        for d in audio_dir.glob("sample_*")
        if d.is_dir() and (d / _PRED_FILENAME).is_file() and (d / _PARAMS_FILENAME).is_file()
    )


def _assert_uniform_params(sample_dirs: list[Path]) -> None:
    """Raise unless every sample dir's ``params.csv`` equals the first dir's.

    One writer (``predict_vst_audio.params_to_csv``) emits every CSV, so exact
    ``DataFrame.equals`` comparison is safe.

    :param sample_dirs: Dirs to compare against ``sample_dirs[0]``.
    :raises ValueError: when any ``params.csv`` differs, naming the offending dir.
    """
    reference = pd.read_csv(sample_dirs[0] / _PARAMS_FILENAME, index_col=0)
    for sample_dir in sample_dirs[1:]:
        other = pd.read_csv(sample_dir / _PARAMS_FILENAME, index_col=0)
        if not reference.equals(other):
            raise ValueError(
                "shuffle_pred_audio requires identical params across all sample dirs; "
                f"{sample_dir.name}/{_PARAMS_FILENAME} differs from "
                f"{sample_dirs[0].name}/{_PARAMS_FILENAME}."
            )


def _draw_non_identity_permutation(n: int, seed: int) -> list[int]:
    """Draw a seeded permutation of ``range(n)`` that is never the identity.

    The identity would silently reproduce the unshuffled baseline, making the
    render-order probe a no-op; redraw on the same RNG until ``pred.wav`` moves.
    Expected redraws are O(1) — the identity has probability ``1 / n!``.

    :param n: Number of elements; the caller guarantees ``n >= 2``.
    :param seed: Seed for the RNG; identical seeds yield identical permutations.
    :returns: A permutation of ``range(n)`` that differs from ``list(range(n))``.
    """
    identity = list(range(n))
    rng = np.random.default_rng(seed)
    permutation = identity
    while permutation == identity:
        permutation = rng.permutation(n).tolist()
    return permutation


def shuffle_pred_audio(audio_dir: Path, seed: int) -> list[int]:  # noqa: DOC502
    """Permute ``pred.wav`` across the sample dirs in ``audio_dir`` in place.

    Fewer than two sample dirs cannot be shuffled, so the identity permutation is
    returned and no file is touched. Otherwise the params gate runs before any
    write, so a rejection leaves the tree untouched. The permutation is applied
    from per-dir ``.shuffle-src`` snapshots, so the unshuffled baseline survives a
    crash and the next run restores each ``pred.wav`` from its snapshot first.

    :param audio_dir: ``audio/`` dir of ``sample_*`` subdirs, each with a
        ``pred.wav`` and a ``params.csv``.
    :param seed: Seed for the permutation; identical seeds reproduce it.
    :returns: The permutation as ``dest_idx -> src_idx`` over the sorted sample
        dirs — sample dir ``i`` ends up holding the pred.wav of dir ``perm[i]``.
    :raises ValueError: when the params gate finds a non-uniform ``params.csv``,
        or a partial set of ``.shuffle-src`` snapshots makes the tree ambiguous.
    """
    sample_dirs = _sample_dirs(audio_dir)
    if len(sample_dirs) < 2:
        return list(range(len(sample_dirs)))

    _assert_uniform_params(sample_dirs)

    snapshots = [d / (_PRED_FILENAME + _SNAPSHOT_SUFFIX) for d in sample_dirs]
    present = [s for s in snapshots if s.exists()]
    if present and len(present) != len(snapshots):
        raise ValueError(
            f"shuffle_pred_audio found {len(present)} of {len(snapshots)} .shuffle-src "
            f"snapshots (e.g. {present[0]}) — an interrupted shuffle left an ambiguous "
            "tree. Restore each pred.wav from its snapshot or delete the snapshots before retrying."
        )
    # A full set of snapshots is left by an interrupted run and holds the pre-shuffle
    # pred.wav; restore from them so a retry starts from the baseline, not a half-shuffled tree.
    for sample_dir, snapshot in zip(sample_dirs, snapshots):
        if snapshot.exists():
            shutil.copyfile(snapshot, sample_dir / _PRED_FILENAME)

    # Snapshot every pred.wav before overwriting any, so a permuted source is read
    # from an immutable copy and the baseline is never destroyed mid-operation.
    for sample_dir, snapshot in zip(sample_dirs, snapshots):
        shutil.copyfile(sample_dir / _PRED_FILENAME, snapshot)

    permutation = _draw_non_identity_permutation(len(sample_dirs), seed)
    for dest_idx, src_idx in enumerate(permutation):
        shutil.copyfile(snapshots[src_idx], sample_dirs[dest_idx] / _PRED_FILENAME)

    # missing_ok: a successful shuffle must not raise if a snapshot is already gone.
    for snapshot in snapshots:
        snapshot.unlink(missing_ok=True)

    return permutation
