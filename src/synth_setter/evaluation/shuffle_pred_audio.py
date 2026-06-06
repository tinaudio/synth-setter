"""Build a symlinked view of rendered audio with ``pred.wav`` permuted across sample dirs.

Each ``sample_*`` child symlinks ``target.wav`` to its own target and ``pred.wav``
under a seeded non-identity permutation, so recomputed metrics score each target
against a pred rendered from identical params but a different render-order
position. The permutation moves at least one ``pred.wav`` but may keep others in
place. The source ``audio/`` dir is never modified.
Gated on uniform params across dirs — see #489.
"""

import shutil
from pathlib import Path

import numpy as np

_PRED_FILENAME = "pred.wav"
_TARGET_FILENAME = "target.wav"
_PARAMS_FILENAME = "params.csv"


def _sample_dirs(audio_dir: Path) -> list[Path]:
    """Find the sample dirs holding a ``pred.wav``, a ``target.wav``, and a ``params.csv``.

    All three are required because the shuffle symlinks both ``pred.wav`` and
    ``target.wav``; a dir missing either would yield a dangling link.

    :param audio_dir: Directory whose ``sample_*`` children are candidates.
    :returns: Matches sorted by path, so the permutation is stable across filesystems.
    """
    return sorted(
        d
        for d in audio_dir.glob("sample_*")
        if d.is_dir()
        and (d / _PRED_FILENAME).is_file()
        and (d / _TARGET_FILENAME).is_file()
        and (d / _PARAMS_FILENAME).is_file()
    )


def _assert_uniform_params(sample_dirs: list[Path]) -> None:
    """Raise unless every sample dir's ``params.csv`` equals the first dir's.

    One writer (``predict_vst_audio.params_to_csv``) emits every CSV, so exact
    text comparison is safe (identical params produce byte-identical files).

    :param sample_dirs: Dirs to compare against ``sample_dirs[0]``.
    :raises ValueError: when any ``params.csv`` differs, naming the offending dir.
    """
    reference = (sample_dirs[0] / _PARAMS_FILENAME).read_text()
    for sample_dir in sample_dirs[1:]:
        if (sample_dir / _PARAMS_FILENAME).read_text() != reference:
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


def params_are_uniform(sample_dirs: list[Path]) -> bool:
    """Return True when all ``params.csv`` files are byte-identical, or fewer than two dirs exist.

    Missing ``params.csv`` in any dir returns False — signals a non-oracle dataset where
    shuffle is not meaningful.

    :param sample_dirs: Candidate dirs to compare (typically from ``find_possible_subdirs``).
    :returns: True when uniformity holds or the list is too short to compare.
    """
    if len(sample_dirs) < 2:
        return True
    try:
        reference = (sample_dirs[0] / _PARAMS_FILENAME).read_text()
    except FileNotFoundError:
        return False
    for sample_dir in sample_dirs[1:]:
        try:
            if (sample_dir / _PARAMS_FILENAME).read_text() != reference:
                return False
        except FileNotFoundError:
            return False
    return True


def shuffle_pred_audio(audio_dir: Path, dest_dir: Path, seed: int) -> list[int]:
    """Build ``dest_dir`` as a symlinked view of ``audio_dir`` with ``pred.wav`` permuted.

    Fewer than two sample dirs cannot be shuffled, so the identity permutation is
    returned and ``dest_dir`` is left untouched — the caller falls back to
    ``audio_dir``. Otherwise the params gate runs before any write, so a rejection
    leaves the filesystem untouched. Each ``dest_dir/sample_i`` holds two symlinks
    to absolute source paths: ``target.wav`` to its own sample's target and
    ``pred.wav`` to the pred of the permuted source dir. The source tree is never
    modified.

    When the shuffle proceeds, a pre-existing ``dest_dir`` is cleared before the
    rebuild — a symlink or file is unlinked, a real directory removed wholesale.
    Because the dest tree holds only symlinks, clearing never touches the real
    audio it points at.

    :param audio_dir: ``audio/`` dir of ``sample_*`` subdirs, each with a
        ``pred.wav``, a ``target.wav``, and a ``params.csv``.
    :param dest_dir: Directory to build; must not be inside ``audio_dir``, so
        that clearing or building it cannot mutate the source tree.
    :param seed: Seed for the permutation; identical seeds reproduce it.
    :returns: The permutation as ``dest_idx -> src_idx`` over the sorted sample
        dirs — ``dest_dir/sample_i/pred.wav`` links to the pred of dir ``perm[i]``.
    :raises ValueError: when ``dest_dir`` is inside ``audio_dir``, or when the
        params gate finds a non-uniform ``params.csv``.
    """
    resolved_audio = audio_dir.resolve()
    resolved_dest = dest_dir.resolve()
    if resolved_dest == resolved_audio or resolved_audio in resolved_dest.parents:
        raise ValueError(
            f"dest_dir ({dest_dir}) must not be inside audio_dir ({audio_dir}); "
            "building the symlink view there would mutate the source tree."
        )

    sample_dirs = _sample_dirs(audio_dir)
    if len(sample_dirs) < 2:
        return list(range(len(sample_dirs)))

    _assert_uniform_params(sample_dirs)

    if dest_dir.is_symlink() or dest_dir.is_file():
        dest_dir.unlink()
    elif dest_dir.is_dir():
        shutil.rmtree(dest_dir)

    permutation = _draw_non_identity_permutation(len(sample_dirs), seed)
    for dest_idx, src_idx in enumerate(permutation):
        out = dest_dir / sample_dirs[dest_idx].name
        out.mkdir(parents=True)
        (out / _TARGET_FILENAME).symlink_to((sample_dirs[dest_idx] / _TARGET_FILENAME).resolve())
        (out / _PRED_FILENAME).symlink_to((sample_dirs[src_idx] / _PRED_FILENAME).resolve())

    return permutation
