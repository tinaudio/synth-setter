"""Behavior tests for ``synth_setter.evaluation.shuffle_pred_audio``.

Drive the real function against real files in ``tmp_path`` — no mocks — and
assert on the observable filesystem state and the returned permutation.
"""

from pathlib import Path

import pytest

from synth_setter.evaluation.shuffle_pred_audio import shuffle_pred_audio

_UNIFORM_PARAMS_CSV = ",pred,target\ncutoff,0.5,0.5\nresonance,0.2,0.2\n"
# A params.csv differing from _UNIFORM_PARAMS_CSV, used to trip the uniform-params gate.
_DIFFERENT_PARAMS_CSV = ",pred,target\ncutoff,0.9,0.9\n"

# Default sample-dir count and seed shared by the behaviour tests. Individual
# tests override only for a boundary cardinality (a pair, a single dir, or none),
# which is the behaviour under test rather than an arbitrary value.
_SAMPLE_COUNT = 6
_PAIR = 2
_SEED = 42


def _make_sample_dir(
    audio_dir: Path,
    index: int,
    *,
    params_csv: str = _UNIFORM_PARAMS_CSV,
) -> Path:
    """Create ``audio_dir/sample_<index>`` with a uniquely-tagged pred.wav.

    :param audio_dir: Parent ``audio/`` directory.
    :param index: Sample index; ``pred.wav`` is tagged ``pred-<index>`` so a
        permutation is observable by reading bytes back.
    :param params_csv: Body written to ``params.csv``; vary it to break the gate.
    :returns: The created sample directory.
    """
    sample_dir = audio_dir / f"sample_{index}"
    sample_dir.mkdir(parents=True)
    (sample_dir / "pred.wav").write_bytes(f"pred-{index}".encode())
    (sample_dir / "target.wav").write_bytes(f"target-{index}".encode())
    (sample_dir / "params.csv").write_text(params_csv)
    return sample_dir


def _build_audio_dir(base: Path, n: int = _SAMPLE_COUNT) -> Path:
    """Create ``base/audio`` with ``n`` uniform-params sample dirs.

    :param base: Directory under which ``audio/`` is created.
    :param n: Number of ``sample_<i>`` dirs to create.
    :returns: The created ``audio/`` directory.
    """
    audio_dir = base / "audio"
    audio_dir.mkdir(parents=True)
    for i in range(n):
        _make_sample_dir(audio_dir, i)
    return audio_dir


def test_shuffle_pred_audio_reassigns_pred_to_permuted_source(tmp_path: Path) -> None:
    """Each sample dir ends up holding the pred.wav of its permuted source dir.

    :param tmp_path: Holds the ``audio/`` tree the shuffle rewrites.
    """
    audio_dir = _build_audio_dir(tmp_path)
    original = {
        i: (audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)
    }

    permutation = shuffle_pred_audio(audio_dir, seed=_SEED)

    for dest_idx, src_idx in enumerate(permutation):
        assert (audio_dir / f"sample_{dest_idx}" / "pred.wav").read_bytes() == original[src_idx]


def test_shuffle_pred_audio_preserves_every_pred_file(tmp_path: Path) -> None:
    """No pred.wav is lost or duplicated — the multiset of contents is preserved.

    :param tmp_path: Holds the ``audio/`` tree the shuffle rewrites.
    """
    audio_dir = _build_audio_dir(tmp_path)
    before = sorted(
        (audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)
    )

    shuffle_pred_audio(audio_dir, seed=_SEED)

    after = sorted(
        (audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)
    )
    assert after == before


def test_shuffle_pred_audio_changes_pred_arrangement(tmp_path: Path) -> None:
    """A non-trivial seed actually reorders the pred.wav files across dirs.

    :param tmp_path: Holds the ``audio/`` tree the shuffle rewrites.
    """
    audio_dir = _build_audio_dir(tmp_path)
    before = [(audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)]

    permutation = shuffle_pred_audio(audio_dir, seed=_SEED)

    after = [(audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)]
    assert after != before
    assert permutation != list(range(_SAMPLE_COUNT))


def test_shuffle_pred_audio_never_returns_identity_for_two_dirs(tmp_path: Path) -> None:
    """With two dirs the only non-identity permutation is forced, regardless of seed.

    Guards the render-order probe against silently degenerating into the
    unshuffled baseline when a seed would otherwise draw the identity.

    :param tmp_path: Holds the two-dir ``audio/`` tree the shuffle rewrites.
    """
    audio_dir = _build_audio_dir(tmp_path, _PAIR)

    permutation = shuffle_pred_audio(audio_dir, seed=_SEED)

    assert permutation == [1, 0]


def test_shuffle_pred_audio_recovers_baseline_from_full_snapshot_set(tmp_path: Path) -> None:
    """A full set of ``.shuffle-src`` snapshots restores the baseline before re-shuffling.

    Simulates an interrupted run: each pred.wav holds half-overwritten junk while
    its snapshot holds the true baseline. The retry must discard the junk, recover
    the baseline from the snapshots, and permute that — yielding the clean result.

    :param tmp_path: Holds the ``audio/`` tree seeded with a full snapshot set.
    """
    audio_dir = _build_audio_dir(tmp_path, _PAIR)
    for i in range(_PAIR):
        (audio_dir / f"sample_{i}" / "pred.wav").write_bytes(f"junk-{i}".encode())
        (audio_dir / f"sample_{i}" / "pred.wav.shuffle-src").write_bytes(f"pred-{i}".encode())

    shuffle_pred_audio(audio_dir, seed=_SEED)

    assert (audio_dir / "sample_0" / "pred.wav").read_bytes() == b"pred-1"
    assert (audio_dir / "sample_1" / "pred.wav").read_bytes() == b"pred-0"


def test_shuffle_pred_audio_raises_on_partial_snapshot_set(tmp_path: Path) -> None:
    """A partial ``.shuffle-src`` set is ambiguous, so the shuffle refuses to run.

    :param tmp_path: Holds the ``audio/`` tree with a snapshot in only one dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    (audio_dir / "sample_1" / "pred.wav.shuffle-src").write_bytes(b"orphan")
    before = [(audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)]

    with pytest.raises(ValueError, match="ambiguous"):
        shuffle_pred_audio(audio_dir, seed=_SEED)

    after = [(audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)]
    assert after == before


def test_shuffle_pred_audio_removes_snapshots_after_run(tmp_path: Path) -> None:
    """A completed shuffle leaves no ``.shuffle-src`` snapshot files behind.

    :param tmp_path: Holds the ``audio/`` tree the shuffle rewrites.
    """
    audio_dir = _build_audio_dir(tmp_path)

    shuffle_pred_audio(audio_dir, seed=_SEED)

    assert list(audio_dir.glob("*/pred.wav.shuffle-src")) == []


def test_shuffle_pred_audio_same_seed_yields_same_permutation(tmp_path: Path) -> None:
    """The permutation is deterministic for a fixed seed.

    :param tmp_path: Parents two independent ``audio/`` trees shuffled with one seed.
    """
    audio_dir_a = _build_audio_dir(tmp_path / "a")
    audio_dir_b = _build_audio_dir(tmp_path / "b")

    perm_a = shuffle_pred_audio(audio_dir_a, seed=_SEED)
    perm_b = shuffle_pred_audio(audio_dir_b, seed=_SEED)

    assert perm_a == perm_b


def test_shuffle_pred_audio_leaves_target_wav_in_place(tmp_path: Path) -> None:
    """Only pred.wav moves; each target.wav stays in its own dir.

    :param tmp_path: Holds the ``audio/`` tree the shuffle rewrites.
    """
    audio_dir = _build_audio_dir(tmp_path)

    shuffle_pred_audio(audio_dir, seed=_SEED)

    for i in range(_SAMPLE_COUNT):
        assert (audio_dir / f"sample_{i}" / "target.wav").read_bytes() == f"target-{i}".encode()


def test_shuffle_pred_audio_raises_when_params_differ(tmp_path: Path) -> None:
    """The gate refuses to shuffle when a sample dir's params.csv differs.

    :param tmp_path: Holds the ``audio/`` tree with one mismatched params.csv.
    """
    audio_dir = _build_audio_dir(tmp_path)
    (audio_dir / "sample_2" / "params.csv").write_text(_DIFFERENT_PARAMS_CSV)

    with pytest.raises(ValueError, match="identical params"):
        shuffle_pred_audio(audio_dir, seed=_SEED)


def test_shuffle_pred_audio_does_not_move_files_when_params_differ(tmp_path: Path) -> None:
    """A gate rejection leaves every pred.wav untouched (validate before moving).

    :param tmp_path: Holds the ``audio/`` tree with one mismatched params.csv.
    """
    audio_dir = _build_audio_dir(tmp_path)
    (audio_dir / "sample_2" / "params.csv").write_text(_DIFFERENT_PARAMS_CSV)
    before = [(audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)]

    with pytest.raises(ValueError):
        shuffle_pred_audio(audio_dir, seed=_SEED)

    after = [(audio_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)]
    assert after == before


def test_shuffle_pred_audio_single_dir_is_noop(tmp_path: Path) -> None:
    """One sample dir cannot be shuffled — return identity, leave the file in place.

    :param tmp_path: Holds the single-dir ``audio/`` tree.
    """
    audio_dir = _build_audio_dir(tmp_path, 1)

    permutation = shuffle_pred_audio(audio_dir, seed=_SEED)

    assert permutation == [0]
    assert (audio_dir / "sample_0" / "pred.wav").read_bytes() == b"pred-0"


def test_shuffle_pred_audio_no_samples_returns_empty(tmp_path: Path) -> None:
    """An audio dir with no sample dirs yields an empty permutation.

    :param tmp_path: Parents an empty ``audio/`` directory.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    assert shuffle_pred_audio(audio_dir, seed=_SEED) == []


def test_shuffle_pred_audio_ignores_non_sample_dirs(tmp_path: Path) -> None:
    """Only ``sample_*`` dirs count, even when a stray dir holds both files.

    :param tmp_path: Holds an ``audio/`` tree with the default samples plus a
        non-sample dir that has a pred.wav + params.csv (must still be excluded).
    """
    audio_dir = _build_audio_dir(tmp_path)
    stray = audio_dir / "metrics_scratch"
    stray.mkdir()
    (stray / "pred.wav").write_bytes(b"stray")
    (stray / "params.csv").write_text(_UNIFORM_PARAMS_CSV)

    permutation = shuffle_pred_audio(audio_dir, seed=_SEED)

    assert len(permutation) == _SAMPLE_COUNT
    assert (stray / "pred.wav").read_bytes() == b"stray"
