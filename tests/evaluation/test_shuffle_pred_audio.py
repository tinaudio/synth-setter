"""Behavior tests for ``synth_setter.evaluation.shuffle_pred_audio``.

Drive the real function against real files in ``tmp_path`` — no mocks — and
assert on the observable filesystem state and the returned permutation. The
shuffle builds a *symlink* tree, so the source ``audio/`` dir must survive every
run untouched.
"""

from pathlib import Path

import pytest

from synth_setter.evaluation.shuffle_pred_audio import shuffle_pred_audio

_UNIFORM_PARAMS_CSV = ",pred,target\ncutoff,0.5,0.5\nresonance,0.2,0.2\n"
# A params.csv differing from _UNIFORM_PARAMS_CSV, used to trip the uniform-params gate.
_DIFFERENT_PARAMS_CSV = ",pred,target\ncutoff,0.9,0.9\n"

# Shared defaults; individual tests override only for a boundary cardinality
# (pair, single, none) — that cardinality is the behaviour under test.
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
        permutation is observable by reading bytes back through the symlink.
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


def test_shuffle_pred_audio_builds_symlink_dir_with_permuted_pred(tmp_path: Path) -> None:
    """Each dest sample dir's pred.wav resolves to the permuted source's real pred.

    :param tmp_path: Parents the source ``audio/`` tree and the dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"

    permutation = shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    for dest_idx, src_idx in enumerate(permutation):
        pred = dest_dir / f"sample_{dest_idx}" / "pred.wav"
        assert pred.read_bytes() == f"pred-{src_idx}".encode()


def test_shuffle_pred_audio_target_symlink_points_to_same_sample(tmp_path: Path) -> None:
    """target.wav is never permuted — each dest target resolves to its own sample.

    :param tmp_path: Parents the source ``audio/`` tree and the dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"

    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    for i in range(_SAMPLE_COUNT):
        target = dest_dir / f"sample_{i}" / "target.wav"
        assert target.read_bytes() == f"target-{i}".encode()


def test_shuffle_pred_audio_dest_entries_are_symlinks(tmp_path: Path) -> None:
    """The dest tree holds symlinks, not copies, so it stays near-free on disk.

    :param tmp_path: Parents the source ``audio/`` tree and the dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"

    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    for i in range(_SAMPLE_COUNT):
        assert (dest_dir / f"sample_{i}" / "pred.wav").is_symlink()
        assert (dest_dir / f"sample_{i}" / "target.wav").is_symlink()


def test_shuffle_pred_audio_symlink_targets_are_absolute(tmp_path: Path) -> None:
    """Symlink targets are absolute, so the dest tree resolves from any cwd.

    :param tmp_path: Parents the source ``audio/`` tree and the dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"

    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    link_target = (dest_dir / "sample_0" / "pred.wav").readlink()
    assert link_target.is_absolute()


def test_shuffle_pred_audio_does_not_mutate_source(tmp_path: Path) -> None:
    """The source ``audio/`` tree is byte-for-byte identical after a shuffle.

    :param tmp_path: Parents the source ``audio/`` tree and the dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"
    before = {
        p.relative_to(audio_dir): p.read_bytes() for p in audio_dir.rglob("*") if p.is_file()
    }

    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    after = {p.relative_to(audio_dir): p.read_bytes() for p in audio_dir.rglob("*") if p.is_file()}
    assert after == before


def test_shuffle_pred_audio_preserves_pred_multiset(tmp_path: Path) -> None:
    """Every source pred.wav appears exactly once across the dest tree.

    :param tmp_path: Parents the source ``audio/`` tree and the dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"

    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    dest_preds = sorted(
        (dest_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)
    )
    source_preds = sorted(f"pred-{i}".encode() for i in range(_SAMPLE_COUNT))
    assert dest_preds == source_preds


def test_shuffle_pred_audio_changes_arrangement(tmp_path: Path) -> None:
    """A non-trivial seed actually reorders pred.wav relative to the source.

    :param tmp_path: Parents the source ``audio/`` tree and the dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"

    permutation = shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    dest = [(dest_dir / f"sample_{i}" / "pred.wav").read_bytes() for i in range(_SAMPLE_COUNT)]
    source = [f"pred-{i}".encode() for i in range(_SAMPLE_COUNT)]
    assert dest != source
    assert permutation != list(range(_SAMPLE_COUNT))


def test_shuffle_pred_audio_two_dirs_forces_swap(tmp_path: Path) -> None:
    """With two dirs the only non-identity permutation is forced, regardless of seed.

    :param tmp_path: Parents the two-dir source tree and the dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path, _PAIR)
    dest_dir = tmp_path / "shuffled"

    permutation = shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    assert permutation == [1, 0]


def test_shuffle_pred_audio_same_seed_yields_same_permutation(tmp_path: Path) -> None:
    """The permutation is deterministic for a fixed seed.

    :param tmp_path: Parents two independent source trees shuffled with one seed.
    """
    audio_dir_a = _build_audio_dir(tmp_path / "a")
    audio_dir_b = _build_audio_dir(tmp_path / "b")

    perm_a = shuffle_pred_audio(audio_dir_a, tmp_path / "a" / "shuffled", seed=_SEED)
    perm_b = shuffle_pred_audio(audio_dir_b, tmp_path / "b" / "shuffled", seed=_SEED)

    assert perm_a == perm_b


def test_shuffle_pred_audio_rerun_clears_stale_dest(tmp_path: Path) -> None:
    """A re-run replaces a stale dest tree rather than colliding with it.

    :param tmp_path: Parents the source tree and the reused dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"
    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)
    (dest_dir / "stale_marker").write_text("leftover")

    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    assert not (dest_dir / "stale_marker").exists()
    assert len(list(dest_dir.glob("sample_*"))) == _SAMPLE_COUNT


def test_shuffle_pred_audio_rerun_does_not_touch_source(tmp_path: Path) -> None:
    """Clearing a stale dest never deletes the real files it linked to.

    :param tmp_path: Parents the source tree and the reused dest dir.
    """
    audio_dir = _build_audio_dir(tmp_path)
    dest_dir = tmp_path / "shuffled"

    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)
    shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    for i in range(_SAMPLE_COUNT):
        assert (audio_dir / f"sample_{i}" / "pred.wav").read_bytes() == f"pred-{i}".encode()


def test_shuffle_pred_audio_raises_when_params_differ(tmp_path: Path) -> None:
    """The gate refuses to shuffle when a sample dir's params.csv differs.

    :param tmp_path: Holds the source tree with one mismatched params.csv.
    """
    audio_dir = _build_audio_dir(tmp_path)
    (audio_dir / "sample_2" / "params.csv").write_text(_DIFFERENT_PARAMS_CSV)
    dest_dir = tmp_path / "shuffled"

    with pytest.raises(ValueError, match="identical params"):
        shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)


def test_shuffle_pred_audio_does_not_create_dest_when_params_differ(tmp_path: Path) -> None:
    """A gate rejection leaves no dest dir behind (validate before writing).

    :param tmp_path: Holds the source tree with one mismatched params.csv.
    """
    audio_dir = _build_audio_dir(tmp_path)
    (audio_dir / "sample_2" / "params.csv").write_text(_DIFFERENT_PARAMS_CSV)
    dest_dir = tmp_path / "shuffled"

    with pytest.raises(ValueError):
        shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    assert not dest_dir.exists()


def test_shuffle_pred_audio_single_dir_returns_identity_no_dest(tmp_path: Path) -> None:
    """One sample dir cannot be shuffled — return identity, build no dest.

    :param tmp_path: Holds the single-dir source tree.
    """
    audio_dir = _build_audio_dir(tmp_path, 1)
    dest_dir = tmp_path / "shuffled"

    permutation = shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    assert permutation == [0]
    assert not dest_dir.exists()


def test_shuffle_pred_audio_no_samples_returns_empty_no_dest(tmp_path: Path) -> None:
    """An audio dir with no sample dirs yields an empty permutation and no dest.

    :param tmp_path: Parents an empty ``audio/`` directory.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    dest_dir = tmp_path / "shuffled"

    assert shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED) == []
    assert not dest_dir.exists()


def test_shuffle_pred_audio_ignores_non_sample_dirs(tmp_path: Path) -> None:
    """Only ``sample_*`` dirs count, even when a stray dir holds both files.

    :param tmp_path: Holds a source tree with the default samples plus a non-sample dir that has a
        pred.wav + params.csv (must still be excluded).
    """
    audio_dir = _build_audio_dir(tmp_path)
    stray = audio_dir / "metrics_scratch"
    stray.mkdir()
    (stray / "pred.wav").write_bytes(b"stray")
    (stray / "params.csv").write_text(_UNIFORM_PARAMS_CSV)
    dest_dir = tmp_path / "shuffled"

    permutation = shuffle_pred_audio(audio_dir, dest_dir, seed=_SEED)

    assert len(permutation) == _SAMPLE_COUNT
    assert not (dest_dir / "metrics_scratch").exists()
