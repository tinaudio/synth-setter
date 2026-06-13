"""Tests for `synth_setter.pipeline.data.stats` degenerate-bin handling (#998)."""

from __future__ import annotations

import io
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest

from synth_setter.pipeline.data import stats as _stats_module
from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard


@pytest.fixture(scope="module")
def stats_script() -> ModuleType:
    """Module-scoped handle to the stats module.

    :returns: The imported module shared across this file's tests.
    :rtype: ModuleType
    """
    return _stats_module


def _existing_from_samples(
    stats_script: ModuleType, samples: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """Run the script's Welford ``update`` over ``samples`` and return the state.

    :param stats_script: Imported get_dataset_stats module (the module-scoped
        fixture, passed in to avoid re-importing the script per call).
    :param samples: Array of shape ``(N, D)``. Each row is one observation.

    :returns: ``(count, mean, M2)`` tuple matching the layout the script's
        ``finalize()`` consumes.
    :rtype: tuple[int, np.ndarray, np.ndarray]
    """
    existing = (0, np.zeros(samples.shape[1]), np.zeros(samples.shape[1]))
    for row in samples:
        existing = stats_script.update(existing, row)
    return existing


def test_finalize_healthy_data_returns_positive_std(stats_script: ModuleType) -> None:
    """Welford output on random Gaussian data matches numpy and is positive everywhere.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(50, 4))
    existing = _existing_from_samples(stats_script, samples)

    mean, std = stats_script.finalize(existing)

    np.testing.assert_allclose(mean, samples.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(std, samples.std(axis=0, ddof=0), rtol=1e-6)
    assert (std > 0).all()


def test_finalize_constant_bin_raises_by_default(stats_script: ModuleType) -> None:
    """A single constant bin raises ``ValueError`` naming its index.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(50, 4))
    samples[:, 2] = 3.14  # bin 2 is constant across every sample

    existing = _existing_from_samples(stats_script, samples)

    with pytest.raises(ValueError, match=r"zero variance.*indices \[2\]"):
        stats_script.finalize(existing)


def test_finalize_constant_bin_masked_substitutes_unit_std_and_warns(
    stats_script: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    """With ``mask_degenerate=True``, the degenerate bin's ``std`` is replaced by 1.0 and logged.

    The substitution makes downstream ``(spec - mean) / std`` yield 0 for that
    bin on the training distribution (since ``spec == mean == constant``), so
    no datamodule changes are needed.

    :param stats_script: Imported get_dataset_stats module (fixture).
    :param caplog: pytest log-capture fixture.
    """
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(50, 4))
    samples[:, 1] = -7.0

    existing = _existing_from_samples(stats_script, samples)

    with caplog.at_level(logging.WARNING):
        mean, std = stats_script.finalize(existing, mask_degenerate=True)

    assert std[1] == 1.0
    assert (std[[0, 2, 3]] > 0).all() and (std[[0, 2, 3]] != 1.0).all()
    np.testing.assert_allclose(mean[1], -7.0)
    assert any("[1]" in record.message for record in caplog.records), caplog.text


def test_finalize_multiple_constant_bins_lists_all_indices(
    stats_script: ModuleType,
) -> None:
    """The raise message enumerates every degenerate bin, not just the first one.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(50, 5))
    samples[:, 0] = 1.0
    samples[:, 3] = -2.0

    existing = _existing_from_samples(stats_script, samples)

    with pytest.raises(ValueError, match=r"indices \[0, 3\]"):
        stats_script.finalize(existing)


def test_check_degenerate_bins_no_zeros_returns_silently(
    stats_script: ModuleType,
) -> None:
    """All-positive ``std`` makes the pure check a no-op (no raise).

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.array([0.1, 0.5, 2.0])

    stats_script._check_degenerate_bins(std)


def test_check_degenerate_bins_zero_entry_raises(stats_script: ModuleType) -> None:
    """A single ``std==0`` entry raises ``ValueError`` naming the index.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.array([0.1, 0.0, 0.5])

    with pytest.raises(ValueError, match=r"zero variance.*indices \[1\]"):
        stats_script._check_degenerate_bins(std)


def test_check_degenerate_bins_multi_d_std_reports_coordinate_tuples(
    stats_script: ModuleType,
) -> None:
    """For multi-D std (real mel layouts), degenerate positions are reported as coordinate tuples.

    First-axis-only indexing would collapse all degenerate element coordinates to channel/row
    indices and lose the bin location.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.array([[0.5, 0.0], [0.3, 0.7]])  # one degenerate element at [0, 1]

    with pytest.raises(ValueError, match=r"shape \(2, 2\).*indices \[\[0, 1\]\]"):
        stats_script._check_degenerate_bins(std)


def test_check_degenerate_bins_caps_index_preview_with_overflow_summary(
    stats_script: ModuleType,
) -> None:
    """Large degenerate counts truncate the listing with a ``+N more`` suffix.

    A fully-degenerate 3-D mel spec can contain ~100k zero-variance elements; listing them all
    would produce a megabyte-scale message.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.zeros(100)  # 100 degenerate bins; preview cap is 20 → expect "+80 more"

    with pytest.raises(ValueError, match=r"\+80 more"):
        stats_script._check_degenerate_bins(std)


def test_fix_degenerate_bins_no_zeros_returns_input_unchanged(
    stats_script: ModuleType,
) -> None:
    """All-positive ``std`` is returned unchanged (no substitution, no warning).

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.array([0.1, 0.5, 2.0])

    returned = stats_script._fix_degenerate_bins(std)

    np.testing.assert_array_equal(returned, std)


def test_fix_degenerate_bins_zero_entry_substitutes_unit_std_and_warns(
    stats_script: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    """A single ``std==0`` entry is substituted with ``1.0`` and the index is logged.

    :param stats_script: Imported get_dataset_stats module (fixture).
    :param caplog: pytest log-capture fixture.
    """
    std = np.array([0.1, 0.0, 0.5])

    with caplog.at_level(logging.WARNING):
        returned = stats_script._fix_degenerate_bins(std)

    np.testing.assert_array_equal(returned, np.array([0.1, 1.0, 0.5]))
    assert any("[1]" in record.message for record in caplog.records), caplog.text


def test_fix_degenerate_bins_preserves_input_dtype(
    stats_script: ModuleType,
) -> None:
    """Float32 ``std`` stays float32 — substituted entries do not promote to float64.

    Real ``stats.npz`` files written from torch tensors / HDF5 are float32; a
    ``np.where(std == 0, 1.0, std)`` substitution would promote the literal
    1.0 and silently double the on-disk size of the artifact.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.array([0.1, 0.0, 0.5], dtype=np.float32)

    returned = stats_script._fix_degenerate_bins(std)

    assert returned.dtype == np.float32
    np.testing.assert_array_equal(returned, np.array([0.1, 1.0, 0.5], dtype=np.float32))


def test_finalize_with_at_most_one_sample_raises_distinct_error(stats_script: ModuleType) -> None:
    """Count<=1 (empty state or single sample) raises a distinct ``<=1 samples`` error.

    The pre-existing ``M2 / count if count > 1 else 0`` line in ``finalize``
    returns a scalar variance for these cases, which the degeneracy helpers
    surface as a "scalar / <=1 samples" failure rather than the per-bin
    "zero variance" path that requires a populated std array. The error must
    fire regardless of ``mask_degenerate`` — substituting a scalar makes no
    sense either.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    empty_existing = (0, 0, 0)
    sample = np.array([0.1, 0.2, 0.3])
    single_sample_existing = stats_script.update((0, np.zeros(3), np.zeros(3)), sample)

    for existing in (empty_existing, single_sample_existing):
        with pytest.raises(ValueError, match=r"<=1 samples"):
            stats_script.finalize(existing)
        with pytest.raises(ValueError, match=r"<=1 samples"):
            stats_script.finalize(existing, mask_degenerate=True)


def test_stream_stats_lance_matches_numpy(stats_script: ModuleType, tmp_path: Path) -> None:
    """Lance stats fold projects ``mel_spec`` and matches numpy mean/std.

    :param stats_script: Imported get_dataset_stats module (fixture).
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = build_lance_smoke_spec()
    shard = tmp_path / spec.shards[0].filename
    write_minimal_lance_shard(shard, spec)

    mean, std = stats_script.stream_stats_lance([shard])

    from synth_setter.pipeline.data.lance_shard import iter_lance_column_rows

    rows = list(iter_lance_column_rows(shard, "mel_spec"))
    expected = np.stack(rows, axis=0)
    np.testing.assert_allclose(mean, expected.mean(axis=0))
    np.testing.assert_allclose(std, expected.std(axis=0))


def test_stream_stats_lance_accumulates_across_shards_matches_numpy(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """Welford folds two distinct lance shards to the same mean/std as numpy over their union.

    Single-shard coverage cannot catch a cross-shard accumulator bug (wrong
    count or M2 merge); distinct per-shard mel data pins the multi-shard fold.

    :param stats_script: Imported get_dataset_stats module (fixture).
    :param tmp_path: Hosts the two written ``.lance`` shards.
    """
    from tests.helpers.lance_fixtures import write_lance_shard

    rng = np.random.default_rng(0)
    mel_a = rng.standard_normal((4, 2, 4, 5)).astype(np.float32)
    mel_b = rng.standard_normal((3, 2, 4, 5)).astype(np.float32)
    shard_a = tmp_path / "shard-000000.lance"
    shard_b = tmp_path / "shard-000001.lance"
    write_lance_shard(shard_a, {"mel_spec": mel_a})
    write_lance_shard(shard_b, {"mel_spec": mel_b})

    mean, std = stats_script.stream_stats_lance([shard_a, shard_b])

    stacked = np.concatenate([mel_a, mel_b], axis=0)
    np.testing.assert_allclose(mean, stacked.mean(axis=0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(std, stacked.std(axis=0), rtol=1e-5, atol=1e-5)


def _write_mel_shard(path: Path, mel_batches: list[np.ndarray]) -> None:
    """Write a tar shard at ``path`` containing one ``mel_spec.npy`` per batch.

    Mirrors the writer's per-batch naming convention exactly — the tar key
    for each batch is the *cumulative row offset* where the batch starts
    (``writers.py``: ``"__key__": f"{start_idx:08d}"``), not the batch
    ordinal. So a shard with batches of size 4, 7, 3 emits members
    ``00000000.mel_spec.npy``, ``00000004.mel_spec.npy``,
    ``00000011.mel_spec.npy``. Using ordinals instead would silently
    hide bugs in code that depends on batch keys. The ``metadata.json``
    sentinel mirrors the writer; other per-row fields (``audio``,
    ``param_array``) are intentionally omitted — the stats path only
    consumes ``mel_spec``.

    :param path: Filesystem path where the tar archive will be written.
    :param mel_batches: One mel array per batch, each shaped
        ``(rows, ...inner)``. Inner shape must agree across batches so the
        Welford reduction over rows is well-defined.
    :returns: ``None``.
    :rtype: None
    """
    with tarfile.open(path, mode="w:") as tar:
        start_idx = 0
        for mel in mel_batches:
            buf = io.BytesIO()
            np.save(buf, mel)
            payload = buf.getvalue()
            info = tarfile.TarInfo(name=f"{start_idx:08d}.mel_spec.npy")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            start_idx += mel.shape[0]
        metadata = b"{}"
        info = tarfile.TarInfo(name="metadata.json")
        info.size = len(metadata)
        tar.addfile(info, io.BytesIO(metadata))


def test_get_stats_wds_writes_sibling_stats_npz_with_mel_inner_shape(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """One shard with a known mel batch yields ``stats.npz`` sibling whose arrays match
    ``mel.shape[1:]``.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    rng = np.random.default_rng(0)
    mel = rng.normal(size=(8, 2, 4, 5)).astype(np.float32)
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    stats_script.get_stats_wds(str(tmp_path))

    out = np.load(tmp_path / "stats.npz")
    assert out["mean"].shape == (2, 4, 5)
    assert out["std"].shape == (2, 4, 5)


def test_get_stats_wds_matches_numpy_baseline_across_shards(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """Mean and std across two shards match ``np.mean``/``np.std`` over the stacked rows.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    rng = np.random.default_rng(1)
    mel_a = rng.normal(size=(5, 3, 4)).astype(np.float32)
    mel_b = rng.normal(size=(7, 3, 4)).astype(np.float32)
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel_a])
    _write_mel_shard(tmp_path / "shard-000001.tar", [mel_b])

    stats_script.get_stats_wds(str(tmp_path))

    stacked = np.concatenate([mel_a, mel_b], axis=0)
    out = np.load(tmp_path / "stats.npz")
    np.testing.assert_allclose(out["mean"], stacked.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out["std"], stacked.std(axis=0, ddof=0), rtol=1e-5, atol=1e-6)


def test_get_stats_wds_two_value_textbook_welford_state(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """The textbook ``[1.0, 3.0]`` two-value example yields the documented mean and population std.

    Pins the canonical Welford worked-example: after two updates the state
    is ``count=2, mean=2.0, M2=2.0``. ``finalize()`` returns population
    std (``variance = M2 / count`` — matches every other test in this
    file), so std is ``sqrt(2.0 / 2) = 1.0``. Sample std (``ddof=1``,
    which the spec sketches as ``M2 / (n-1) = 2.0``) is intentionally
    *not* what this stats path computes.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    mel = np.array([[1.0], [3.0]], dtype=np.float64)
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    stats_script.get_stats_wds(str(tmp_path))

    out = np.load(tmp_path / "stats.npz")
    np.testing.assert_allclose(out["mean"], np.array([2.0]))
    np.testing.assert_allclose(out["std"], np.array([1.0]))


def test_get_stats_wds_large_near_equal_values_avoid_catastrophic_cancellation(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """Welford preserves precision on ``[1e9+1..1e9+4]`` where the naive two-pass formula loses
    significance.

    The naive ``var = (sum(x**2) - sum(x)**2 / n) / n`` subtracts two
    near-equal ~4e18 floats and loses most of the answer to rounding.
    Welford's running correction keeps full precision because it
    operates on deviations from the running mean rather than the raw
    sums. Mean is asserted with an absolute tolerance (not relative) so
    a regression that drops the ``.5`` fractional part — the exact
    failure mode of catastrophic cancellation here — would fail loudly.

    Inputs are float64; float32 cannot represent ``1e9 + k`` for
    ``k in {1..4}`` distinctly (its integer-precision ceiling is
    ``2**24 ≈ 1.68e7``), so casting to float32 would collapse the four
    values before they ever reached Welford.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    values = np.array([1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4], dtype=np.float64)
    mel = values.reshape(4, 1)
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    stats_script.get_stats_wds(str(tmp_path))

    out = np.load(tmp_path / "stats.npz")
    np.testing.assert_allclose(out["mean"], np.array([1_000_000_002.5]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(out["std"], np.array([np.sqrt(1.25)]), rtol=1e-9, atol=0)


def test_get_stats_wds_handles_multiple_batches_per_shard(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """Multiple per-batch ``mel_spec.npy`` members in one shard are summed via Welford.

    The writer can split a shard into several ``<batch_key>.mel_spec.npy`` members. The stats path
    must iterate all of them, not just the first.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    rng = np.random.default_rng(2)
    batch_a = rng.normal(size=(4, 3, 4)).astype(np.float32)
    batch_b = rng.normal(size=(6, 3, 4)).astype(np.float32)
    _write_mel_shard(tmp_path / "shard-000000.tar", [batch_a, batch_b])

    stats_script.get_stats_wds(str(tmp_path))

    stacked = np.concatenate([batch_a, batch_b], axis=0)
    out = np.load(tmp_path / "stats.npz")
    np.testing.assert_allclose(out["mean"], stacked.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out["std"], stacked.std(axis=0, ddof=0), rtol=1e-5, atol=1e-6)


def test_get_stats_wds_raises_on_constant_bin_by_default(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """A bin that is constant across all rows raises ``ValueError`` from the existing degeneracy
    check.

    Reuses ``_check_degenerate_bins`` via ``finalize``; this test pins
    that get_stats_wds wires into the same code path as the other two
    entrypoints.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    rng = np.random.default_rng(3)
    mel = rng.normal(size=(8, 2, 3)).astype(np.float32)
    mel[:, 1, 2] = 0.5
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    with pytest.raises(ValueError, match=r"zero variance"):
        stats_script.get_stats_wds(str(tmp_path))


def test_get_stats_wds_masks_constant_bin_when_flag_set(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """``mask_degenerate=True`` substitutes ``std=1.0`` at the constant position and writes the
    file.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    rng = np.random.default_rng(4)
    mel = rng.normal(size=(8, 2, 3)).astype(np.float32)
    mel[:, 1, 2] = 0.5
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    stats_script.get_stats_wds(str(tmp_path), mask_degenerate=True)

    out = np.load(tmp_path / "stats.npz")
    assert out["std"][1, 2] == 1.0
    np.testing.assert_allclose(out["mean"][1, 2], 0.5)


def test_get_stats_wds_no_shards_in_directory_raises(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """A directory with no ``shard-*.tar`` raises ``FileNotFoundError`` naming the directory.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the (empty) shard directory.
    """
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        stats_script.get_stats_wds(str(tmp_path))


def test_get_stats_wds_iterates_sorted_shard_order(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shard filenames are processed in lexicographic order so logs and any future order-sensitive
    consumers stay reproducible.

    Welford mean/std are order-invariant, so a result-only assertion would
    pass regardless of iteration order. Spying on ``_iter_mel_batches``
    captures the actual processing sequence so the determinism
    requirement is observable.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    :param monkeypatch: pytest fixture used to install the order-recording spy.
    """
    rng = np.random.default_rng(5)
    mel = rng.normal(size=(3, 2, 2)).astype(np.float32)
    for shard_name in ("shard-000002.tar", "shard-000000.tar", "shard-000001.tar"):
        _write_mel_shard(tmp_path / shard_name, [mel])

    visited: list[str] = []
    real_iter = stats_script._iter_mel_batches

    def recording_iter(shard_path: Path) -> Iterator[np.ndarray]:
        visited.append(Path(shard_path).name)
        yield from real_iter(shard_path)

    monkeypatch.setattr(stats_script, "_iter_mel_batches", recording_iter)

    stats_script.get_stats_wds(str(tmp_path))

    assert visited == ["shard-000000.tar", "shard-000001.tar", "shard-000002.tar"]


def test_get_stats_wds_shard_without_mel_members_raises(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """A shard tar that exists but has zero readable ``mel_spec.npy`` members raises
    ``ValueError``.

    A silent skip would let a truncated or wrong-schema shard contribute
    nothing to the Welford accumulator while the function still wrote
    ``stats.npz`` from the remaining shards — partial normalization stats
    are a footgun.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    rng = np.random.default_rng(7)
    mel = rng.normal(size=(2, 2, 2)).astype(np.float32)
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    empty_tar = tmp_path / "shard-000001.tar"
    with tarfile.open(empty_tar, mode="w:") as tar:
        info = tarfile.TarInfo(name="metadata.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))

    with pytest.raises(ValueError, match=r"shard-000001\.tar contained no readable"):
        stats_script.get_stats_wds(str(tmp_path))


def test_get_stats_wds_shard_with_unextractable_mel_member_raises(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """A matched ``*.mel_spec.npy`` member that is not a regular file raises ``ValueError``.

    A non-regular matched member (directory, symlink, hardlink, device)
    must be rejected eagerly. Silently skipping such a member would let
    a shard with *some* readable mel members defeat the per-shard
    ``shard_rows == 0`` guard and still write partial stats.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    rng = np.random.default_rng(8)
    mel = rng.normal(size=(2, 2, 2)).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, mel)
    payload = buf.getvalue()

    shard_path = tmp_path / "shard-000000.tar"
    with tarfile.open(shard_path, mode="w:") as tar:
        info = tarfile.TarInfo(name="00000000.mel_spec.npy")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        # Second matched member is a *directory* entry; ``TarInfo.isfile()``
        # is False for it. A silent ``continue`` would let the readable
        # member above contribute rows and write partial stats.
        dir_info = tarfile.TarInfo(name="00000002.mel_spec.npy")
        dir_info.type = tarfile.DIRTYPE
        tar.addfile(dir_info)

    with pytest.raises(ValueError, match=r"00000002\.mel_spec\.npy.*not a regular file"):
        stats_script.get_stats_wds(str(tmp_path))


def test_get_stats_wds_shard_with_symlink_mel_member_raises(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """A matched ``*.mel_spec.npy`` symlink to another archive member raises ``ValueError``.

    ``tarfile.extractfile`` happily returns a file object for a symlink
    or hardlink that resolves to another archive member, so a bare
    ``extractfile is None`` check is not enough. Using
    ``TarInfo.isfile()`` (matches only ``REGTYPE``/``AREGTYPE``) rejects
    a link named like a mel batch — otherwise the same payload could be
    counted multiple times.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    rng = np.random.default_rng(9)
    mel = rng.normal(size=(2, 2, 2)).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, mel)
    payload = buf.getvalue()

    shard_path = tmp_path / "shard-000000.tar"
    with tarfile.open(shard_path, mode="w:") as tar:
        info = tarfile.TarInfo(name="00000000.mel_spec.npy")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        sym_info = tarfile.TarInfo(name="00000002.mel_spec.npy")
        sym_info.type = tarfile.SYMTYPE
        sym_info.linkname = "00000000.mel_spec.npy"
        tar.addfile(sym_info)

    with pytest.raises(ValueError, match=r"00000002\.mel_spec\.npy.*not a regular file"):
        stats_script.get_stats_wds(str(tmp_path))


def test_get_stats_wds_shard_with_duplicate_named_mel_members_yields_each_payload(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """Two matched members with the same name yield two distinct payloads, not one twice.

    ``tarfile.extractfile(name)`` resolves by name lookup and returns
    only the *last* matching entry, so iterating ``TarInfo`` objects and
    extracting each by the ``TarInfo`` itself is required to avoid
    counting the same payload twice on a malformed (duplicate-name)
    archive.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    payload_a = io.BytesIO()
    np.save(payload_a, np.zeros((1, 2, 2), dtype=np.float32))
    bytes_a = payload_a.getvalue()
    payload_b = io.BytesIO()
    np.save(payload_b, np.ones((1, 2, 2), dtype=np.float32))
    bytes_b = payload_b.getvalue()

    shard_path = tmp_path / "shard-000000.tar"
    with tarfile.open(shard_path, mode="w:") as tar:
        for raw in (bytes_a, bytes_b):
            info = tarfile.TarInfo(name="00000000.mel_spec.npy")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))

    arrays = list(stats_script._iter_mel_batches(shard_path))
    assert len(arrays) == 2
    # If extractfile resolved by name, both entries would resolve to the
    # last addfile (all-ones); pin that we see both payloads.
    sums = sorted(float(a.sum()) for a in arrays)
    assert sums == [0.0, 4.0]


def test_cli_dispatches_directory_of_tar_shards_to_wds_path_with_default_flag(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` routes a directory containing ``shard-*.tar`` to ``get_stats_wds`` with
    ``mask_degenerate=False`` by default.

    Spies record ``(branch, mask_degenerate)`` so this test pins both the
    chosen branch *and* the default-False forwarding. A regression that
    accidentally hardcoded ``True`` (or dropped the kwarg entirely) would
    fail here, not silently change normalization behavior in production.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    rng = np.random.default_rng(6)
    mel = rng.normal(size=(2, 2, 2)).astype(np.float32)
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        stats_script,
        "get_stats_wds",
        lambda *a, **kw: calls.append(("wds", kw.get("mask_degenerate", False))),
    )
    monkeypatch.setattr(
        stats_script,
        "get_stats_hdf5",
        lambda *a, **kw: calls.append(("hdf5", kw.get("mask_degenerate", False))),
    )
    monkeypatch.setattr(
        stats_script,
        "get_stats_directory",
        lambda *a, **kw: calls.append(("dir", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(tmp_path)])

    assert calls == [("wds", False)]


def test_cli_dispatches_h5_input_to_hdf5_path_with_default_flag(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` routes a ``.h5`` input to ``get_stats_hdf5`` with ``mask_degenerate=False`` by
    default.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used to place a sentinel ``.h5`` path.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    h5_path = tmp_path / "train.h5"
    h5_path.write_bytes(b"")

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        stats_script,
        "get_stats_wds",
        lambda *a, **kw: calls.append(("wds", kw.get("mask_degenerate", False))),
    )
    monkeypatch.setattr(
        stats_script,
        "get_stats_hdf5",
        lambda *a, **kw: calls.append(("hdf5", kw.get("mask_degenerate", False))),
    )
    monkeypatch.setattr(
        stats_script,
        "get_stats_directory",
        lambda *a, **kw: calls.append(("dir", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(h5_path)])

    assert calls == [("hdf5", False)]


def test_cli_dispatches_directory_without_tars_to_audio_directory_path_with_default_flag(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` falls back to ``get_stats_directory`` with ``mask_degenerate=False`` by default.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the (no-shard) input directory.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        stats_script,
        "get_stats_wds",
        lambda *a, **kw: calls.append(("wds", kw.get("mask_degenerate", False))),
    )
    monkeypatch.setattr(
        stats_script,
        "get_stats_hdf5",
        lambda *a, **kw: calls.append(("hdf5", kw.get("mask_degenerate", False))),
    )
    monkeypatch.setattr(
        stats_script,
        "get_stats_directory",
        lambda *a, **kw: calls.append(("dir", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(tmp_path)])

    assert calls == [("dir", False)]


def test_cli_forwards_mask_degenerate_bins_flag_to_wds_path(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` with ``--mask-degenerate-bins`` forwards ``mask_degenerate=True`` to
    ``get_stats_wds``.

    The CLI dispatch hands the parsed argparse flag through as a kwarg — a regression that
    hardcoded the value or dropped the kwarg would silently change normalization behavior on
    degenerate-bin datasets without changing which entrypoint runs.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    rng = np.random.default_rng(8)
    mel = rng.normal(size=(2, 2, 2)).astype(np.float32)
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        stats_script,
        "get_stats_wds",
        lambda *a, **kw: calls.append(("wds", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(tmp_path), "--mask-degenerate-bins"])

    assert calls == [("wds", True)]


def test_cli_forwards_mask_degenerate_bins_flag_to_hdf5_path(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` with ``--mask-degenerate-bins`` forwards ``mask_degenerate=True`` to
    ``get_stats_hdf5``.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used to place a sentinel ``.h5`` path.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    h5_path = tmp_path / "train.h5"
    h5_path.write_bytes(b"")

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        stats_script,
        "get_stats_hdf5",
        lambda *a, **kw: calls.append(("hdf5", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(h5_path), "--mask-degenerate-bins"])

    assert calls == [("hdf5", True)]


def test_cli_forwards_mask_degenerate_bins_flag_to_audio_directory_path(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` with ``--mask-degenerate-bins`` forwards ``mask_degenerate=True`` to
    ``get_stats_directory``.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the (no-shard) input directory.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        stats_script,
        "get_stats_directory",
        lambda *a, **kw: calls.append(("dir", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(tmp_path), "--mask-degenerate-bins"])

    assert calls == [("dir", True)]


def test_cli_help_advertises_mask_degenerate_bins_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI ``--help`` text documents the new flag so operators can discover it.

    Invokes ``_parse_args`` in-process rather than shelling out to
    ``python -m synth_setter.pipeline.data.stats``: under ``mutmut run``'s
    stats phase, a subprocess inherits ``MUTANT_UNDER_TEST=stats`` and the
    mutated module's trampoline calls into ``mutmut.config`` which is
    ``None`` in any fresh interpreter — see CLAUDE.md "Mutation Testing".
    In-process also lets mutations of ``_parse_args`` actually reach this
    test (the subprocess form would always run the un-mutated function).

    :param capsys: pytest fixture that captures ``sys.stdout``/``sys.stderr``
        emitted by argparse's ``--help`` handler.
    """
    with pytest.raises(SystemExit) as exc_info:
        _stats_module._parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--mask-degenerate-bins" in captured.out


def _write_tiny_mel_h5(path: Path, rows: int = 4, seed: int = 0) -> None:
    """Write a small HDF5 file with a ``mel_spec`` dataset.

    :param path: Filesystem path where the ``.h5`` file is written.
    :param rows: Leading-axis length; must be >=2 so Welford variance is
        non-degenerate.
    :param seed: PRNG seed so the same call writes the same payload across
        repeated invocations in one process.
    """
    rng = np.random.default_rng(seed)
    payload = rng.normal(size=(rows, 2, 4, 5)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("mel_spec", data=payload)


def test_get_stats_hdf5_closes_dask_client_and_h5_file(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_stats_hdf5`` releases its Dask client and h5py file handle on return.

    The PR's stated motivation is leak prevention in the in-process CLI:
    the new ``with`` blocks around ``Client(...)`` and ``h5py.File(...)``
    must teardown both resources before control returns. A regression
    that re-introduces a bare ``Client(...)`` (no ``with``) would leave
    ``status == "running"`` and a worker scheduler bound to a TCP port,
    eventually preventing a second call.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: Pytest tmp dir; hosts the seeded ``train.h5``.
    :param monkeypatch: Pytest fixture used to swap the module-level
        ``Client`` symbol for a spy subclass that records the instance.
    """
    from typing import Any

    from dask.distributed import Client as _Client

    instances: list[_Client] = []

    class _SpyClient(_Client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(stats_script, "Client", _SpyClient)
    train_h5 = tmp_path / "train.h5"
    _write_tiny_mel_h5(train_h5)

    stats_script.get_stats_hdf5(str(train_h5))

    assert len(instances) == 1
    assert instances[0].status == "closed"
    # Reopening the h5 must succeed — proves the file handle owned by
    # ``get_stats_hdf5`` released its lock before returning.
    with h5py.File(train_h5, "r") as reopened:
        assert reopened.id.valid


def test_get_stats_hdf5_writes_sibling_stats_npz_with_mean_std_keys(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """``get_stats_hdf5(path)`` writes ``stats.npz`` sibling to ``path``.

    ``finalize_hdf5`` asserts ``stats_npz.is_file()`` after the call, but
    every test of finalize stubs ``get_stats_hdf5``. Pin the real
    output-path derivation (``VSTDataset.get_stats_file_path``) and
    the on-disk schema (``mean`` + ``std`` keyed arrays in the input
    dtype) directly against the real function.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: Pytest tmp dir; hosts the seeded ``train.h5``.
    """
    train_h5 = tmp_path / "train.h5"
    _write_tiny_mel_h5(train_h5)

    stats_script.get_stats_hdf5(str(train_h5))

    stats_npz = tmp_path / "stats.npz"
    assert stats_npz.is_file()
    with np.load(stats_npz) as loaded:
        assert set(loaded.files) == {"mean", "std"}
        # Input dtype is float32; stats must preserve it (a float64 std
        # would silently double on-disk size and downstream cast cost).
        assert loaded["mean"].dtype == np.float32
        assert loaded["std"].dtype == np.float32
        # Trailing shape matches the per-sample inner of the mel dataset.
        assert loaded["mean"].shape == (2, 4, 5)
        assert loaded["std"].shape == (2, 4, 5)


def test_get_stats_hdf5_callable_twice_in_same_process(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """Two back-to-back calls succeed; no leaked Client port, no stale h5py handle.

    ``finalize_dataset.finalize_hdf5`` invokes ``get_stats_hdf5`` once per run,
    but multiple runs in the same process (CI matrix shards, an operator
    REPL) must not regress. Without the ``with`` blocks the second call
    would hit a port-in-use ``OSError`` from the scheduler or open the
    h5 file on top of an unreleased handle.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: Pytest tmp dir; hosts two seeded train shards.
    """
    # Same path for both calls: a stale h5py handle would prevent the
    # second open; a leaked dask client would collide on its scheduler
    # port. Either failure mode surfaces as an exception below.
    train_h5 = tmp_path / "train.h5"
    _write_tiny_mel_h5(train_h5, seed=1)

    stats_script.get_stats_hdf5(str(train_h5))
    stats_script.get_stats_hdf5(str(train_h5))

    assert (tmp_path / "stats.npz").is_file()
