"""Tests for `synth_setter.pipeline.data.stats` degenerate-bin handling (#998)."""

from __future__ import annotations

import io
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from synth_setter.pipeline.data import stats as _stats_module


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


def test_cli_dispatches_directory_of_tar_shards_to_wds_path(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` routes a directory containing ``shard-*.tar`` to ``get_stats_wds``.

    Spies on each dispatch target so the assertion verifies the chosen branch, not the resulting
    stats file (those are covered by the behavior tests above).

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    rng = np.random.default_rng(6)
    mel = rng.normal(size=(2, 2, 2)).astype(np.float32)
    _write_mel_shard(tmp_path / "shard-000000.tar", [mel])

    calls: list[str] = []
    monkeypatch.setattr(stats_script, "get_stats_wds", lambda *a, **kw: calls.append("wds"))
    monkeypatch.setattr(stats_script, "get_stats_hdf5", lambda *a, **kw: calls.append("hdf5"))
    monkeypatch.setattr(stats_script, "get_stats_directory", lambda *a, **kw: calls.append("dir"))

    stats_script.main([str(tmp_path)])

    assert calls == ["wds"]


def test_cli_dispatches_h5_input_to_hdf5_path(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` routes a ``.h5`` input to ``get_stats_hdf5`` (regression guard).

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used to place a sentinel ``.h5`` path.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    h5_path = tmp_path / "train.h5"
    h5_path.write_bytes(b"")

    calls: list[str] = []
    monkeypatch.setattr(stats_script, "get_stats_wds", lambda *a, **kw: calls.append("wds"))
    monkeypatch.setattr(stats_script, "get_stats_hdf5", lambda *a, **kw: calls.append("hdf5"))
    monkeypatch.setattr(stats_script, "get_stats_directory", lambda *a, **kw: calls.append("dir"))

    stats_script.main([str(h5_path)])

    assert calls == ["hdf5"]


def test_cli_dispatches_directory_without_tars_to_audio_directory_path(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` falls back to ``get_stats_directory`` when the directory has no shards.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the (no-shard) input directory.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    calls: list[str] = []
    monkeypatch.setattr(stats_script, "get_stats_wds", lambda *a, **kw: calls.append("wds"))
    monkeypatch.setattr(stats_script, "get_stats_hdf5", lambda *a, **kw: calls.append("hdf5"))
    monkeypatch.setattr(stats_script, "get_stats_directory", lambda *a, **kw: calls.append("dir"))

    stats_script.main([str(tmp_path)])

    assert calls == ["dir"]


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
