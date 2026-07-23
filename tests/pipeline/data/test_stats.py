"""Tests for `synth_setter.pipeline.data.stats` degenerate-bin handling (#998)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from synth_setter.data.vst.shapes import MEL_SPEC_FIELD, dataset_field_shapes
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


def test_merge_welford_two_states_matches_single_pass_over_all_rows(
    stats_script: ModuleType,
) -> None:
    """Chan-et-al. merge of two per-shard states equals one pass over the union.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    rng = np.random.default_rng(1)
    shard_a = rng.normal(size=(7, 3))
    shard_b = rng.normal(loc=5.0, size=(13, 3))

    merged = stats_script.merge_welford(
        _existing_from_samples(stats_script, shard_a),
        _existing_from_samples(stats_script, shard_b),
    )
    single_pass = _existing_from_samples(stats_script, np.concatenate([shard_a, shard_b]))

    assert merged[0] == 20
    np.testing.assert_allclose(merged[1], single_pass[1], rtol=1e-9)
    np.testing.assert_allclose(merged[2], single_pass[2], rtol=1e-9)


def test_merge_welford_chained_three_way_matches_single_pass(
    stats_script: ModuleType,
) -> None:
    """Chained folds over three shards equal one pass over the union (associativity).

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    rng = np.random.default_rng(2)
    shards = [rng.normal(loc=i * 3.0, size=(5 + i, 2)) for i in range(3)]

    chained = (0, 0, 0)
    for shard in shards:
        chained = stats_script.merge_welford(chained, _existing_from_samples(stats_script, shard))
    single_pass = _existing_from_samples(stats_script, np.concatenate(shards))

    assert chained[0] == single_pass[0]
    np.testing.assert_allclose(chained[1], single_pass[1], rtol=1e-9)
    np.testing.assert_allclose(chained[2], single_pass[2], rtol=1e-9)


def test_merge_welford_zero_state_is_identity_seed(stats_script: ModuleType) -> None:
    """``(0, 0, 0)`` folds as identity on both sides and merges with itself.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    state = (2, np.array([1.0, 3.0]), np.array([0.5, 2.0]))

    left = stats_script.merge_welford((0, 0, 0), state)
    right = stats_script.merge_welford(state, (0, 0, 0))

    assert left[0] == right[0] == 2
    np.testing.assert_allclose(left[1], state[1])
    np.testing.assert_allclose(right[2], state[2])
    assert stats_script.merge_welford((0, 0, 0), (0, 0, 0)) == (0, 0, 0)


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


def test_finalize_textbook_two_samples_returns_exact_mean_and_unit_std(
    stats_script: ModuleType,
) -> None:
    """Welford ``update``/``finalize`` on ``[1.0, 3.0]`` yields mean 2.0 and std 1.0 exactly.

    Pins the update/finalize core against the closed-form answer rather than random data, so a
    flipped increment order or a wrong variance formula is caught directly instead of hiding behind
    sampling noise.

    :param stats_script: Imported stats module (fixture).
    """
    samples = np.array([[1.0], [3.0]])
    existing = _existing_from_samples(stats_script, samples)

    mean, std = stats_script.finalize(existing)

    np.testing.assert_allclose(mean, [2.0], rtol=0, atol=0)
    np.testing.assert_allclose(std, [1.0], rtol=0, atol=0)


def test_finalize_large_offset_values_matches_numpy_std_without_cancellation(
    stats_script: ModuleType,
) -> None:
    """Welford over values near 1e9 recovers mean 1e9+2.5 and std matching ``np.std`` tightly.

    A naive ``E[x^2] - E[x]^2`` variance catastrophically cancels at this offset
    (the squared magnitudes swamp the tiny spread); Welford's incremental ``M2``
    does not. Pins that the finalize core keeps full precision on large-mean data.

    :param stats_script: Imported stats module (fixture).
    """
    values = np.array([1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4])
    existing = _existing_from_samples(stats_script, values.reshape(-1, 1))

    mean, std = stats_script.finalize(existing)

    np.testing.assert_allclose(mean, [1e9 + 2.5], rtol=0, atol=1e-6)
    np.testing.assert_allclose(std, [values.std(ddof=0)], rtol=1e-12, atol=0)


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

    Real ``stats.npz`` files written from the Lance mel rows are float32; a
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


def test_fold_lance_float16_mel_accumulates_in_float32(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """Float16 storage does not reduce the precision of Welford state.

    :param stats_script: Imported stats module fixture.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    base = build_lance_smoke_spec()
    render = base.render.model_copy(update={"mel_spec_dtype": "float16"})
    spec = build_lance_smoke_spec(render=render)
    shard = tmp_path / spec.shards[0].filename
    write_minimal_lance_shard(shard, spec)

    count, mean, m2 = stats_script.fold_lance_shard_into_welford((0, 0, 0), shard)

    assert count == spec.render.samples_per_shard
    assert isinstance(mean, np.ndarray) and mean.dtype == np.float32
    assert isinstance(m2, np.ndarray) and m2.dtype == np.float32
    assert np.isfinite(m2).all()


def test_stream_stats_lance_rejects_empty_shard_sequence(stats_script: ModuleType) -> None:
    """An empty input is rejected instead of returning meaningless zero statistics.

    :param stats_script: Imported stats module fixture.
    """
    with pytest.raises(FileNotFoundError, match="no shard URIs"):
        stats_script.stream_stats_lance([])


def test_stream_stats_lance_shard_with_zero_readable_rows_raises(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """A shard carrying zero readable ``mel_spec`` rows aborts instead of folding silently.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir holding the zero-row shard.
    """
    spec = build_lance_smoke_spec()
    shard = tmp_path / spec.shards[0].filename
    write_minimal_lance_shard(shard, spec, num_rows=0)

    with pytest.raises(ValueError, match="no readable"):
        stats_script.stream_stats_lance([shard])


def _seed_lance_shard_dir(tmp_path: Path) -> Path:
    """Write one ``shard-000000.lance`` dataset under ``tmp_path`` for the Lance stats path.

    :param tmp_path: Directory that becomes the Lance shard directory.
    :returns: ``tmp_path`` (the directory holding the seeded shard).
    """
    spec = build_lance_smoke_spec()
    write_minimal_lance_shard(tmp_path / spec.shards[0].filename, spec)
    return tmp_path


def test_get_stats_lance_writes_sibling_stats_npz_with_mel_inner_shape(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """A directory of ``shard-*.lance`` writes a ``stats.npz`` matching the mel inner shape.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    """
    _seed_lance_shard_dir(tmp_path)

    stats_script.get_stats_lance(str(tmp_path), mask_degenerate=True)

    stats_npz = tmp_path / "stats.npz"
    assert stats_npz.is_file()
    spec = build_lance_smoke_spec()
    mel_inner = dataset_field_shapes(spec.render, spec.num_params)[MEL_SPEC_FIELD][1:]
    with np.load(stats_npz) as loaded:
        assert set(loaded.files) == {"mean", "std"}
        assert loaded["mean"].shape == mel_inner
        assert loaded["mean"].dtype == np.float32
        assert loaded["std"].shape == mel_inner
        assert loaded["std"].dtype == np.float32


def test_get_stats_lance_no_shards_in_directory_raises(
    stats_script: ModuleType, tmp_path: Path
) -> None:
    """An empty directory (no ``shard-*.lance``) raises rather than writing partial stats.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir with no shards.
    """
    with pytest.raises(FileNotFoundError, match="shard-"):
        stats_script.get_stats_lance(str(tmp_path))


def test_cli_dispatches_directory_of_lance_shards_to_lance_path_with_default_flag(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` routes a ``shard-*.lance`` directory to ``get_stats_lance``, unmasked by default.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    _seed_lance_shard_dir(tmp_path)

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        stats_script,
        "get_stats_lance",
        lambda *a, **kw: calls.append(("lance", kw.get("mask_degenerate", False))),
    )
    monkeypatch.setattr(
        stats_script,
        "get_stats_directory",
        lambda *a, **kw: calls.append(("dir", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(tmp_path)])

    assert calls == [("lance", False)]


def test_cli_dispatches_directory_without_shards_to_audio_directory_path_with_default_flag(
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
        "get_stats_lance",
        lambda *a, **kw: calls.append(("lance", kw.get("mask_degenerate", False))),
    )
    monkeypatch.setattr(
        stats_script,
        "get_stats_directory",
        lambda *a, **kw: calls.append(("dir", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(tmp_path)])

    assert calls == [("dir", False)]


def test_cli_forwards_mask_degenerate_bins_flag_to_lance_path(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--mask-degenerate-bins`` forwards ``mask_degenerate=True`` to ``get_stats_lance``.

    :param stats_script: Imported stats module (fixture).
    :param tmp_path: pytest tmp dir used as the synthetic shard directory.
    :param monkeypatch: pytest fixture for swapping in spy callables.
    """
    _seed_lance_shard_dir(tmp_path)

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        stats_script,
        "get_stats_lance",
        lambda *a, **kw: calls.append(("lance", kw.get("mask_degenerate", False))),
    )

    stats_script.main([str(tmp_path), "--mask-degenerate-bins"])

    assert calls == [("lance", True)]


def test_cli_forwards_mask_degenerate_bins_flag_to_audio_directory_path(
    stats_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--mask-degenerate-bins`` forwards ``mask_degenerate=True`` to ``get_stats_directory``.

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
    """The CLI ``--help`` text documents the flag so operators can discover it.

    :param capsys: pytest fixture capturing argparse's ``--help`` output.
    """
    with pytest.raises(SystemExit) as exc_info:
        _stats_module._parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--mask-degenerate-bins" in captured.out
