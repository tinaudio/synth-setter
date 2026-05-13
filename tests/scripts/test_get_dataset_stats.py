"""Tests for `scripts/get_dataset_stats.py` degenerate-bin handling (#998)."""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "get_dataset_stats.py"


def _load_stats_script() -> ModuleType:
    """Import ``scripts/get_dataset_stats.py`` outside the package path.

    The script lives under ``scripts/`` which isn't on ``pythonpath`` (only
    ``src/`` is), so use ``importlib.util`` to import it by path without
    polluting ``sys.path``.

    :returns: The freshly imported script module.
    :rtype: ModuleType
    """
    spec = importlib.util.spec_from_file_location("_get_dataset_stats", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stats_script() -> ModuleType:
    """Module-scoped handle to the imported stats script.

    :returns: The imported script module shared across this file's tests.
    :rtype: ModuleType
    """
    return _load_stats_script()


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


def test_check_degenerate_bins_no_zeros_returns_unchanged(
    stats_script: ModuleType,
) -> None:
    """All-positive ``std`` is returned unchanged in both modes (no raise, no warning).

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.array([0.1, 0.5, 2.0])

    returned_strict = stats_script._check_degenerate_bins(std, mask_degenerate=False)
    returned_masked = stats_script._check_degenerate_bins(std, mask_degenerate=True)

    np.testing.assert_array_equal(returned_strict, std)
    np.testing.assert_array_equal(returned_masked, std)


def test_check_degenerate_bins_zero_entry_raises_when_not_masked(
    stats_script: ModuleType,
) -> None:
    """A single ``std==0`` entry raises ``ValueError`` naming the index.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.array([0.1, 0.0, 0.5])

    with pytest.raises(ValueError, match=r"zero variance.*indices \[1\]"):
        stats_script._check_degenerate_bins(std, mask_degenerate=False)


def test_check_degenerate_bins_zero_entry_substitutes_unit_std_and_warns(
    stats_script: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    """A single ``std==0`` entry with masking returns std with 1.0 substituted and logs the index.

    :param stats_script: Imported get_dataset_stats module (fixture).
    :param caplog: pytest log-capture fixture.
    """
    std = np.array([0.1, 0.0, 0.5])

    with caplog.at_level(logging.WARNING):
        returned = stats_script._check_degenerate_bins(std, mask_degenerate=True)

    np.testing.assert_array_equal(returned, np.array([0.1, 1.0, 0.5]))
    assert any("[1]" in record.message for record in caplog.records), caplog.text


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
        stats_script._check_degenerate_bins(std, mask_degenerate=False)


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
        stats_script._check_degenerate_bins(std, mask_degenerate=False)


def test_check_degenerate_bins_masking_preserves_input_dtype(
    stats_script: ModuleType,
) -> None:
    """Masking float32 ``std`` keeps it float32 — substituted entries do not promote to float64.

    Real ``stats.npz`` files written from torch tensors / HDF5 are float32; a
    ``np.where(std == 0, 1.0, std)`` substitution would promote the literal
    1.0 and silently double the on-disk size of the artifact.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    std = np.array([0.1, 0.0, 0.5], dtype=np.float32)

    returned = stats_script._check_degenerate_bins(std, mask_degenerate=True)

    assert returned.dtype == np.float32
    np.testing.assert_array_equal(returned, np.array([0.1, 1.0, 0.5], dtype=np.float32))


def test_finalize_with_at_most_one_sample_raises_distinct_error(stats_script: ModuleType) -> None:
    """Count<=1 (empty state or single sample) raises a distinct ``<=1 samples`` error.

    The pre-existing ``M2 / count if count > 1 else 0`` line in ``finalize``
    returns a scalar variance for these cases, which the degeneracy check
    surfaces as a "scalar / <=1 samples" failure rather than the per-bin
    "zero variance" path that requires a populated std array.

    :param stats_script: Imported get_dataset_stats module (fixture).
    """
    empty_existing = (0, 0, 0)
    sample = np.array([0.1, 0.2, 0.3])
    single_sample_existing = stats_script.update((0, np.zeros(3), np.zeros(3)), sample)

    with pytest.raises(ValueError, match=r"<=1 samples"):
        stats_script.finalize(empty_existing)
    with pytest.raises(ValueError, match=r"<=1 samples"):
        stats_script.finalize(single_sample_existing)


def test_cli_help_advertises_mask_degenerate_bins_flag() -> None:
    """The CLI ``--help`` text documents the new flag so operators can discover it."""
    # 60s rather than 30s: under `make test-fast`'s parallel xdist load,
    # rootutils.setup_root walking the project tree from many workers at
    # once stretches script import time past the conservative default.
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--mask-degenerate-bins" in result.stdout
