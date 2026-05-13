"""Tests for `synth_setter.data.stats_utils.compute_scale` (#998)."""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.data.stats_utils import compute_scale


def test_compute_scale_all_positive_std_returns_elementwise_reciprocal() -> None:
    """All-positive ``std`` produces element-wise ``1/std``."""
    std = np.array([0.5, 1.0, 2.0, 4.0])

    scale = compute_scale(std)

    np.testing.assert_allclose(scale, np.array([2.0, 1.0, 0.5, 0.25]))


def test_compute_scale_zero_entry_maps_to_zero_not_inf() -> None:
    """Degenerate ``std==0`` entries become ``0`` in the scale, not ``inf``."""
    std = np.array([0.5, 0.0, 2.0])

    scale = compute_scale(std)

    assert np.isfinite(scale).all()
    np.testing.assert_allclose(scale, np.array([2.0, 0.0, 0.5]))


def test_compute_scale_does_not_emit_divide_warning() -> None:
    """Computing scale with degenerate bins does not trigger numpy divide warnings."""
    std = np.array([1.0, 0.0, 0.0, 2.0])

    with np.errstate(divide="raise", invalid="raise"):
        scale = compute_scale(std)

    np.testing.assert_allclose(scale, np.array([1.0, 0.0, 0.0, 0.5]))


def test_compute_scale_preserves_dtype_as_float64() -> None:
    """Output is always ``float64`` regardless of input dtype."""
    std = np.array([0.5, 1.0, 2.0], dtype=np.float32)

    scale = compute_scale(std)

    assert scale.dtype == np.float64


def test_compute_scale_rejects_nan_std() -> None:
    """``NaN`` in ``std`` indicates corrupted stats; raise rather than silently mask."""
    std = np.array([0.5, np.nan, 2.0])

    with pytest.raises(ValueError, match="non-finite"):
        compute_scale(std)


def test_compute_scale_rejects_inf_std() -> None:
    """``inf`` in ``std`` indicates corrupted stats; raise rather than mapping to 0."""
    std = np.array([0.5, np.inf, 2.0])

    with pytest.raises(ValueError, match="non-finite"):
        compute_scale(std)


def test_compute_scale_rejects_negative_std() -> None:
    """Negative ``std`` violates the contract; raise rather than silently mask."""
    std = np.array([0.5, -1.0, 2.0])

    with pytest.raises(ValueError, match="negative"):
        compute_scale(std)
