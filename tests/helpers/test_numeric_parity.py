"""Behavior tests for the float32 parity assertion."""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.numeric_parity import assert_float32_parity


def _reference() -> np.ndarray:
    """Return a reference block spanning four decades, including near-zero coordinates.

    :returns: Float32 values shaped like a pooled embedding sequence.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(scale=2.5, size=(2, 64, 32)).astype(np.float32)
    values[0, 0, :] = np.float32(1e-7)
    return values


def test_assert_float32_parity_rounding_scale_difference_passes() -> None:
    """Accumulation-order noise is the difference the assertion exists to tolerate."""
    expected = _reference()
    noise = np.random.default_rng(1).normal(scale=1e-6, size=expected.shape)

    assert_float32_parity((expected + noise).astype(np.float32), expected, name="sequence")


def test_assert_float32_parity_near_zero_coordinates_pass_on_absolute_agreement() -> None:
    """A coordinate near zero must not fail for a large *relative* error alone.

    Element-wise ``rtol`` is what made the PupuJEPA parity test report a relative
    difference of 14 on coordinates worth 1e-7 — see #2834.
    """
    expected = _reference()
    actual = expected.copy()
    actual[0, 0, :] *= np.float32(3.0)

    assert_float32_parity(actual, expected, name="sequence")


def test_assert_float32_parity_uniform_drift_raises() -> None:
    """A systematic scale error is a real divergence at every magnitude."""
    expected = _reference()

    with pytest.raises(AssertionError, match="sequence"):
        assert_float32_parity(expected * np.float32(1.001), expected, name="sequence")


def test_assert_float32_parity_single_diverged_element_raises() -> None:
    """One wrong coordinate stays visible although the norm still agrees.

    The offset is a thousandth of the reference scale: spread over 4096 coordinates that is
    well inside the norm criterion, so only the element-wise term can catch it.
    """
    expected = _reference()
    actual = expected.copy()
    actual[1, 5, 7] += np.float32(np.abs(expected).max() * 1e-3)

    with pytest.raises(AssertionError, match="largest element"):
        assert_float32_parity(actual, expected, name="sequence")


def test_assert_float32_parity_zero_reference_requires_exact_match() -> None:
    """An all-zero reference has no magnitude to scale by, so only equality can pass."""
    expected = np.zeros((4, 3), dtype=np.float32)

    assert_float32_parity(expected.copy(), expected, name="sequence")
    with pytest.raises(AssertionError, match="zero"):
        assert_float32_parity(np.full((4, 3), 1e-9, dtype=np.float32), expected, name="sequence")


def test_assert_float32_parity_shape_mismatch_raises() -> None:
    """Comparing different shapes is a contract break, not a tolerance question."""
    expected = np.zeros((4, 3), dtype=np.float32)

    with pytest.raises(AssertionError, match="shape"):
        assert_float32_parity(np.zeros((4, 2), dtype=np.float32), expected, name="sequence")
