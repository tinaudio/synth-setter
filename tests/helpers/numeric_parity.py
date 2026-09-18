"""Parity assertion for two float32 computations of one quantity."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Rounding headroom for one float32 quantity computed two ways, as a fraction of its scale.
DEFAULT_FLOAT32_REL_TOL = 1e-4


def assert_float32_parity(
    actual: ArrayLike,
    expected: ArrayLike,
    *,
    name: str,
    rel_tol: float = DEFAULT_FLOAT32_REL_TOL,
) -> None:
    """Assert two float32 computations of one quantity differ only by rounding.

    Fails if the shapes differ or if either error exceeds ``rel_tol``. Both the
    norm-relative error and the largest element error are scored against
    ``expected``'s own magnitude: near-zero coordinates then cannot fail on relative error
    alone, and one diverged element cannot hide inside an otherwise matching norm.

    :param actual: Values from the path under test.
    :param expected: Values from the reference path.
    :param name: Quantity label for the failure message.
    :param rel_tol: Tolerated error as a fraction of ``expected``'s magnitude.
    """
    reference = np.asarray(expected, dtype=np.float64)
    candidate = np.asarray(actual, dtype=np.float64)
    assert candidate.shape == reference.shape, (
        f"{name} shape {candidate.shape} does not match reference shape {reference.shape}"
    )

    error = np.abs(candidate - reference)
    scale = float(np.abs(reference).max())
    if scale == 0.0:
        assert not error.any(), (
            f"{name} reference is all zero, so parity requires equality; "
            f"largest element differs by {error.max():.3e}"
        )
        return

    relative_norm_error = float(np.linalg.norm(error.ravel()) / np.linalg.norm(reference.ravel()))
    relative_max_error = float(error.max() / scale)
    assert relative_norm_error <= rel_tol, (
        f"{name} differs beyond float32 rounding: relative norm error "
        f"{relative_norm_error:.3e} exceeds {rel_tol:.1e}"
    )
    assert relative_max_error <= rel_tol, (
        f"{name} differs beyond float32 rounding: largest element error "
        f"{error.max():.3e} is {relative_max_error:.3e} of the reference scale "
        f"{scale:.3e}, exceeding {rel_tol:.1e}"
    )
