"""Shared numeric helpers for normalization statistics."""

from __future__ import annotations

import numpy as np


def compute_scale(std: np.ndarray) -> np.ndarray:
    """Return ``1/std`` with degenerate bins (``std == 0``) masked to 0.

    Datamodules apply normalization as ``(spec - mean) * scale``; masking the
    reciprocal turns a zero-variance bin into a constant-zero contribution rather
    than an ``inf``/``nan`` divide-by-zero. The stats writer
    (``scripts/get_dataset_stats.py``) surfaces degenerate bins to the operator
    before they reach this stage — see #998.

    :param std: Per-bin standard deviation array.

    :returns: Per-bin scale array, the same shape as ``std``, where each entry
        is ``1/std`` when ``std > 0`` and ``0`` otherwise. Always ``float64``.
    :rtype: np.ndarray
    """
    return np.divide(
        1.0, std, out=np.zeros_like(std, dtype=np.float64), where=std > 0
    )
