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

    Corrupted stats (``NaN``, ``inf``, or negative ``std``) are rejected loudly
    rather than silently zeroed; the masking branch is for legitimately
    zero-variance bins only.

    :param std: Per-bin standard deviation array.

    :returns: Per-bin scale array, the same shape as ``std``, where each entry
        is ``1/std`` when ``std > 0`` and ``0`` otherwise. Always ``float64``.
    :rtype: np.ndarray
    :raises ValueError: If ``std`` contains ``NaN``, ``inf``, or negative values.
    """
    std = np.asarray(std)
    if not np.isfinite(std).all():
        raise ValueError(
            "compute_scale received non-finite std values (NaN or inf); "
            "stats.npz is corrupted. Regenerate it with "
            "scripts/get_dataset_stats.py."
        )
    if (std < 0).any():
        raise ValueError(
            "compute_scale received negative std values; stats.npz is "
            "corrupted (std must be non-negative). Regenerate it with "
            "scripts/get_dataset_stats.py."
        )
    return np.divide(
        1.0, std, out=np.zeros_like(std, dtype=np.float64), where=std > 0
    )
