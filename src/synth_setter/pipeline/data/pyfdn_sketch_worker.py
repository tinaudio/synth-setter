"""Process-pool worker entry for the pyFDN reverb-sketch extractor.

Spawn-context pool workers import this module when unpickling the mapped callable, so its
import chain must stay free of ``lance`` (which is not fork- or multi-process-safe here)
and of the rest of the backfill pipeline.
"""

from __future__ import annotations

import warnings

import numpy as np

from synth_setter.features.pyfdn_controls import extract_reverb_sketch

__all__ = ["extract_reverb_sketch_row"]


def extract_reverb_sketch_row(ir: np.ndarray, sample_rate: float) -> np.ndarray:
    """Extract one row's temporal sketch, silencing the mixing-time warning.

    pyFDN warns when no mixing time is found; the scalar is discarded and warning dedup is per-
    process, so an unsuppressed pool would print it once per worker.

    :param ir: One mono impulse response.
    :param sample_rate: Response sample rate in Hz.
    :returns: Stacked control matrix from the public extractor, numerically unchanged.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Mixing time not found", category=UserWarning
        )
        return extract_reverb_sketch(ir, sample_rate)
