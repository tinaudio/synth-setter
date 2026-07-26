"""Inputs and provenance shared by SAME parity tests and regeneration."""

from pathlib import Path

import numpy as np

SAME_REFERENCE_DIR = Path(__file__).parents[1] / "fixtures" / "same"
SAME_REFERENCE_RANDOM_SEED = 0
SAME_REFERENCE_ROWS = 2

SAME_HF_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "same_s": (
        "stabilityai/SAME-S",
        "fbeb3dcf53a326e5682f38e22e7f740202d44232",
    ),
    "same_l": (
        "stabilityai/SAME-L",
        "41acf79dd242877d6499a1108ca5dba5d5eecfc5",
    ),
}


def same_reference_audio(sample_rate: int) -> np.ndarray:
    """Build two deterministic stereo chirps at the requested sample rate.

    :param sample_rate: Samples per second.
    :returns: ``(2, 2, sample_rate)`` float32 audio.
    """
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    rows = []
    for start_hz, end_hz in ((110.0, 880.0), (1760.0, 220.0)):
        frequency = start_hz + (end_hz - start_hz) * time
        phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
        rows.append(np.stack((np.sin(phase), np.sin(phase * 1.01))))
    return (0.5 * np.stack(rows)).astype(np.float32)
