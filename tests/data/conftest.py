"""Shared fixtures for data-layer tests."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def source_file(tmp_path: Path) -> tuple[Path, str]:
    """Write a checksum-pinned lossless source with the pyFDN production geometry.

    :param tmp_path: Temporary directory owned by pytest.
    :returns: Source path and SHA-256 of its exact stored bytes.
    """
    path = tmp_path / "source.wav"
    time = np.arange(192_000, dtype=np.float64) / 48_000.0
    source = 0.1 * np.sin(2.0 * np.pi * 220.0 * time)
    sf.write(path, source, 48_000, subtype="PCM_16")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()
