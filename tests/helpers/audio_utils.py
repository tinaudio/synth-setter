"""Deterministic audio generators and a WAV writer shared across ``tests/evaluation``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pedalboard.io import AudioFile

DEFAULT_SR = 44100


def sine(
    *,
    freq: float = 440.0,
    amplitude: float = 0.5,
    channels: int = 1,
    sr: float = DEFAULT_SR,
    samples: int | None = None,
    seconds: float | None = None,
) -> np.ndarray:
    """Constant-amplitude sine broadcast across ``channels`` — ``(channels, N)`` float32.

    Give exactly one of ``samples`` or ``seconds`` to set the length; ``seconds``
    is converted via ``round(seconds * sr)``.

    :param freq: Sine frequency in Hz.
    :param amplitude: Peak amplitude in linear units (not dB).
    :param channels: Number of output channels; the tone is broadcast across them.
    :param sr: Sample rate in Hz.
    :param samples: Length in samples; mutually exclusive with ``seconds``.
    :param seconds: Length in seconds; mutually exclusive with ``samples``.
    :return: ``(channels, N)`` float32 array.
    :raises ValueError: If neither or both of ``samples``/``seconds`` are given.
    """
    if samples is not None and seconds is None:
        n = samples
    elif seconds is not None and samples is None:
        n = round(seconds * sr)
    else:
        raise ValueError("pass exactly one of `samples` or `seconds`")
    t = np.arange(n, dtype=np.float32) / sr
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.broadcast_to(tone, (channels, n)).copy()


def noise(channels: int, samples: int, *, seed: int = 0) -> np.ndarray:
    """Deterministic ``(channels, samples)`` float32 Gaussian noise — non-silent input.

    :param channels: Number of output channels.
    :param samples: Length in samples per channel.
    :param seed: RNG seed — keeps generated audio reproducible across runs.
    :return: ``(channels, samples)`` float32 array.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((channels, samples)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, *, sr: int = DEFAULT_SR) -> None:
    """Write ``(channels, N)`` float32 ``audio`` to ``path`` as a WAV at ``sr`` Hz.

    :param path: Destination WAV path.
    :param audio: ``(channels, N)`` float32 audio; channel count read from ``audio.shape[0]``.
    :param sr: Sample rate in Hz written into the file header.
    """
    channels = audio.shape[0]
    with AudioFile(str(path), "w", sr, channels) as f:
        f.write(audio)
