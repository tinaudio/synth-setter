"""Session-scoped fixtures shared by ``tests/evaluation``.

The fixtures here are the canonical inputs the wandb-metrics plan's pin tests
(`Phase 0`) and the later refactor tests (Phases 1–2) share: ``fixture_audio_dir``
holds two ``sample_*/{pred.wav,target.wav}`` pairs that ``compute_audio_metrics``
ingests; ``fixture_pred_dir`` holds the ``pred-0.pt`` / ``target-audio-0.pt`` /
``target-params-0.pt`` tensors that ``predict_vst_audio`` ingests.

Materialised once per pytest session via ``tmp_path_factory.mktemp`` so the I/O
budget stays under the plan's 60s combined pin-test cap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pedalboard.io import AudioFile

from synth_setter.data.vst import param_specs

_SR = 44100
_DURATION_SECONDS = 1.0
_CHANNELS = 2

_PARAM_SPEC_NAME = "surge_simple"
_PARAM_SPEC = param_specs[_PARAM_SPEC_NAME]
_PRED_BATCH_SIZE = 2
_PRED_AUDIO_CHANNELS = 2
_PRED_AUDIO_SAMPLES = 1024


def _stereo_sine(freq: float, *, amplitude: float = 0.5) -> np.ndarray:
    """Constant-amplitude sine broadcast across ``_CHANNELS`` channels at ``_SR`` Hz.

    :param freq: Sine frequency in Hz.
    :param amplitude: Peak amplitude in linear units (not dB).
    :return: ``(_CHANNELS, N)`` float32 stereo audio array.
    """
    n = int(_DURATION_SECONDS * _SR)
    t = np.arange(n, dtype=np.float32) / _SR
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.broadcast_to(tone, (_CHANNELS, n)).copy()


def _stereo_sine_plus_noise(freq: float, *, noise_amplitude: float, seed: int) -> np.ndarray:
    """Sine at ``freq`` plus deterministic Gaussian noise — non-silent, non-identical pred.

    :param freq: Sine frequency in Hz.
    :param noise_amplitude: Standard deviation of the additive Gaussian noise.
    :param seed: RNG seed — keeps the test snapshot reproducible across runs.
    :return: ``(_CHANNELS, N)`` float32 stereo audio array.
    """
    base = _stereo_sine(freq)
    rng = np.random.default_rng(seed)
    noise = (noise_amplitude * rng.standard_normal(base.shape)).astype(np.float32)
    return base + noise


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Write ``(channels, N)`` float32 ``audio`` to ``path`` at ``_SR`` Hz.

    :param path: Destination filesystem path for the WAV.
    :param audio: ``(channels, N)`` float32 audio array.
    """
    channels = audio.shape[0]
    with AudioFile(str(path), "w", _SR, channels) as f:
        f.write(audio)


@pytest.fixture(scope="session")
def fixture_audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two ``sample_*/{pred.wav,target.wav}`` pairs — the canonical pin-test input.

    ``sample_0`` is an identical 440 Hz sine pair (zero-error baseline for the
    distance metrics, ``rms`` cosine of ``1``).
    ``sample_1`` is a 440 Hz sine target vs a 440 Hz sine + small Gaussian noise
    pred (positive-error case so aggregate ``std`` is non-zero).

    :param tmp_path_factory: Pytest fixture providing session-scoped tmp paths.
    :return: Path to the populated audio directory.
    """
    audio_dir = tmp_path_factory.mktemp("eval_audio")

    sample_0 = audio_dir / "sample_0"
    sample_0.mkdir()
    _write_wav(sample_0 / "target.wav", _stereo_sine(440.0))
    _write_wav(sample_0 / "pred.wav", _stereo_sine(440.0))

    sample_1 = audio_dir / "sample_1"
    sample_1.mkdir()
    _write_wav(sample_1 / "target.wav", _stereo_sine(440.0))
    _write_wav(sample_1 / "pred.wav", _stereo_sine_plus_noise(440.0, noise_amplitude=0.05, seed=2))

    return audio_dir


@pytest.fixture(scope="session")
def fixture_pred_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``pred-0.pt`` / ``target-audio-0.pt`` / ``target-params-0.pt`` for ``surge_simple``.

    Tensors are deterministic (seeded) so subsequent refactors of the loader path
    surface as test failures, not flakes. Encoded params live on ``[-1, 1]`` to
    match the CLI's ``(x + 1) / 2`` rescale + ``np.clip(..., 0, 1)`` path.
    Pred and target params are drawn from distinct RNG seeds so the resulting
    ``params.csv`` ``pred``/``target`` columns are guaranteed to differ row-wise.

    :param tmp_path_factory: Pytest fixture providing session-scoped tmp paths.
    :return: Path to the populated pred-tensor directory.
    """
    pred_dir = tmp_path_factory.mktemp("eval_pred")

    rng = np.random.default_rng(0)
    encoded = (rng.random((_PRED_BATCH_SIZE, len(_PARAM_SPEC))) * 2 - 1).astype(np.float32)
    torch.save(torch.from_numpy(encoded), pred_dir / "pred-0.pt")

    target_audio = rng.standard_normal(
        (_PRED_BATCH_SIZE, _PRED_AUDIO_CHANNELS, _PRED_AUDIO_SAMPLES)
    ).astype(np.float32)
    torch.save(torch.from_numpy(target_audio), pred_dir / "target-audio-0.pt")

    target_encoded = (
        np.random.default_rng(1).random((_PRED_BATCH_SIZE, len(_PARAM_SPEC))) * 2 - 1
    ).astype(np.float32)
    torch.save(torch.from_numpy(target_encoded), pred_dir / "target-params-0.pt")

    return pred_dir
