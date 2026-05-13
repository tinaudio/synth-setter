"""Unit tests for ``synth_setter.evaluation.compute_audio_metrics``."""

import numpy as np
import pytest

from synth_setter.evaluation.compute_audio_metrics import compute_rms

_SR = 44100


def _sine(seconds: float = 1.0, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    """A mono ``(1, N)`` sine — the shape ``compute_rms`` expects."""
    t = np.arange(int(seconds * _SR), dtype=np.float32) / _SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).reshape(1, -1)


def test_compute_rms_identical_signal_returns_one() -> None:
    """``cosine_sim(x, x)`` of a non-silent signal is ``1.0``."""
    audio = _sine()
    rms = compute_rms(audio, audio)
    assert np.isfinite(rms)
    assert rms == pytest.approx(1.0, abs=1e-6)


def test_compute_rms_silent_pred_returns_zero_not_nan() -> None:
    """Silent pred → ``pred_norm == 0`` → clamped denominator → ``cosine_sim = 0``.

    Regression guard: prior to the denominator clamp, this produced ``0/0 = NaN`` and
    poisoned downstream metric aggregation. See the MPS flake on
    ``test_train_eval_surge_xt[mps]`` where a 1-step-trained model can predict params
    that Surge XT renders as bit-silent audio.
    """
    target = _sine()
    pred = np.zeros_like(target)
    rms = compute_rms(target, pred)
    assert np.isfinite(rms), f"compute_rms produced non-finite {rms!r} for silent pred"
    assert rms == 0.0


def test_compute_rms_quiet_nonzero_inputs_return_zero() -> None:
    """Quiet (but non-zero) inputs whose ``target_norm * pred_norm < 1e-12`` return 0.

    Without the explicit short-circuit, the pre-fix path of
    ``dot(target_rms, pred_rms) / np.clip(denom, 1e-12, None)`` would return ~``0.4``
    here (numerator and clamped denominator both ≈ ``4e-13``), contradicting the
    warning's "returning 0" claim — see the Copilot review on PR #899.
    """
    quiet = np.full((1, _SR), 1e-7, dtype=np.float64)
    rms = compute_rms(quiet, quiet)
    assert rms == 0.0
