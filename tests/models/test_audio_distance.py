"""Behaviour tests for the pluggable audio-distance spaces the feedback term measures in."""

import pytest
import torch

from synth_setter.evaluation.compute_audio_metrics import MEL_PARAMS
from synth_setter.models.components.audio_distance import (
    MEL_SCALES,
    MultiScaleSpectralDistance,
)

_SAMPLE_RATE = 16000
_LENGTH = 8192


def _mss() -> MultiScaleSpectralDistance:
    """Build a multi-scale spectral distance at test geometry.

    :returns: Configured distance module.
    """
    return MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE)


def test_scales_match_the_reported_evaluation_metric() -> None:
    """Training and reporting must not drift onto different resolutions."""
    assert MEL_SCALES == tuple(tuple(entry) for entry in MEL_PARAMS)


def test_identical_audio_has_zero_distance() -> None:
    """A perfect render is the fixed point the term drives toward."""
    torch.manual_seed(0)
    audio = torch.randn(3, _LENGTH).clamp(-1.0, 1.0)

    assert _mss()(audio, audio).abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_different_audio_has_positive_distance() -> None:
    """Distinct spectra must be separated, not collapsed."""
    torch.manual_seed(0)
    rendered = torch.randn(3, _LENGTH).clamp(-1.0, 1.0)
    target = torch.zeros(3, _LENGTH)

    assert (_mss()(rendered, target) > 0).all()


def test_distance_is_reported_per_sample() -> None:
    """The caller weights each row by its own flow time."""
    torch.manual_seed(0)
    rendered = torch.randn(4, _LENGTH).clamp(-1.0, 1.0)
    target = torch.randn(4, _LENGTH).clamp(-1.0, 1.0)

    assert _mss()(rendered, target).shape == (4,)


def test_gradient_reaches_the_rendered_waveform() -> None:
    """Without this the term cannot train the field that produced the render."""
    torch.manual_seed(0)
    rendered = torch.randn(2, _LENGTH).clamp(-1.0, 1.0).requires_grad_(True)
    target = torch.zeros(2, _LENGTH)

    _mss()(rendered, target).sum().backward()

    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert torch.count_nonzero(rendered.grad) > 0


def test_silent_pair_stays_finite() -> None:
    """Silent renders are common online and must not poison the gradient."""
    silent = torch.zeros(2, _LENGTH, requires_grad=True)

    distance = _mss()(silent, torch.zeros(2, _LENGTH))
    distance.sum().backward()

    assert torch.isfinite(distance).all()
    assert silent.grad is not None
    assert torch.isfinite(silent.grad).all()
