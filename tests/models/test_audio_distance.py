"""Behaviour tests for the pluggable audio-distance spaces the feedback term measures in."""

import pytest
import torch

from synth_setter.evaluation.compute_audio_metrics import MEL_PARAMS
from synth_setter.models.components.audio_distance import (
    MEL_SCALES,
    LatentMseDistance,
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


class _ReshapeEncoder(torch.nn.Module):
    """Frozen fake encoder exposing waveform samples as a latent grid."""

    def __init__(self, latent_dim: int) -> None:
        """Fix the latent width the waveform is folded into.

        :param latent_dim: Number of latent channels.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.gain = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Fold each waveform into a latent grid without changing its values.

        :param audio: Waveform batch shaped ``(batch, samples)``.
        :returns: Latents shaped ``(batch, latent_dim, samples / latent_dim)``.
        """
        return (audio * self.gain).reshape(audio.shape[0], self.latent_dim, -1)


def _latent_mse() -> LatentMseDistance:
    """Build a latent distance over the value-preserving fake encoder.

    :returns: Configured distance module.
    """
    return LatentMseDistance(encoder=_ReshapeEncoder(4))


def test_latent_distance_of_identical_audio_is_zero() -> None:
    """A perfect render is the fixed point the term drives toward."""
    torch.manual_seed(0)
    audio = torch.randn(3, _LENGTH).clamp(-1.0, 1.0)

    assert _latent_mse()(audio, audio).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_latent_distance_separates_different_audio_per_sample() -> None:
    """The caller weights each row by its own flow time, so rows must not merge."""
    torch.manual_seed(0)
    rendered = torch.randn(4, _LENGTH).clamp(-1.0, 1.0)
    target = torch.randn(4, _LENGTH).clamp(-1.0, 1.0)

    distance = _latent_mse()(rendered, target)

    assert distance.shape == (4,)
    assert (distance > 0).all()


def test_latent_distance_is_invariant_to_target_magnitude() -> None:
    """Normalization is what stops loud targets from dominating quiet ones."""
    torch.manual_seed(0)
    target = torch.randn(2, _LENGTH).clamp(-1.0, 1.0)
    rendered = 0.5 * target

    quiet = _latent_mse()(rendered, target)
    loud = _latent_mse()(10.0 * rendered, 10.0 * target)

    torch.testing.assert_close(loud, quiet, rtol=1e-4, atol=1e-6)


def test_latent_distance_gradient_reaches_only_the_rendered_waveform() -> None:
    """The target is data; a gradient into it would train the wrong tensor."""
    torch.manual_seed(0)
    rendered = torch.randn(2, _LENGTH).clamp(-1.0, 1.0).requires_grad_(True)
    target = torch.randn(2, _LENGTH).clamp(-1.0, 1.0).requires_grad_(True)

    _latent_mse()(rendered, target).sum().backward()

    assert rendered.grad is not None
    assert torch.count_nonzero(rendered.grad) > 0
    assert target.grad is None


def test_latent_distance_of_a_constant_target_stays_finite() -> None:
    """Silent targets have zero latent variance and must not divide by it."""
    rendered = torch.full((2, _LENGTH), 0.25, requires_grad=True)

    distance = _latent_mse()(rendered, torch.zeros(2, _LENGTH))
    distance.sum().backward()

    assert torch.isfinite(distance).all()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()


def test_latent_distance_rejects_a_trainable_encoder() -> None:
    """A trainable space could shrink the distance without improving the render."""
    with pytest.raises(ValueError, match="frozen"):
        LatentMseDistance(encoder=_ReshapeEncoder(4).requires_grad_(True))
