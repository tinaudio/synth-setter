"""Smoke tests for the #2557 torchsynth-x-m2l simulator-feedback spike prototypes.

The spike code lives outside ``src`` under ``prototypes/``; the helper below
puts the repo root on ``sys.path`` before importing it (declared spike — see
issue #2557 for the TDD deviation). Tests avoid the music2latent checkpoint so
they run without network access; the safe STFT frontend is exercised directly.
"""

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_spike():
    """Import the spike modules with the repo root on ``sys.path``.

    :returns: The ``grad_render``, ``m2l_grad``, and ``flow_m2l`` prototype modules.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from prototypes.torchsynth_feedback_m2l import flow_m2l, grad_render, m2l_grad

    return grad_render, m2l_grad, flow_m2l


def test_render_torchsynth_grad_random_batch_propagates_finite_gradients():
    """A grad-enabled render backprops finite, non-zero gradients to the params."""
    grad_render, _, _ = _import_spike()
    torch.manual_seed(0)
    params = torch.rand((4, 76), requires_grad=True)
    target = torch.rand((4, 76))
    with torch.no_grad():
        target_audio = grad_render.render_torchsynth_grad(
            target, sample_rate=44_100, signal_length=4_410, midi_pitch=60
        )
    audio = grad_render.render_torchsynth_grad(
        params, sample_rate=44_100, signal_length=4_410, midi_pitch=60
    )
    assert audio.shape == (4, 4_410)
    grad_render.log_spectral_distance(audio, target_audio).sum().backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()
    assert (params.grad != 0).any()


def test_safe_representation_encoder_nonsilent_audio_matches_original_frontend():
    """The NaN-safe STFT frontend reproduces music2latent's representation."""
    pytest.importorskip("music2latent")
    _, m2l_grad, _ = _import_spike()
    from music2latent.audio import to_representation_encoder

    torch.manual_seed(0)
    audio = 0.3 * torch.randn(2, 8_192)
    original = to_representation_encoder(audio)
    safe = m2l_grad._safe_representation_encoder(audio)
    assert safe.shape == original.shape
    assert (original - safe).abs().max().item() < 1e-4


def test_safe_representation_encoder_silent_audio_backprops_finite_gradients():
    """Silent audio (zero STFT bins) yields finite gradients through the safe frontend.

    The stock ``normalize_complex`` produces NaN gradients here — the exact
    failure Step A hit on near-silent synth renders.
    """
    pytest.importorskip("music2latent")
    _, m2l_grad, _ = _import_spike()
    audio = torch.zeros(1, 8_192, requires_grad=True)
    representation = m2l_grad._safe_representation_encoder(audio)
    representation.square().sum().backward()
    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()


def test_control_field_forward_zero_init_returns_zero_correction():
    """A freshly built control field is the identity correction (zero output)."""
    _, _, flow_m2l = _import_spike()
    control = flow_m2l.ControlField()
    t = torch.rand(3, 1)
    velocity = torch.randn(3, 76)
    signal = torch.randn(3, 77)
    correction = control(t, velocity, signal)
    assert correction.shape == (3, 76)
    assert torch.all(correction == 0)


def test_m2l_pooled_l2_identical_latents_returns_zero():
    """Pooled-L2 distance of a latent with itself is exactly zero."""
    _, m2l_grad, _ = _import_spike()
    latents = torch.randn(3, 64, 5)
    distance = m2l_grad.m2l_pooled_l2(latents, latents)
    assert distance.shape == (3,)
    assert torch.all(distance == 0)
