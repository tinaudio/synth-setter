"""Smoke tests for the #2553 torchsynth simulator-feedback spike prototypes.

The spike code lives outside ``src`` under ``prototypes/``; the helper below
puts the repo root on ``sys.path`` before importing it (declared spike — see
issue #2553 for the TDD deviation).
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_spike():
    """Import the spike modules with the repo root on ``sys.path``.

    :returns: The ``grad_render`` and ``flow`` prototype modules.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from prototypes.torchsynth_feedback import flow, grad_render

    return grad_render, flow


def test_render_torchsynth_grad_random_batch_propagates_finite_gradients():
    """A grad-enabled render backprops finite, non-zero gradients to the params."""
    grad_render, _ = _import_spike()
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


def test_control_field_forward_zero_init_returns_zero_correction():
    """A freshly built control field is the identity correction (zero output)."""
    _, flow = _import_spike()
    control = flow.ControlField()
    t = torch.rand(3, 1)
    velocity = torch.randn(3, 76)
    signal = torch.randn(3, 77)
    correction = control(t, velocity, signal)
    assert correction.shape == (3, 76)
    assert torch.all(correction == 0)
