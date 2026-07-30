"""Behaviour tests for simulator-feedback control signals and the controlled flow.

Every test renders real torchsynth audio and scores it with the production spectral distance;
nothing here substitutes a stand-in for the simulator or the cost.
"""

from functools import partial

import pytest
import torch

from synth_setter.data.torchsynth_grad_render import render_torchsynth_grad
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_FULL_PARAM_SPEC
from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
from synth_setter.models.components.simulator_control import (
    ControlledFlow,
    ControlNet,
    gradient_control_signal,
    learned_control_signal,
)
from synth_setter.models.components.vector_field import VectorField

_SAMPLE_RATE = 16_000
_SIGNAL_LENGTH = 8_192
_BATCH = 2
_WIDTH = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
_COND_DIM = 6


def _render():
    """Bind the production differentiable render to test geometry.

    :returns: Callable mapping decoded params to audio.
    """
    return partial(
        render_torchsynth_grad,
        sample_rate=_SAMPLE_RATE,
        signal_length=_SIGNAL_LENGTH,
        render_batch_size=_BATCH,
    )


def _theta(requires_grad: bool = True) -> torch.Tensor:
    """Build a mid-range estimate in the flow's model space.

    :param requires_grad: Whether the estimate participates in autograd.
    :returns: Estimate shaped ``(batch, width)``.
    """
    torch.manual_seed(0)
    theta = torch.zeros(_BATCH, _WIDTH)
    return theta.requires_grad_(requires_grad)


def _target() -> torch.Tensor:
    """Render a real target from a distinct estimate.

    :returns: Target audio shaped ``(batch, signal_length)``.
    """
    with torch.no_grad():
        return _render()(torch.full((_BATCH, _WIDTH), 0.25))


def test_gradient_signal_reports_cost_beside_its_gradient() -> None:
    """Equation 9 concatenates the cost with its parameter gradient."""
    signal = gradient_control_signal(
        theta_hat=_theta(),
        target_audio=_target(),
        render=_render(),
        cost=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
    )

    assert signal.shape == (_BATCH, 1 + _WIDTH)


def test_gradient_signal_is_not_a_loss_path() -> None:
    """The signal is a network input, so a graph through it would make training 2nd order."""
    signal = gradient_control_signal(
        theta_hat=_theta(),
        target_audio=_target(),
        render=_render(),
        cost=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
    )

    assert not signal.requires_grad


def test_gradient_signal_is_finite_for_a_silent_target() -> None:
    """Silent rows are common online and must not put NaN into a network input."""
    signal = gradient_control_signal(
        theta_hat=_theta(),
        target_audio=torch.zeros(_BATCH, _SIGNAL_LENGTH),
        render=_render(),
        cost=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
    )

    assert torch.isfinite(signal).all()


def test_gradient_signal_normalises_its_gradient_block() -> None:
    """Raw render gradients span orders of magnitude, which swamps the cost entry."""
    signal = gradient_control_signal(
        theta_hat=_theta(),
        target_audio=_target(),
        render=_render(),
        cost=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
    )

    norms = signal[:, 1:].norm(dim=-1)
    assert ((norms <= 1.0 + 1e-5) & (norms >= 0.0)).all()


def test_learned_signal_detaches_the_render() -> None:
    """Equation 10 stop-grads at the simulator, so a non-differentiable synth also works."""
    encoder = torch.nn.Linear(_SIGNAL_LENGTH, 4)

    signal = learned_control_signal(
        theta_hat=_theta(),
        target_audio=_target(),
        render=_render(),
        encoder=encoder,
    )

    assert signal.shape == (_BATCH, 4)
    assert signal.grad_fn is not None
    assert _theta().grad is None


def _controlled(t_min: float = 0.8) -> ControlledFlow:
    """Build a controlled flow over a frozen field and a zero-initialised control net.

    :param t_min: Flow time above which the control engages.
    :returns: Configured controlled flow.
    """
    torch.manual_seed(0)
    flow = VectorField(field_dim=_WIDTH, hidden_dim=16, conditioning_dim=_COND_DIM, num_blocks=1)
    flow.requires_grad_(False)
    return ControlledFlow(
        flow=flow,
        control=ControlNet(field_dim=_WIDTH, control_dim=1 + _WIDTH, hidden_dim=16),
        t_min=t_min,
    )


def _flow_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build field inputs shared by the controlled-flow tests.

    :returns: ``(x_t, z, control_input)``.
    """
    torch.manual_seed(1)
    return (
        torch.randn(_BATCH, _WIDTH),
        torch.randn(_BATCH, _COND_DIM),
        torch.randn(_BATCH, 1 + _WIDTH),
    )


def test_controlled_flow_is_an_identity_at_initialisation() -> None:
    """A zero-initialised control must not disturb the pretrained field on step one."""
    controlled = _controlled()
    x_t, z, control_input = _flow_inputs()
    t = torch.full((_BATCH, 1), 0.9)

    with torch.no_grad():
        assert torch.equal(controlled(x_t, t, z, control_input), controlled.flow(x_t, t, z))


def test_controlled_flow_below_t_min_bypasses_the_control() -> None:
    """Estimates are unreliable early, so those rows must return the pretrained velocity."""
    controlled = _controlled()
    with torch.no_grad():
        for parameter in controlled.control.parameters():
            parameter.add_(0.5)
    x_t, z, control_input = _flow_inputs()
    t = torch.full((_BATCH, 1), 0.5)

    with torch.no_grad():
        assert torch.equal(controlled(x_t, t, z, control_input), controlled.flow(x_t, t, z))


def test_controlled_flow_above_t_min_applies_the_control() -> None:
    """Above the threshold a non-zero control must change the velocity."""
    controlled = _controlled()
    with torch.no_grad():
        for parameter in controlled.control.parameters():
            parameter.add_(0.5)
    x_t, z, control_input = _flow_inputs()
    t = torch.full((_BATCH, 1), 0.9)

    with torch.no_grad():
        assert not torch.equal(controlled(x_t, t, z, control_input), controlled.flow(x_t, t, z))


def test_controlled_flow_mixes_bypassed_and_controlled_rows() -> None:
    """A batch straddling t_min must route each row on its own flow time."""
    controlled = _controlled()
    with torch.no_grad():
        for parameter in controlled.control.parameters():
            parameter.add_(0.5)
    x_t, z, control_input = _flow_inputs()
    t = torch.tensor([[0.5], [0.9]])

    with torch.no_grad():
        velocity = controlled(x_t, t, z, control_input)
        baseline = controlled.flow(x_t, t, z)

    assert torch.equal(velocity[0], baseline[0])
    assert not torch.equal(velocity[1], baseline[1])


def test_controlled_flow_rejects_a_trainable_field() -> None:
    """The paper freezes the pretrained field; a trainable one would defeat the finetune."""
    flow = VectorField(field_dim=_WIDTH, hidden_dim=16, conditioning_dim=_COND_DIM, num_blocks=1)

    with pytest.raises(ValueError, match="frozen"):
        ControlledFlow(
            flow=flow,
            control=ControlNet(field_dim=_WIDTH, control_dim=1 + _WIDTH, hidden_dim=16),
            t_min=0.8,
        )


def test_controlled_flow_trains_only_the_control() -> None:
    """Gradients must reach the control and never the frozen field."""
    controlled = _controlled()
    x_t, z, control_input = _flow_inputs()
    t = torch.full((_BATCH, 1), 0.9)

    controlled(x_t, t, z, control_input).square().mean().backward()

    assert all(p.grad is None for p in controlled.flow.parameters())
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in controlled.control.parameters()
    )
