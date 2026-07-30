"""Behaviour tests for simulator-feedback control signals and the controlled flow.

Every test renders real torchsynth audio and scores it with the production spectral distance;
nothing here substitutes a stand-in for the simulator or the cost.
"""

from functools import partial

import numpy as np
import pytest
import torch

from synth_setter.data.torchsynth_grad_render import render_torchsynth_grad
from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_FULL_PARAM_SPEC
from synth_setter.models.components.audio_distance import MultiScaleSpectralDistance
from synth_setter.models.components.simulator_control import (
    ControlledFlow,
    ControlNet,
    RenderFn,
    gradient_control_signal,
    learned_control_signal,
)
from synth_setter.models.components.vector_field import VectorField

_SAMPLE_RATE = 16_000
_SIGNAL_LENGTH = 8_192
_BATCH = 2
_WIDTH = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
_COND_DIM = 6
_BUFFER_SECONDS = _SIGNAL_LENGTH / _SAMPLE_RATE
# A render this quiet is silence; scoring it against a silent target agrees for the wrong
# reason and leaves the cost and its gradient identically zero.
_AUDIBLE_PEAK = 1e-4


def _sounding_note_columns() -> torch.Tensor:
    """Encode a note that starts at zero and sounds across the whole render buffer.

    The spec draws ``note_start_and_end`` across a multi-second range, so a uniform-random
    row on this short buffer starts its note past the end and renders silence.

    :returns: The encoded note tail shaped ``(note columns,)``.
    """
    synth_values, _ = TORCHSYNTH_FULL_PARAM_SPEC.sample(np.random.default_rng(0))
    reference = TORCHSYNTH_FULL_PARAM_SPEC.encode(
        synth_values, {"pitch": 60, "note_start_and_end": (0.0, _BUFFER_SECONDS)}
    )
    return torch.from_numpy(reference)[TORCHSYNTH_FULL_PARAM_SPEC.synth_columns.stop :]


def _audible_rows(value: float) -> torch.Tensor:
    """Build rows at a fixed synth setting whose note actually sounds.

    :param value: Constant every synth column takes, in ``[0, 1]``.
    :returns: Encoded rows shaped ``(batch, encoded_width)``.
    """
    synth = torch.full((_BATCH, TORCHSYNTH_FULL_PARAM_SPEC.synth_param_length), value)
    return torch.cat([synth, _sounding_note_columns().expand(_BATCH, -1)], dim=1)


def _render() -> RenderFn:
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
    """Build a mid-range estimate that renders audible audio.

    :param requires_grad: Whether the estimate participates in autograd.
    :returns: Estimate shaped ``(batch, width)``.
    """
    return _audible_rows(0.5).requires_grad_(requires_grad)


def _target() -> torch.Tensor:
    """Render a real target from a distinct, audible estimate.

    :returns: Target audio shaped ``(batch, signal_length)``.
    """
    with torch.no_grad():
        target = _render()(_audible_rows(0.25))
    assert target.abs().max() > _AUDIBLE_PEAK, "target is silent; the suite would be vacuous"
    return target


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
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_learned_signal_detaches_the_render() -> None:
    """Equation 10 stop-grads at the simulator, so a non-differentiable synth also works."""
    encoder = torch.nn.Linear(_SIGNAL_LENGTH, 4)
    theta = _theta()

    signal = learned_control_signal(
        theta_hat=theta,
        target_audio=_target(),
        render=_render(),
        encoder=encoder,
    )
    signal.sum().backward()

    assert signal.shape == (_BATCH, 4)
    assert theta.grad is None
    assert all(
        p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in encoder.parameters()
    )


def _controlled(t_min: float = 0.8) -> ControlledFlow:
    """Build a controlled flow over a frozen field and a zero-initialised control net.

    :param t_min: Flow time above which the control engages.
    :returns: Controlled flow over a frozen field.
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
        assert torch.equal(
            controlled(x_t, t, z, control_input=control_input), controlled.flow(x_t, t, z)
        )


def test_controlled_flow_below_t_min_bypasses_the_control() -> None:
    """Estimates are unreliable early, so those rows must return the pretrained velocity."""
    controlled = _controlled()
    with torch.no_grad():
        for parameter in controlled.control.parameters():
            parameter.add_(0.5)
    x_t, z, control_input = _flow_inputs()
    t = torch.full((_BATCH, 1), 0.5)

    with torch.no_grad():
        assert torch.equal(
            controlled(x_t, t, z, control_input=control_input), controlled.flow(x_t, t, z)
        )


def test_controlled_flow_above_t_min_applies_the_control() -> None:
    """Above the threshold a non-zero control must change the velocity."""
    controlled = _controlled()
    with torch.no_grad():
        for parameter in controlled.control.parameters():
            parameter.add_(0.5)
    x_t, z, control_input = _flow_inputs()
    t = torch.full((_BATCH, 1), 0.9)

    with torch.no_grad():
        assert not torch.equal(
            controlled(x_t, t, z, control_input=control_input), controlled.flow(x_t, t, z)
        )


def test_controlled_flow_mixes_bypassed_and_controlled_rows() -> None:
    """A batch straddling t_min must route each row on its own flow time."""
    controlled = _controlled()
    with torch.no_grad():
        for parameter in controlled.control.parameters():
            parameter.add_(0.5)
    x_t, z, control_input = _flow_inputs()
    t = torch.tensor([[0.5], [0.9]])

    with torch.no_grad():
        velocity = controlled(x_t, t, z, control_input=control_input)
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

    controlled(x_t, t, z, control_input=control_input).square().mean().backward()

    assert all(p.grad is None for p in controlled.flow.parameters())
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in controlled.control.parameters()
    )


def test_gradient_signal_inside_no_grad_still_produces_a_gradient() -> None:
    """The ODE integration loop runs under no_grad; requires_grad survives it, the graph does not."""
    with torch.no_grad():
        signal = gradient_control_signal(
            theta_hat=_theta(),
            target_audio=_target(),
            render=_render(),
            cost=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
        )

    assert torch.count_nonzero(signal[:, 1:]) > 0


def test_gradient_signal_sanitises_a_non_finite_cost() -> None:
    """One NaN in the cost column reaches every row through the control network."""

    class _NonFiniteCost(torch.nn.Module):
        """Cost returning NaN, standing in for a blown-up render."""

        def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Return a NaN per row that still carries a graph.

            :param rendered: Rendered audio.
            :param target: Target audio.
            :returns: Per-row NaN cost.
            """
            return rendered.mean(dim=-1) * float("nan")

    signal = gradient_control_signal(
        theta_hat=_theta(),
        target_audio=_target(),
        render=_render(),
        cost=_NonFiniteCost(),
    )

    assert torch.isfinite(signal).all()


def test_gradient_signal_compresses_the_cost_beside_the_gradient_block() -> None:
    """A raw dB cost outweighs a unit-norm gradient block by orders of magnitude."""
    signal = gradient_control_signal(
        theta_hat=_theta(),
        target_audio=_target(),
        render=_render(),
        cost=MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE),
    )

    raw = MultiScaleSpectralDistance(sample_rate=_SAMPLE_RATE)(
        _render()(_theta(requires_grad=False)), _target()
    )
    assert torch.allclose(signal[:, 0], torch.log1p(raw.clamp_min(0.0)), atol=1e-5)
    # The point of the compression: it lands on the gradient block's own scale.
    assert (signal[:, 0] < raw).all()


def test_learned_signal_does_not_backpropagate_into_the_target() -> None:
    """A differentiable upstream target would otherwise be trained by the control signal."""
    target = _target().requires_grad_(True)

    learned_control_signal(
        theta_hat=_theta(),
        target_audio=target,
        render=_render(),
        encoder=torch.nn.Linear(_SIGNAL_LENGTH, 4),
    ).sum().backward()

    assert target.grad is None


def test_control_net_accepts_a_rank_one_control() -> None:
    """A cost-only signal is naturally shaped (batch,), which flatten(start_dim=1) rejects."""
    net = ControlNet(field_dim=_WIDTH, control_dim=1, hidden_dim=8)

    correction = net(
        torch.full((_BATCH, 1), 0.9), torch.randn(_BATCH, _WIDTH), torch.randn(_BATCH)
    )

    assert correction.shape == (_BATCH, _WIDTH)


def test_control_net_with_a_mismatched_control_width_raises() -> None:
    """A width mismatch must name itself rather than surface as a bare matmul error."""
    net = ControlNet(field_dim=_WIDTH, control_dim=4, hidden_dim=8)

    with pytest.raises(ValueError, match="width"):
        net(torch.full((_BATCH, 1), 0.9), torch.randn(_BATCH, _WIDTH), torch.randn(_BATCH, 5))


def test_controlled_flow_bypassed_row_survives_a_non_finite_control() -> None:
    """A bypassed row's NaN must not reach the control gradients through the GELU backward."""
    controlled = _controlled()
    x_t, z, control_input = _flow_inputs()
    control_input[0] = float("nan")
    t = torch.tensor([[0.5], [0.9]])

    controlled(x_t, t, z, control_input=control_input).square().mean().backward()

    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in controlled.control.parameters()
    )


def test_controlled_flow_bypassed_row_preserves_negative_zero() -> None:
    """Adding a zero correction to a negative-zero velocity would flip its sign bit."""
    controlled = _controlled()
    x_t, z, control_input = _flow_inputs()
    t = torch.full((_BATCH, 1), 0.5)

    with torch.no_grad():
        baseline = controlled.flow(x_t, t, z)
        velocity = controlled(x_t, t, z, control_input=control_input)

    assert torch.equal(torch.signbit(velocity), torch.signbit(baseline))


def test_controlled_flow_train_reasserts_the_freeze() -> None:
    """The field is a submodule, so a later requires_grad_ or unfreeze can reach it."""
    controlled = _controlled()
    controlled.requires_grad_(True)

    controlled.train()

    assert not any(p.requires_grad for p in controlled.flow.parameters())
    assert not controlled.flow.training
    assert controlled.control.training


def test_control_net_overfits_a_single_batch() -> None:
    """A disconnected hidden path still emits gradient; only optimisation proves it learns."""
    torch.manual_seed(0)
    net = ControlNet(field_dim=_WIDTH, control_dim=4, hidden_dim=32)
    t = torch.full((_BATCH, 1), 0.9)
    velocity = torch.randn(_BATCH, _WIDTH)
    control = torch.randn(_BATCH, 4)
    wanted = torch.randn(_BATCH, _WIDTH)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    initial = (net(t, velocity, control) - wanted).square().mean().item()
    for _ in range(300):
        optimizer.zero_grad()
        loss = (net(t, velocity, control) - wanted).square().mean()
        loss.backward()
        optimizer.step()

    assert loss.item() < initial * 0.01
