"""Simulator-feedback control signals and the controlled flow that consumes them.

Implements the finetuning scheme of "Flow Matching for Posterior Inference with Simulator Feedbac
(Holzschuh & Thuerey, "
https://arxiv.org/abs/2410.22573):
a frozen pretrained flow
supplies a velocity, a simulator scores the one-step estimate that velocity implies, and a
small control network aggregates the two. The control signal is an *input*, never a loss
term, which is what leaves the plain flow-matching objective and its guarantees intact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

# Guards the per-row gradient normalizer; render gradients span ~9 orders of magnitude, so
# the unnormalized block would swamp the cost entry beside it.
_GRAD_NORM_EPS: Final = 1e-12
# Flow time below which the paper finds the one-step estimate too poor to control on.
DEFAULT_CONTROL_T_MIN: Final = 0.8

_BATCH_PARAMS_SHAPE = "batch params"
_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_TIME_SHAPE = "batch 1"
_BATCH_ANY_SHAPE = "batch ..."

type RenderFn = Callable[[Float[Tensor, _BATCH_PARAMS_SHAPE]], Float[Tensor, _BATCH_AUDIO_SHAPE]]


@jaxtyped(typechecker=beartype)
def gradient_control_signal(
    *,
    theta_hat: Float[Tensor, _BATCH_PARAMS_SHAPE],
    target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
    render: RenderFn,
    cost: nn.Module,
) -> Float[Tensor, _BATCH_ANY_SHAPE]:
    """Score the estimate and report the cost beside its parameter gradient.

    The paper's equation 9. Requires a differentiable simulator and cost; use
    :func:`learned_control_signal` when either is unavailable.

    :param theta_hat: One-step parameter estimate carrying ``requires_grad``.
    :param target_audio: Observed audio the estimate is scored against.
    :param render: Differentiable simulator over decoded parameters.
    :param cost: Module mapping ``(rendered, target)`` to a per-sample cost.
    :returns: Signal shaped ``(batch, 1 + params)``, detached and finite.
    :raises ValueError: ``theta_hat`` cannot carry a gradient, so no signal exists.
    """
    if not theta_hat.requires_grad:
        raise ValueError("theta_hat must require grad for a gradient control signal")
    per_sample = cost(render(theta_hat), target_audio)
    # create_graph=False: the signal is an input, and a graph here would silently make the
    # finetune second-order.
    (gradient,) = torch.autograd.grad(per_sample.sum(), theta_hat, create_graph=False)
    gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    scale = gradient.norm(dim=-1, keepdim=True).clamp_min(_GRAD_NORM_EPS)
    return torch.cat((per_sample.detach().unsqueeze(-1), gradient / scale), dim=-1)


@jaxtyped(typechecker=beartype)
def learned_control_signal(
    *,
    theta_hat: Float[Tensor, _BATCH_PARAMS_SHAPE],
    target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
    render: RenderFn,
    encoder: nn.Module,
) -> Float[Tensor, _BATCH_ANY_SHAPE]:
    """Encode the rendered estimate beside the observation, stopping at the simulator.

    The paper's equation 10. Because the render is detached, neither the simulator nor any
    cost need be differentiable, which is what lets a VST host serve as the simulator.

    :param theta_hat: One-step parameter estimate.
    :param target_audio: Observed audio the estimate is scored against.
    :param render: Simulator over decoded parameters; its graph is discarded.
    :param encoder: Trainable module mapping rendered audio to a control vector.
    :returns: Signal shaped ``(batch, control)``, differentiable w.r.t. ``encoder`` only.
    """
    with torch.no_grad():
        rendered = render(theta_hat.detach())
    return encoder(rendered - target_audio)


class ControlNet(nn.Module):
    """Zero-initialised residual correction to a pretrained velocity."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, field_dim: int, control_dim: int, hidden_dim: int) -> None:
        """Build the aggregation network over flow time, velocity, and control signal.

        The output layer starts at zero so a finetune begins as an exact identity on the pretrained
        velocity, independently of whether the field is frozen.

        :param field_dim: Width of the velocity this corrects.
        :param control_dim: Width of the control signal.
        :param hidden_dim: Hidden width of the correction.
        :raises ValueError: Any width is not positive.
        """
        super().__init__()
        for name, value in (
            ("field_dim", field_dim),
            ("control_dim", control_dim),
            ("hidden_dim", hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        output = nn.Linear(hidden_dim, field_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.body = nn.Sequential(
            nn.Linear(1 + field_dim + control_dim, hidden_dim), nn.GELU(), output
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        t: Float[Tensor, _BATCH_TIME_SHAPE],
        velocity: Float[Tensor, _BATCH_PARAMS_SHAPE],
        control: Float[Tensor, _BATCH_ANY_SHAPE],
    ) -> Float[Tensor, _BATCH_PARAMS_SHAPE]:
        """Return the correction to add to the pretrained velocity.

        :param t: Flow time shaped ``(batch, 1)``.
        :param velocity: Pretrained velocity shaped ``(batch, params)``.
        :param control: Control signal shaped ``(batch, control)``.
        :returns: Correction shaped ``(batch, params)``; zeros at initialisation.
        """
        return self.body(torch.cat((t, velocity, control.flatten(start_dim=1)), dim=-1))


class ControlledFlow(nn.Module):
    """Frozen flow whose velocity a control network corrects above a flow-time threshold."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        flow: nn.Module,
        control: nn.Module,
        t_min: float = DEFAULT_CONTROL_T_MIN,
    ) -> None:
        """Pair a frozen pretrained field with the control that refines it.

        :param flow: Pretrained velocity field; must carry no trainable parameters.
        :param control: Module mapping ``(t, velocity, control_signal)`` to a correction.
        :param t_min: Flow time above which the control engages, in ``[0, 1)``.
        :raises ValueError: ``flow`` is trainable, or ``t_min`` lies outside ``[0, 1)``.
        """
        super().__init__()
        trainable = [name for name, p in flow.named_parameters() if p.requires_grad]
        if trainable:
            raise ValueError(
                f"flow must be frozen; {len(trainable)} trainable parameter(s) {trainable} "
                "would let the finetune move the pretrained field it refines"
            )
        if not 0.0 <= t_min < 1.0:
            raise ValueError(f"t_min must lie in [0, 1), got {t_min}")
        self.flow = flow
        self.control = control
        self.t_min = t_min

    @jaxtyped(typechecker=beartype)
    def __getattr__(self, name: str) -> object:
        """Fall back to the wrapped field so this reads as the field it replaces.

        Callers hold a ``ControlledFlow`` wherever they held the pretrained field, and reach
        for that field's own surface (``apply_dropout``, ``d_model``, ``penalty``).

        :param name: Attribute absent from this wrapper.
        :returns: The wrapped field's attribute of that name.
        :raises AttributeError: Neither this wrapper nor the wrapped field defines it.
        """
        try:
            return super().__getattr__(name)  # pyright: ignore[reportReturnType]
        except AttributeError:
            # Read through __dict__-backed _modules; a plain self.flow would recurse here.
            flow = self._modules.get("flow")
            if flow is None:
                raise
            return getattr(flow, name)

    @jaxtyped(typechecker=beartype)
    def combine(
        self,
        velocity: Float[Tensor, _BATCH_PARAMS_SHAPE],
        t: Float[Tensor, _BATCH_TIME_SHAPE],
        control_input: Float[Tensor, _BATCH_ANY_SHAPE] | None,
    ) -> Float[Tensor, _BATCH_PARAMS_SHAPE]:
        """Correct an already-evaluated velocity, gated on flow time.

        Split out of :meth:`forward` so a caller that derives ``control_input`` from the
        velocity evaluates the field once, rather than once for the estimate and again to
        correct it — two evaluations a field carrying dropout would not even agree on.

        Rows below ``t_min`` are returned unchanged rather than corrected by zero, so a
        bypassed row is bit-identical to the pretrained field's own output.

        :param velocity: Pretrained velocity shaped ``(batch, params)``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param control_input: Control signal, or ``None`` for the capacity-matched ablation.
        :returns: Velocity shaped ``(batch, params)``.
        """
        if control_input is None:
            return velocity
        correction = self.control(t, velocity, control_input)
        return velocity + torch.where(t >= self.t_min, correction, torch.zeros_like(correction))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x_t: Float[Tensor, _BATCH_PARAMS_SHAPE],
        t: Float[Tensor, _BATCH_TIME_SHAPE],
        z: Float[Tensor, _BATCH_ANY_SHAPE] | None,
        control_input: Float[Tensor, _BATCH_ANY_SHAPE] | None = None,
    ) -> Float[Tensor, _BATCH_PARAMS_SHAPE]:
        """Return the pretrained velocity below ``t_min`` and the corrected one above.

        :param x_t: Trajectory point shaped ``(batch, params)``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param z: Conditioning the pretrained field consumes, or ``None`` for the
            classifier-free-guidance unconditional branch.
        :param control_input: Control signal, or ``None`` for the capacity-matched ablation.
        :returns: Velocity shaped ``(batch, params)``.
        """
        return self.combine(self.flow(x_t, t, z), t, control_input)
