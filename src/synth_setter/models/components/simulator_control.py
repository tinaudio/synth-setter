"""Simulator-feedback control signals and the controlled flow that consumes them.

Implements the finetuning scheme of Holzschuh & Thuerey, "Flow Matching for Posterior
Inference with Simulator Feedback" (https://arxiv.org/abs/2410.22573): a frozen pretrained
flow supplies a velocity, a simulator scores the one-step estimate that velocity implies,
and a small control network aggregates the two. The control signal is an *input*, never a
loss term, which is what leaves the plain flow-matching objective and its guarantees intact.

Typical usage:
    control = ControlledFlow(
        flow=pretrained_field.requires_grad_(False),
        control=ControlNet(field_dim=num_params, control_dim=1 + num_params, hidden_dim=256),
    )
    signal = gradient_control_signal(
        theta_hat=estimate, target_audio=observed, render=render_fn, cost=distance
    )
    velocity = control(x_t, t, z, control_input=signal)
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
# Flow time below which the paper finds the one-step estimate too poor to control on
# (appendix B). Experiments set this through their own config; this is the paper's value.
DEFAULT_CONTROL_T_MIN: Final = 0.8

_BATCH_PARAMS_SHAPE = "batch params"
_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_TIME_SHAPE = "batch 1"
_BATCH_SHAPE = "batch"
_BATCH_ANY_SHAPE = "batch ..."

type RenderFn = Callable[[Float[Tensor, _BATCH_PARAMS_SHAPE]], Float[Tensor, _BATCH_AUDIO_SHAPE]]


@jaxtyped(typechecker=beartype)
def _match_target_clamping(
    rendered: Float[Tensor, _BATCH_AUDIO_SHAPE],
) -> Float[Tensor, _BATCH_AUDIO_SHAPE]:
    """Clamp a render into the range stored targets were written in, keeping its gradient.

    ``render_torchsynth`` hard-clamps what it stores, so an unclamped estimate scored against
    a clamped target lets clipping the target can never exhibit dominate the residual on
    exactly the loudest rows. Straight-through, so a clipped row still receives gradient.

    :param rendered: Audio straight from the simulator.
    :returns: Audio clamped to ``[-1, 1]`` in the forward pass only.
    """
    return rendered + (rendered.clamp(-1.0, 1.0) - rendered).detach()


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

    Runs under an explicit ``enable_grad`` so it also works inside the ``no_grad`` an ODE
    integration loop holds: ``requires_grad`` survives that context while the graph the
    gradient needs does not.

    :param theta_hat: One-step parameter estimate carrying ``requires_grad``.
    :param target_audio: Observed audio the estimate is scored against.
    :param render: Differentiable simulator over decoded parameters.
    :param cost: Module mapping ``(rendered, target)`` to a per-sample cost.
    :returns: Signal shaped ``(batch, 1 + params)``, detached and finite.
    :raises ValueError: ``theta_hat`` cannot carry a gradient, so no signal exists.
    """
    if not theta_hat.requires_grad:
        raise ValueError("theta_hat must require grad for a gradient control signal")
    with torch.enable_grad():
        per_sample = cost(_match_target_clamping(render(theta_hat)), target_audio.detach())
        # create_graph=False: the signal is an input, and a graph here would silently make the
        # finetune second-order.
        (gradient,) = torch.autograd.grad(per_sample.sum(), theta_hat, create_graph=False)
    # Both blocks are sanitized, not just the gradient: the contract promises a finite
    # signal, and one NaN entry propagates through the control network to every row.
    gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    scale = gradient.norm(dim=-1, keepdim=True).clamp_min(_GRAD_NORM_EPS)
    return torch.cat((_compress_cost(per_sample.detach()).unsqueeze(-1), gradient / scale), dim=-1)


@jaxtyped(typechecker=beartype)
def _compress_cost(
    per_sample: Float[Tensor, _BATCH_SHAPE],
) -> Float[Tensor, _BATCH_SHAPE]:
    """Compress the cost onto the scale of the unit-norm gradient block beside it.

    A multi-scale spectral distance returns a mean absolute dB gap of ~10-40, while each of
    ~300 unit-norm gradient entries is ~0.06. Concatenated raw, the cost outweighs the entire
    gradient block by two to three orders of magnitude — the same imbalance the gradient
    normalization exists to prevent, just inverted. ``log1p`` is monotone and fixes 0 at 0,
    so a larger entry still reads as a worse cost; the clamp keeps it in domain for a
    distance that returns a small negative from floating-point error.

    :param per_sample: Detached per-row cost.
    :returns: Compressed per-row cost, finite.
    """
    return torch.nan_to_num(
        torch.log1p(per_sample.clamp_min(0.0)), nan=0.0, posinf=0.0, neginf=0.0
    )


@jaxtyped(typechecker=beartype)
def learned_control_signal(
    *,
    theta_hat: Float[Tensor, _BATCH_PARAMS_SHAPE],
    target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
    render: RenderFn,
    encoder: nn.Module,
) -> Float[Tensor, _BATCH_ANY_SHAPE]:
    """Encode the residual between the rendered estimate and the observation.

    The paper's equation 10. Because the render is detached, neither the simulator nor any
    cost need be differentiable, which is what lets a VST host serve as the simulator. The
    target is detached too, so ``encoder`` is the only path gradient takes.

    :param theta_hat: One-step parameter estimate.
    :param target_audio: Observed audio the estimate is scored against.
    :param render: Simulator over decoded parameters; its graph is discarded.
    :param encoder: Trainable module mapping the audio residual to a control vector.
    :returns: Signal shaped ``(batch, control)``, differentiable w.r.t. ``encoder`` only.
    """
    with torch.no_grad():
        rendered = _match_target_clamping(render(theta_hat.detach()))
    return encoder(rendered - target_audio.detach())


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
        self.control_dim = control_dim
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
        :param control: Control signal shaped ``(batch, control)``; ``reshape`` also accepts
            the rank-1 per-row scalar the annotation permits and ``flatten`` rejects.
        :returns: Correction shaped ``(batch, params)``; zeros at initialisation.
        :raises ValueError: The signal's width is not the one this was built for, which
            would otherwise surface as a bare matmul shape error.
        """
        flat = control.reshape(control.shape[0], -1)
        if flat.shape[-1] != self.control_dim:
            raise ValueError(
                f"control signal has width {flat.shape[-1]}, but this ControlNet was built "
                f"for {self.control_dim}"
            )
        return self.body(torch.cat((t, velocity, flat), dim=-1))


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
        self._freeze_flow()

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> ControlledFlow:
        """Enter the given mode, re-asserting the freeze the constructor checked.

        Registering the field as a submodule puts it in reach of a later
        ``requires_grad_(True)`` or an optimizer built over ``self.parameters()``, either of
        which silently undoes the invariant the constructor raises on. Eval mode matters
        separately: clearing ``requires_grad`` does not stop normalization running
        statistics, so a field left in train mode drifts with every weight still frozen.

        :param mode: Training mode requested for the trainable control.
        :returns: This module.
        """
        super().train(mode)
        self._freeze_flow()
        return self

    @jaxtyped(typechecker=beartype)
    def _freeze_flow(self) -> None:
        """Hold the pretrained field frozen and in eval mode."""
        self.flow.requires_grad_(False)
        self.flow.eval()

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x_t: Float[Tensor, _BATCH_PARAMS_SHAPE],
        t: Float[Tensor, _BATCH_TIME_SHAPE],
        z: Float[Tensor, _BATCH_ANY_SHAPE] | None,
        *,
        control_input: Float[Tensor, _BATCH_ANY_SHAPE] | None = None,
    ) -> Float[Tensor, _BATCH_PARAMS_SHAPE]:
        """Return the pretrained velocity below ``t_min`` and the corrected one above.

        A bypassed row selects the pretrained velocity itself rather than adding a zero
        correction, so it is bit-identical to the field's own output — adding ``+0.0`` to a
        ``-0.0`` velocity flips its sign bit. Its control signal is zeroed before the network
        rather than after, because a non-finite entry on a bypassed row would otherwise give
        a clean forward while the GELU backward turned every control gradient in the batch
        into NaN.

        :param x_t: Trajectory point shaped ``(batch, params)``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param z: Conditioning the pretrained field consumes, or ``None`` for the
            classifier-free-guidance unconditional branch.
        :param control_input: Control signal, keyword-only so it cannot be mistaken for
            ``z`` at a call site. ``None`` bypasses the control entirely, which trains
            nothing and so is *not* the capacity-matched ablation; that arm feeds a zeroed
            signal through this same network.
        :returns: Velocity shaped ``(batch, params)``.
        """
        velocity = self.flow(x_t, t, z)
        if control_input is None:
            return velocity
        engaged = t >= self.t_min
        flat = control_input.reshape(control_input.shape[0], -1)
        correction = self.control(t, velocity, torch.where(engaged, flat, torch.zeros_like(flat)))
        return torch.where(engaged, velocity + correction, velocity)
