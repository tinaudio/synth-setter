"""Reinforce Adjoint Matching primitives (arXiv 2605.10759, algorithm 1).

RAM post-trains a flow with a black-box reward while keeping the pretraining regression
structure: an on-policy endpoint is drawn with the ordinary sampler, scored, noised
analytically, and the velocity field regresses onto the pretrained field corrected by the
advantage-weighted reward direction. This repo integrates from noise at t=0 to data at
t=1, the reverse of the paper's time axis, so the reward direction is ``endpoint - noise``
and the noise-biased time law weights ``(1 - t)``.
"""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

_BATCH_SHAPE = "batch"
_BATCH_PARAMS_SHAPE = "batch params"
_BATCH_TIME_SHAPE = "batch 1"

# Guards the pooled-std divisor when every sampled endpoint scored identically.
_ADVANTAGE_EPS = 1e-4


@jaxtyped(typechecker=beartype)
def group_relative_advantages(
    rewards: Float[torch.Tensor, _BATCH_SHAPE],
    group_size: int,
    eps: float = _ADVANTAGE_EPS,
) -> Float[torch.Tensor, _BATCH_SHAPE]:
    """Centre rewards within each group and scale by the std pooled over the whole batch.

    Pooled rather than per-group scaling keeps the advantage scale stable across steps and
    avoids blowing up a group whose samples happened to score alike (paper section 6).

    :param rewards: One reward per sampled endpoint; consecutive ``group_size`` rows share a
        conditioning row.
    :param group_size: Endpoints sampled per conditioning row.
    :param eps: Added to the pooled std so identical rewards give zero, not NaN.
    :returns: Advantages shaped like ``rewards``.
    :raises ValueError: ``group_size`` is not positive, or the batch does not split into
        whole groups.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if rewards.shape[0] % group_size != 0:
        raise ValueError(
            f"{rewards.shape[0]} rewards do not split into groups of group_size={group_size}"
        )
    groups = rewards.reshape(-1, group_size)
    centred = groups - groups.mean(dim=1, keepdim=True)
    return (centred / (rewards.std(correction=0) + eps)).reshape(-1)


@jaxtyped(typechecker=beartype)
def ram_velocity_target(
    *,
    reference: Float[torch.Tensor, _BATCH_PARAMS_SHAPE],
    old: Float[torch.Tensor, _BATCH_PARAMS_SHAPE],
    endpoint: Float[torch.Tensor, _BATCH_PARAMS_SHAPE],
    noise: Float[torch.Tensor, _BATCH_PARAMS_SHAPE],
    advantage: Float[torch.Tensor, _BATCH_SHAPE],
) -> Float[torch.Tensor, _BATCH_PARAMS_SHAPE]:
    """Build the RAM regression target (paper equation 17) in this repo's time convention.

    :param reference: Pretrained velocity at the noised state.
    :param old: Lagged-policy velocity at the noised state; the field the endpoint was sampled
        from.
    :param endpoint: Sampled clean parameters the state was noised from.
    :param noise: Gaussian draw the state was noised with.
    :param advantage: Scaled advantage per row.
    :returns: Target velocity the policy regresses onto.
    """
    return reference + advantage.unsqueeze(-1) * ((endpoint - noise) - old)


@jaxtyped(typechecker=beartype)
def power_law_flow_time(
    quantile: Float[torch.Tensor, _BATCH_TIME_SHAPE], alpha: float
) -> Float[torch.Tensor, _BATCH_TIME_SHAPE]:
    """Map uniform quantiles to flow times with density proportional to ``(1 - t) ** alpha``.

    ``alpha=0`` is uniform; larger values bias toward the noise end, which the paper found
    shapes the trajectory more than the data end.

    :param quantile: Uniform draws in ``[0, 1]`` shaped ``(batch, 1)``.
    :param alpha: Non-negative power-law exponent.
    :returns: Flow times shaped ``(batch, 1)``.
    :raises ValueError: ``alpha`` is negative.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    return 1.0 - quantile ** (1.0 / (alpha + 1.0))
