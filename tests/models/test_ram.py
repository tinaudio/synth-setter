"""Behaviour tests for the Reinforce Adjoint Matching building blocks (arXiv 2605.10759)."""

import pytest
import torch

from synth_setter.models.components.ram import (
    group_relative_advantages,
    power_law_flow_time,
    ram_velocity_target,
)


def test_group_relative_advantages_centres_each_group_and_scales_by_pooled_std() -> None:
    """Groups are centred on their own mean and divided by the std pooled over every reward."""
    rewards = torch.tensor([1.0, 3.0, 4.0, 4.0])

    advantages = group_relative_advantages(rewards, group_size=2, eps=0.0)

    torch.testing.assert_close(
        advantages, torch.tensor([-0.8165, 0.8165, 0.0, 0.0]), atol=1e-4, rtol=0
    )


def test_group_relative_advantages_identical_rewards_yield_zero_without_nan() -> None:
    """A step whose rewards never differ produces zero advantage rather than 0/0."""
    advantages = group_relative_advantages(torch.full((4,), 2.5), group_size=2)

    torch.testing.assert_close(advantages, torch.zeros(4))


def test_group_relative_advantages_rejects_batch_not_divisible_by_group() -> None:
    """A batch that cannot be split into whole groups is a caller bug, not a silent truncation."""
    with pytest.raises(ValueError, match="group_size"):
        group_relative_advantages(torch.zeros(5), group_size=2)


def test_ram_velocity_target_with_zero_advantage_is_the_reference_velocity() -> None:
    """No reward signal leaves the pretrained field as the regression target."""
    reference = torch.tensor([[1.0, -2.0]])

    target = ram_velocity_target(
        reference=reference,
        old=torch.tensor([[5.0, 5.0]]),
        endpoint=torch.tensor([[0.5, 0.5]]),
        noise=torch.tensor([[0.0, 0.0]]),
        advantage=torch.tensor([0.0]),
    )

    torch.testing.assert_close(target, reference)


def test_ram_velocity_target_adds_advantage_scaled_reward_direction() -> None:
    """The target moves the reference toward ``(endpoint - noise)`` relative to the old field."""
    target = ram_velocity_target(
        reference=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        old=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        endpoint=torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
        noise=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        advantage=torch.tensor([2.0, -1.0]),
    )

    torch.testing.assert_close(target, torch.tensor([[5.0, -1.0], [-1.0, 2.0]]))


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        pytest.param(0.0, 0.75, id="alpha0-uniform"),
        pytest.param(1.0, 0.5, id="alpha1-linear-toward-noise"),
    ],
)
def test_power_law_flow_time_maps_quantile_toward_the_noise_end(
    alpha: float, expected: float
) -> None:
    """Density ``(1 - t) ** alpha`` sends the 0.25 quantile to ``1 - 0.25 ** (1 / (alpha + 1))``.

    :param alpha: Power-law exponent.
    :param expected: Flow time the quantile maps to.
    """
    t = power_law_flow_time(torch.tensor([[0.25]]), alpha=alpha)

    torch.testing.assert_close(t, torch.tensor([[expected]]))


def test_power_law_flow_time_rejects_negative_alpha() -> None:
    """A negative exponent has no normalisable density on ``[0, 1]``."""
    with pytest.raises(ValueError, match="alpha"):
        power_law_flow_time(torch.tensor([[0.5]]), alpha=-0.5)
