"""Tests for learning-rate scheduler checkpoint compatibility."""

import pytest
import torch

from synth_setter.utils.lr_scheduler import ResumeAwareCosineAnnealingLR


@pytest.fixture
def parameter() -> torch.nn.Parameter:
    """Create the trainable scalar shared by scheduler tests.

    :returns: A fresh scalar parameter.
    """
    return torch.nn.Parameter(torch.tensor(1.0))


def _step_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    steps: int,
) -> None:
    """Advance an optimizer before its scheduler for each requested step.

    :param optimizer: Optimizer to advance.
    :param scheduler: Scheduler paired with ``optimizer``.
    :param steps: Number of update steps.
    """
    for _ in range(steps):
        optimizer.step()
        scheduler.step()


def test_load_state_dict_matching_horizon_preserves_recovery_state(
    parameter: torch.nn.Parameter,
) -> None:
    """An unchanged horizon restores scheduler and optimizer state exactly.

    :param parameter: Parameter owned by the checkpointed optimizer.
    """
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    scheduler = ResumeAwareCosineAnnealingLR(optimizer, T_max=4, eta_min=0.01)
    parameter.grad = torch.tensor(2.0)
    _step_scheduler(optimizer, scheduler, steps=2)

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=0.1, momentum=0.9)
    resumed_scheduler = ResumeAwareCosineAnnealingLR(resumed_optimizer, T_max=4, eta_min=0.01)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed_scheduler.load_state_dict(scheduler.state_dict())

    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    assert resumed_optimizer.state_dict() == optimizer.state_dict()


def test_load_state_dict_extended_horizon_uses_configured_cosine_position(
    parameter: torch.nn.Parameter,
) -> None:
    """An extended horizon remaps the restored step onto the configured cosine.

    :param parameter: Parameter owned by the checkpointed optimizer.
    """
    old_optimizer = torch.optim.SGD([parameter], lr=0.1)
    old_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        old_optimizer, T_max=4, eta_min=0.01
    )
    _step_scheduler(old_optimizer, old_scheduler, steps=2)

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=0.1)
    resumed_scheduler = ResumeAwareCosineAnnealingLR(resumed_optimizer, T_max=8, eta_min=0.01)
    resumed_optimizer.load_state_dict(old_optimizer.state_dict())
    resumed_scheduler.load_state_dict(old_scheduler.state_dict())

    assert resumed_scheduler.T_max == 8
    assert resumed_scheduler.get_last_lr() == pytest.approx([0.08681980515339464])
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.08681980515339464)

    resumed_optimizer.step()
    resumed_scheduler.step()
    assert resumed_scheduler.get_last_lr() == pytest.approx([0.07222075445642905])


def test_load_state_dict_extended_horizon_under_warmup_uses_configured_cosine(
    parameter: torch.nn.Parameter,
) -> None:
    """Sequential warmup delegates horizon remapping to the cosine scheduler.

    :param parameter: Parameter owned by the checkpointed optimizer.
    """
    old_optimizer = torch.optim.SGD([parameter], lr=0.1)
    old_warmup = torch.optim.lr_scheduler.LinearLR(old_optimizer, 0.5, 1.0, 2)
    old_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(old_optimizer, T_max=4, eta_min=0.01)
    old_scheduler = torch.optim.lr_scheduler.SequentialLR(
        old_optimizer, schedulers=[old_warmup, old_cosine], milestones=[2]
    )
    _step_scheduler(old_optimizer, old_scheduler, steps=4)

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=0.1)
    resumed_warmup = torch.optim.lr_scheduler.LinearLR(resumed_optimizer, 0.5, 1.0, 2)
    resumed_cosine = ResumeAwareCosineAnnealingLR(resumed_optimizer, T_max=8, eta_min=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.SequentialLR(
        resumed_optimizer, schedulers=[resumed_warmup, resumed_cosine], milestones=[2]
    )
    resumed_optimizer.load_state_dict(old_optimizer.state_dict())
    resumed_scheduler.load_state_dict(old_scheduler.state_dict())

    assert resumed_cosine.T_max == 8
    assert resumed_cosine.get_last_lr() == pytest.approx([0.08681980515339464])
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.08681980515339464)


def test_load_state_dict_extended_inactive_cosine_preserves_warmup_lr(
    parameter: torch.nn.Parameter,
) -> None:
    """An inactive cosine child cannot overwrite the restored warmup position.

    :param parameter: Parameter owned by the checkpointed optimizer.
    """
    old_optimizer = torch.optim.SGD([parameter], lr=0.1)
    old_warmup = torch.optim.lr_scheduler.LinearLR(old_optimizer, 0.5, 1.0, 4)
    old_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(old_optimizer, T_max=8, eta_min=0.01)
    old_scheduler = torch.optim.lr_scheduler.SequentialLR(
        old_optimizer, schedulers=[old_warmup, old_cosine], milestones=[4]
    )
    _step_scheduler(old_optimizer, old_scheduler, steps=2)
    warmup_lr = old_optimizer.param_groups[0]["lr"]

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=0.1)
    resumed_warmup = torch.optim.lr_scheduler.LinearLR(resumed_optimizer, 0.5, 1.0, 4)
    resumed_cosine = ResumeAwareCosineAnnealingLR(resumed_optimizer, T_max=16, eta_min=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.SequentialLR(
        resumed_optimizer, schedulers=[resumed_warmup, resumed_cosine], milestones=[4]
    )
    resumed_optimizer.load_state_dict(old_optimizer.state_dict())
    resumed_scheduler.load_state_dict(old_scheduler.state_dict())

    assert resumed_cosine.T_max == 16
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(warmup_lr)


def test_load_state_dict_shorter_horizon_raises_clear_error(
    parameter: torch.nn.Parameter,
) -> None:
    """A reduced configured horizon fails instead of silently cycling cosine LR.

    :param parameter: Parameter owned by the checkpointed optimizer.
    """
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    old_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=0.01)
    resumed_scheduler = ResumeAwareCosineAnnealingLR(optimizer, T_max=4, eta_min=0.01)

    with pytest.raises(ValueError, match="configured T_max 4 is shorter than checkpoint T_max 8"):
        resumed_scheduler.load_state_dict(old_scheduler.state_dict())


def test_load_state_dict_non_integer_horizon_raises_clear_error(
    parameter: torch.nn.Parameter,
) -> None:
    """Malformed checkpoint horizon state fails before scheduler mutation.

    :param parameter: Parameter owned by the checkpointed optimizer.
    """
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = ResumeAwareCosineAnnealingLR(optimizer, T_max=8, eta_min=0.01)
    state: dict[str, object] = scheduler.state_dict()
    state["T_max"] = "eight"

    with pytest.raises(TypeError, match="checkpoint T_max must be an int"):
        scheduler.load_state_dict(state)
