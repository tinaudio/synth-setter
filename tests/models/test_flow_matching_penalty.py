"""Regression tests for the optional penalty path in flow-matching ``training_step`` (#1689)."""

from __future__ import annotations

import functools
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from synth_setter.models.components.vector_field import VectorField
from synth_setter.models.ksin_flow_matching_module import KSinFlowMatchingModule
from synth_setter.models.surge_flow_matching_module import VSTFlowMatchingModule

_SENTINEL_BATCH = object()


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Seed PyTorch's global RNG so each test is deterministic."""
    torch.manual_seed(0)


def _opt_partial() -> Callable[..., torch.optim.Optimizer]:  # noqa: DOC201
    """Return an SGD optimizer partial matching the Hydra ``_partial_`` pattern."""
    return functools.partial(torch.optim.SGD, lr=1e-2)


def _sched_partial() -> Callable[..., torch.optim.lr_scheduler.LRScheduler]:  # noqa: DOC201
    """Return a StepLR scheduler partial matching the Hydra ``_partial_`` pattern."""
    return functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1)


def _make_surge() -> VSTFlowMatchingModule:  # noqa: DOC201
    """Build a stand-in-net module (valid only with ``_train_step`` patched)."""
    return VSTFlowMatchingModule(
        encoder=nn.Identity(),
        vector_field=nn.Identity(),
        optimizer=_opt_partial(),  # type: ignore[arg-type]  # Hydra _partial_ factory
        scheduler=_sched_partial(),  # type: ignore[arg-type]  # Hydra _partial_ factory
    )


def _make_ksin() -> KSinFlowMatchingModule:  # noqa: DOC201
    """Build a stand-in-net module (valid only with ``_train_step`` patched)."""
    return KSinFlowMatchingModule(
        encoder=nn.Identity(),
        vector_field=nn.Identity(),
        optimizer=_opt_partial(),  # type: ignore[arg-type]  # Hydra _partial_ factory
        scheduler=_sched_partial(),  # type: ignore[arg-type]  # Hydra _partial_ factory
    )


@pytest.mark.parametrize("make_module", [_make_surge, _make_ksin], ids=["surge", "ksin"])
def test_training_step_returns_loss_when_field_lacks_penalty(  # noqa: DOC101,DOC103
    make_module: Callable[[], VSTFlowMatchingModule | KSinFlowMatchingModule],
) -> None:
    """``training_step`` returns the bare loss when ``_train_step`` yields no penalty."""
    module = make_module()
    module.log = MagicMock()  # type: ignore[method-assign]
    loss = torch.tensor(2.0, requires_grad=True)
    module._train_step = lambda _batch: (loss, None)  # type: ignore[assignment,method-assign]

    out = module.training_step(_SENTINEL_BATCH, 0)  # type: ignore[arg-type]

    torch.testing.assert_close(out, loss)
    assert out.requires_grad
    logged = [call.args[0] for call in module.log.call_args_list]
    assert "train/penalty" not in logged


@pytest.mark.parametrize("make_module", [_make_surge, _make_ksin], ids=["surge", "ksin"])
def test_training_step_adds_penalty_when_field_provides_it(  # noqa: DOC101,DOC103
    make_module: Callable[[], VSTFlowMatchingModule | KSinFlowMatchingModule],
) -> None:
    """``training_step`` adds the field penalty to the loss when one is reported."""
    module = make_module()
    module.log = MagicMock()  # type: ignore[method-assign]
    loss = torch.tensor(2.0, requires_grad=True)
    penalty = torch.tensor(0.5, requires_grad=True)
    module._train_step = lambda _batch: (loss, penalty)  # type: ignore[assignment,method-assign]

    out = module.training_step(_SENTINEL_BATCH, 0)  # type: ignore[arg-type]

    torch.testing.assert_close(out, torch.tensor(2.5))
    logged = [call.args[0] for call in module.log.call_args_list]
    assert "train/penalty" in logged
    # Pin that the penalty itself stays in the graph, not just the loss.
    out.backward()
    assert penalty.grad is not None


def test_surge_training_step_real_penaltyless_field_returns_finite_scalar() -> None:
    """A real ``VectorField`` (no ``penalty``) drives ``training_step`` to a finite scalar loss."""
    num_params, conditioning_dim = 8, 4
    field = VectorField(
        field_dim=num_params, hidden_dim=16, conditioning_dim=conditioning_dim, num_blocks=2
    )
    module = VSTFlowMatchingModule(
        encoder=nn.Identity(),
        vector_field=field,
        optimizer=_opt_partial(),  # type: ignore[arg-type]  # Hydra _partial_ factory
        scheduler=_sched_partial(),  # type: ignore[arg-type]  # Hydra _partial_ factory
        cfg_dropout_rate=0.0,
        num_params=num_params,
    )
    module.log = MagicMock()  # type: ignore[method-assign]
    batch: dict[str, torch.Tensor | None] = {
        "mel_spec": torch.randn(3, conditioning_dim),
        "params": torch.rand(3, num_params),
        "noise": torch.randn(3, num_params),
    }

    _, penalty = module._train_step(batch)
    assert penalty is None, "VectorField has no penalty(); this pins the reachable None branch"

    out = module.training_step(batch, 0)
    assert out.ndim == 0 and out.requires_grad and torch.isfinite(out)
