"""Tests for the TorchSynth feed-forward Lightning module."""

from __future__ import annotations

import functools
import types
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torchmetrics

from synth_setter.data.vst.torchsynth_param_spec import TORCHSYNTH_FULL_PARAM_SPEC
from synth_setter.models.components.residual_mlp import ResidualMLPBlock
from synth_setter.models.torchsynth_ff_module import TorchSynthBatch, TorchSynthFeedForwardModule

_NUM_PARAMS = 8
_SIGNAL_LENGTH = 64
_BATCH_SIZE = 3


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Make each test deterministic."""
    torch.manual_seed(0)


@pytest.fixture
def tiny_net() -> nn.Module:  # noqa: DOC201,DOC203
    """Build a small network mapping audio to parameters."""
    return ResidualMLPBlock(in_dim=_SIGNAL_LENGTH, hidden_dim=16, out_dim=_NUM_PARAMS)


@pytest.fixture
def optimizer_factory() -> Callable[..., torch.optim.Optimizer]:  # noqa: DOC201,DOC203
    """Return an optimizer factory matching Hydra's partial pattern."""
    return functools.partial(torch.optim.SGD, lr=1e-2)


@pytest.fixture
def scheduler_factory() -> Callable[..., torch.optim.lr_scheduler.LRScheduler]:  # noqa: DOC201,DOC203
    """Return a scheduler factory matching Hydra's partial pattern."""
    return functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1)


@pytest.fixture
def renderer() -> Callable[[torch.Tensor], torch.Tensor]:  # noqa: DOC201,DOC203
    """Return a deterministic renderer satisfying the spectral metric contract."""

    def render(params: torch.Tensor) -> torch.Tensor:
        return params.repeat_interleave(_SIGNAL_LENGTH // _NUM_PARAMS, dim=1)

    return render


@pytest.fixture
def batch(renderer: Callable[[torch.Tensor], torch.Tensor]) -> TorchSynthBatch:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Build one collated TorchSynth batch."""
    inputs = torch.randn(_BATCH_SIZE, _SIGNAL_LENGTH)
    targets = torch.rand(_BATCH_SIZE, _NUM_PARAMS)
    noise = torch.randn_like(targets)
    return inputs, targets, noise, renderer


def _make_module(
    *,
    net: nn.Module,
    optimizer: Callable[..., torch.optim.Optimizer] | None = None,
    scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    compile: bool = False,
) -> TorchSynthFeedForwardModule:
    """Build a TorchSynth module with unit-test defaults.

    :param net: Network under test.
    :param optimizer: Optional optimizer factory.
    :param scheduler: Optional scheduler factory.
    :param compile: Enable compile-on-fit behavior.
    :returns: Configured Lightning module.
    """
    return TorchSynthFeedForwardModule(
        net=net,
        optimizer=optimizer or functools.partial(torch.optim.SGD, lr=1e-2),
        scheduler=scheduler,
        compile=compile,
    )


def _patch_log(module: TorchSynthFeedForwardModule) -> MagicMock:
    """Replace Lightning logging for tests without a Trainer.

    :param module: Module under test.
    :returns: Installed logging mock.
    """
    log_mock = MagicMock()
    module.log = log_mock  # type: ignore[method-assign]
    return log_mock


def test_init_uses_mse_loss(tiny_net: nn.Module) -> None:
    """TorchSynth training uses named-parameter MSE.

    :param tiny_net: Network under test.
    """
    module = _make_module(net=tiny_net)
    assert isinstance(module.criterion, nn.MSELoss)


def test_forward_preserves_batch_shape_and_dtype(
    tiny_net: nn.Module,
    batch: TorchSynthBatch,
) -> None:
    """Forward maps audio rows to normalized parameter rows.

    :param tiny_net: Network under test.
    :param batch: Collated TorchSynth batch.
    """
    output = _make_module(net=tiny_net)(batch[0])
    assert output.shape == (_BATCH_SIZE, _NUM_PARAMS)
    assert output.dtype == batch[0].dtype


def test_model_step_returns_mse_and_batch_values(
    tiny_net: nn.Module,
    batch: TorchSynthBatch,
) -> None:
    """Model step computes MSE without reordering named controls.

    :param tiny_net: Network under test.
    :param batch: Collated TorchSynth batch.
    """
    module = _make_module(net=tiny_net)
    loss, predictions, targets, inputs = module.model_step(batch)
    torch.testing.assert_close(loss, torch.nn.functional.mse_loss(predictions, batch[1]))
    assert torch.equal(targets, batch[1])
    assert torch.equal(inputs, batch[0])


def test_training_step_backpropagates_to_every_parameter(
    tiny_net: nn.Module,
    batch: TorchSynthBatch,
) -> None:
    """Training loss reaches every network parameter.

    :param tiny_net: Network under test.
    :param batch: Collated TorchSynth batch.
    """
    module = _make_module(net=tiny_net)
    _patch_log(module)
    module.training_step(batch, 0).backward()
    assert all(parameter.grad is not None for parameter in module.net.parameters())


def test_validation_step_updates_lsd(
    tiny_net: nn.Module,
    batch: TorchSynthBatch,
) -> None:
    """Validation compares rendered predictions with input audio.

    :param tiny_net: Network under test.
    :param batch: Collated TorchSynth batch.
    """
    module = _make_module(net=tiny_net)
    metric = MagicMock(spec=torchmetrics.Metric)
    module.val_lsd = metric
    _patch_log(module)

    module.validation_step(batch, 0)

    predictions, inputs, renderer = metric.call_args.args
    assert predictions.shape == batch[1].shape
    assert torch.equal(inputs, batch[0])
    assert renderer is batch[-1]


def test_test_step_updates_lsd_and_logs_named_parameter_mse(
    tiny_net: nn.Module,
    batch: TorchSynthBatch,
) -> None:
    """Test metrics preserve TorchSynth control identities.

    :param tiny_net: Network under test.
    :param batch: Collated TorchSynth batch.
    """
    module = _make_module(net=tiny_net)
    metric = MagicMock(spec=torchmetrics.Metric)
    module.test_lsd = metric
    log_mock = _patch_log(module)

    module.test_step(batch, 0)

    assert metric.call_count == 1
    logged = {call.args[0]: call.args[1] for call in log_mock.call_args_list}
    predictions = module(batch[0])
    expected_mse = (predictions - batch[1]).square().mean()
    torch.testing.assert_close(logged["test/param_mse"], expected_mse)


def test_test_step_param_mse_includes_the_sampled_note_columns() -> None:
    """The reported parameter error scores the whole encoded row.

    Pitch and the note window are sampled per row, so they are real prediction targets that carry
    real error rather than constants that would deflate the mean.
    """
    spec = TORCHSYNTH_FULL_PARAM_SPEC
    module = _make_module(net=nn.Linear(_SIGNAL_LENGTH, spec.encoded_width))
    module.test_lsd = MagicMock(spec=torchmetrics.Metric)
    log_mock = _patch_log(module)
    inputs = torch.randn(_BATCH_SIZE, _SIGNAL_LENGTH)
    targets = torch.rand(_BATCH_SIZE, spec.encoded_width)
    note_shifted = targets.clone()
    note_shifted[:, spec.synth_param_length :] += 1.0

    module.test_step((inputs, targets, torch.randn_like(targets), lambda p: p), 0)
    module.test_step((inputs, note_shifted, torch.randn_like(targets), lambda p: p), 0)

    logged = [call.args[1] for call in log_mock.call_args_list if call.args[0] == "test/param_mse"]
    assert not torch.isclose(logged[0], logged[1])


def test_on_train_start_resets_validation_metric(tiny_net: nn.Module) -> None:
    """Sanity-validation state does not leak into fit metrics.

    :param tiny_net: Network under test.
    """
    module = _make_module(net=tiny_net)
    metric = MagicMock(spec=torchmetrics.Metric)
    module.val_lsd = metric
    module.on_train_start()
    metric.reset.assert_called_once_with()


@pytest.mark.parametrize(
    ("compile_flag", "stage", "expect_compiled"),
    [(False, "fit", False), (True, "validate", False), (True, "fit", True)],
)
def test_setup_compiles_only_for_enabled_fit(
    tiny_net: nn.Module,
    compile_flag: bool,
    stage: str,
    expect_compiled: bool,
) -> None:
    """Compilation is gated by both configuration and fit stage.

    :param tiny_net: Network under test.
    :param compile_flag: Whether compilation is enabled.
    :param stage: Lightning stage.
    :param expect_compiled: Expected compile state.
    """
    module = _make_module(net=tiny_net, compile=compile_flag)
    module.setup(stage)
    assert (module.net._compiled_call_impl is not None) is expect_compiled


def test_configure_optimizers_without_scheduler(
    tiny_net: nn.Module,
    optimizer_factory: Callable[..., torch.optim.Optimizer],
) -> None:
    """Optimizer-only configuration has one Lightning entry.

    :param tiny_net: Network under test.
    :param optimizer_factory: Optimizer factory.
    """
    module = _make_module(net=tiny_net, optimizer=optimizer_factory)
    module.trainer = types.SimpleNamespace(model=module)  # type: ignore[assignment]
    result = module.configure_optimizers()
    assert set(result) == {"optimizer"}
    assert isinstance(result["optimizer"], torch.optim.Optimizer)


def test_configure_optimizers_with_scheduler(
    tiny_net: nn.Module,
    optimizer_factory: Callable[..., torch.optim.Optimizer],
    scheduler_factory: Callable[..., torch.optim.lr_scheduler.LRScheduler],
) -> None:
    """Scheduler configuration monitors validation loss each epoch.

    :param tiny_net: Network under test.
    :param optimizer_factory: Optimizer factory.
    :param scheduler_factory: Scheduler factory.
    """
    module = _make_module(
        net=tiny_net,
        optimizer=optimizer_factory,
        scheduler=scheduler_factory,
    )
    module.trainer = types.SimpleNamespace(model=module)  # type: ignore[assignment]
    result = module.configure_optimizers()
    assert result["lr_scheduler"]["monitor"] == "val/loss"
    assert result["lr_scheduler"]["interval"] == "epoch"


@pytest.mark.slow
def test_fixed_batch_training_reduces_loss_by_two_orders_of_magnitude(
    renderer: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """A small network can fit a fixed batch.

    :param renderer: TorchSynth-compatible renderer.
    """
    net = ResidualMLPBlock(in_dim=_SIGNAL_LENGTH, hidden_dim=64, out_dim=_NUM_PARAMS)
    module = _make_module(net=net)
    _patch_log(module)
    inputs = torch.randn(_BATCH_SIZE, _SIGNAL_LENGTH)
    targets = torch.rand(_BATCH_SIZE, _NUM_PARAMS)
    batch = (inputs, targets, torch.randn_like(targets), renderer)
    optimizer = torch.optim.Adam(module.parameters(), lr=3e-4)

    with torch.no_grad():
        initial_loss = module.criterion(module(inputs), targets).item()

    final_loss = float("inf")
    for _ in range(300):
        optimizer.zero_grad()
        loss = module.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < 0.01 * initial_loss
