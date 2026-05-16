"""Unit tests for :class:`synth_setter.models.ksin_ff_module.KSinFeedForwardModule`.

Each section maps to a slice of the module's contract:

* A — construction & ``loss_fn`` dispatch
* B — ``forward`` / ``model_step`` shape contract
* C — gradient flow through the network
* D — validation / test step wire-up to metrics
* E — ``configure_optimizers`` return shape
* F — ``setup("fit")`` compile gating
* G — single-batch overfit sanity (``@pytest.mark.slow``)

The module is exercised in isolation — no ``Trainer.fit``, no Hydra — so the
file deliberately bypasses ``tests/conftest.py``'s heavy fixtures.

ML-test items not covered here (rationale): MT2 output range, MT3 leakage,
MT4 loss-at-init expected value, MT8 input-independent baseline, MT10
directional expectations, MT11 metric thresholds, MT12 human baseline,
MT13 schema, MT19 preproc as pure functions. See the plan at
``/root/.claude/plans/temporal-beaming-treasure.md`` for the full rationale.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torchmetrics

from synth_setter.data.ksin_datamodule import make_sin
from synth_setter.models.components.loss import ChamferLoss, MSESortLoss
from synth_setter.models.components.residual_mlp import ResidualMLPBlock
from synth_setter.models.ksin_ff_module import KSinFeedForwardModule

_K = 4
_PARAMS_PER_TOKEN = 2
_NUM_PARAMS = _K * _PARAMS_PER_TOKEN
_SIGNAL_LENGTH = 64
_BATCH_SIZE = 3


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Seed PyTorch's global RNG so each test is deterministic (MT24)."""
    torch.manual_seed(0)


@pytest.fixture
def params_per_token() -> int:  # noqa: DOC201,DOC203
    """Return the ``params_per_token`` value the metrics and losses default to."""
    return _PARAMS_PER_TOKEN


@pytest.fixture
def num_params() -> int:  # noqa: DOC201,DOC203
    """Return the flat parameter-vector size predicted by the network."""
    return _NUM_PARAMS


@pytest.fixture
def batch_size() -> int:  # noqa: DOC201,DOC203
    """Return the small-but-greater-than-one batch size used by shape tests."""
    return _BATCH_SIZE


@pytest.fixture
def tiny_net() -> nn.Module:  # noqa: DOC201,DOC203
    """Build a small randomly-initialised network mapping audio to params (MT18)."""
    return ResidualMLPBlock(in_dim=_SIGNAL_LENGTH, hidden_dim=16, out_dim=_NUM_PARAMS)


@pytest.fixture
def opt_partial() -> Callable[..., torch.optim.Optimizer]:  # noqa: DOC201,DOC203
    """Return an optimizer ``functools.partial`` matching the Hydra-partial pattern."""
    return functools.partial(torch.optim.SGD, lr=1e-2)


@pytest.fixture
def sched_partial() -> Callable[..., torch.optim.lr_scheduler.LRScheduler]:  # noqa: DOC201,DOC203
    """Return a scheduler ``functools.partial`` matching the Hydra-partial pattern."""
    return functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1)


@pytest.fixture
def synth_fn() -> Callable[[torch.Tensor], torch.Tensor]:  # noqa: DOC201,DOC203
    """Return the real renderer used by the LSD metric, partial-bound to ``signal_length``."""
    return functools.partial(make_sin, length=_SIGNAL_LENGTH, break_symmetry=False)


@pytest.fixture
def batch(  # noqa: DOC101,DOC103,DOC201,DOC203
    synth_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Return a self-contained 3-tuple batch ``(x, y, synth_fn)`` (MT17)."""
    x = torch.randn(_BATCH_SIZE, _SIGNAL_LENGTH)
    y = torch.rand(_BATCH_SIZE, _NUM_PARAMS)
    return x, y, synth_fn


@pytest.fixture
def batch_with_noise(  # noqa: DOC101,DOC103,DOC201,DOC203
    synth_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Return a 4-tuple ``(x, y, noise, synth_fn)`` to pin the ``*_`` unpacking behavior."""
    x = torch.randn(_BATCH_SIZE, _SIGNAL_LENGTH)
    y = torch.rand(_BATCH_SIZE, _NUM_PARAMS)
    noise = torch.randn(_BATCH_SIZE, _SIGNAL_LENGTH)
    return x, y, noise, synth_fn


def _make_module(  # noqa: DOC101,DOC103,DOC201,DOC203
    *,
    net: nn.Module,
    loss_fn: str = "mse",
    optimizer: Callable[..., torch.optim.Optimizer] | None = None,
    scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    compile: bool = False,
    params_per_token: int = _PARAMS_PER_TOKEN,
) -> KSinFeedForwardModule:
    """Build a :class:`KSinFeedForwardModule` with sensible defaults for unit tests."""
    if optimizer is None:
        optimizer = functools.partial(torch.optim.SGD, lr=1e-2)
    return KSinFeedForwardModule(
        net=net,
        loss_fn=loss_fn,
        optimizer=optimizer,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        compile=compile,
        params_per_token=params_per_token,
    )


# --------------------------------------------------------------------------- #
# Section A — Construction & loss-fn dispatch (MT14)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("loss_fn", "expected_type"),
    [
        ("mse", nn.MSELoss),
        ("chamfer", ChamferLoss),
        ("mse_sort", MSESortLoss),
    ],
)
def test_init_dispatches_loss_fn(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    loss_fn: str,
    expected_type: type,
) -> None:
    """Each loss_fn string dispatches to the matching criterion class."""
    module = _make_module(net=tiny_net, loss_fn=loss_fn, optimizer=opt_partial)
    assert isinstance(module.criterion, expected_type), (
        f"loss_fn={loss_fn!r} produced {type(module.criterion).__name__}, "
        f"expected {expected_type.__name__}"
    )


def test_init_chamfer_loss_receives_params_per_token(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    params_per_token: int,
) -> None:
    """ChamferLoss is constructed with the module's params_per_token."""
    module = _make_module(
        net=tiny_net,
        loss_fn="chamfer",
        optimizer=opt_partial,
        params_per_token=params_per_token,
    )
    assert isinstance(module.criterion, ChamferLoss)
    assert module.criterion.params_per_token == params_per_token


def test_init_rejects_unknown_loss(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
) -> None:
    """An unknown loss_fn string raises NotImplementedError naming the bad value."""
    with pytest.raises(NotImplementedError, match="bogus"):
        _make_module(net=tiny_net, loss_fn="bogus", optimizer=opt_partial)


def test_save_hyperparameters_captures_init_args(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    params_per_token: int,
) -> None:
    """Scalar __init__ args round-trip through Lightning's save_hyperparameters."""
    module = _make_module(
        net=tiny_net,
        loss_fn="mse",
        optimizer=opt_partial,
        compile=False,
        params_per_token=params_per_token,
    )
    assert module.hparams["loss_fn"] == "mse"
    assert module.hparams["params_per_token"] == params_per_token
    assert module.hparams["compile"] is False


def test_metrics_instantiated_with_params_per_token(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    params_per_token: int,
) -> None:
    """ChamferDistance / LinearAssignmentDistance metrics carry the module's params_per_token."""
    module = _make_module(net=tiny_net, optimizer=opt_partial, params_per_token=params_per_token)
    assert module.val_chamfer.params_per_token == params_per_token
    assert module.test_chamfer.params_per_token == params_per_token
    assert module.test_lad.params_per_token == params_per_token


# --------------------------------------------------------------------------- #
# Section B — forward & model_step shape contract (MT1, MT25)
# --------------------------------------------------------------------------- #


def test_forward_shape(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    batch_size: int,
    num_params: int,
) -> None:
    """Forward maps (B, signal_length) audio to (B, num_params) and preserves dtype."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    x = batch[0]
    out = module(x)
    assert out.shape == (batch_size, num_params)
    assert out.dtype == x.dtype


def test_forward_is_passthrough_to_net(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
) -> None:
    """Module.forward is a pure passthrough — equal to net(x) element-wise."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    x = batch[0]
    assert torch.equal(module(x), module.net(x))


def test_model_step_return_tuple_ordering(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    batch_size: int,
    num_params: int,
) -> None:
    """model_step returns (loss, preds, y, x) with correct shapes and identity-preserves x."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    loss, preds, y_out, x_out = module.model_step(batch)  # type: ignore[arg-type]
    x_in, y_in, _ = batch
    assert loss.ndim == 0, f"loss should be scalar, got shape {tuple(loss.shape)}"
    assert preds.shape == (batch_size, num_params)
    assert y_out.shape == (batch_size, num_params)
    assert x_out is x_in, "model_step must pass x through by identity"
    assert y_out is y_in, "model_step must pass y through by identity"


def test_model_step_accepts_4_tuple_batch(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch_with_noise: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Callable[[torch.Tensor], torch.Tensor],
    ],
    batch_size: int,
    num_params: int,
) -> None:
    """The ``*_`` unpacking absorbs a middle ``noise`` element from a 4-tuple batch."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    loss, preds, y_out, x_out = module.model_step(batch_with_noise)  # type: ignore[arg-type]
    x_in, y_in, _noise, _synth_fn = batch_with_noise
    assert loss.ndim == 0
    assert preds.shape == (batch_size, num_params)
    assert y_out.shape == (batch_size, num_params)
    assert x_out is x_in
    assert y_out is y_in


# --------------------------------------------------------------------------- #
# Section C — Gradient flow (MT6, MT7)
# --------------------------------------------------------------------------- #


def test_all_parameters_receive_gradients(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
) -> None:
    """Every network parameter receives a non-zero gradient after one training_step + backward."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    loss = module.training_step(batch, 0)  # type: ignore[arg-type]
    loss.backward()
    missing: list[str] = []
    zero: list[str] = []
    for name, param in module.net.named_parameters():
        if param.grad is None:
            missing.append(name)
        elif param.grad.abs().sum().item() == 0.0:
            zero.append(name)
    assert not missing, f"net params with grad=None: {missing}"
    assert not zero, f"net params with all-zero grad: {zero}"


def test_backprop_per_example_independence(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    batch_size: int,
) -> None:
    """Backprop on row i affects only x[i]'s gradient, not x[j!=i] — the feed-forward invariant."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    x = batch[0].detach().clone().requires_grad_(True)
    preds = module(x)
    for i in range(batch_size):
        if x.grad is not None:
            x.grad.zero_()
        preds[i].sum().backward(retain_graph=True)
        assert x.grad is not None
        row_norms = x.grad.abs().sum(dim=-1)
        for j in range(batch_size):
            if j == i:
                assert row_norms[j].item() > 0.0, (
                    f"row {i} backward produced zero grad on its own row"
                )
            else:
                assert row_norms[j].item() == 0.0, (
                    f"row {i} backward leaked grad to row {j} (norm={row_norms[j].item()})"
                )


def test_training_step_returns_scalar_with_grad(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
) -> None:
    """training_step returns a scalar tensor wired into autograd."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    loss = module.training_step(batch, 0)  # type: ignore[arg-type]
    assert loss.ndim == 0
    assert loss.requires_grad
    assert loss.grad_fn is not None


# --------------------------------------------------------------------------- #
# Section D — Validation / test step wire-up (MT9, MT25)
# --------------------------------------------------------------------------- #


def _patch_metrics_and_log(  # noqa: DOC101,DOC103,DOC201,DOC203
    module: KSinFeedForwardModule,
) -> dict[str, MagicMock]:
    """Replace metric instances and ``log`` with mocks and return a dict for assertions."""
    mocks = {
        "val_lsd": MagicMock(spec=torchmetrics.Metric),
        "val_chamfer": MagicMock(spec=torchmetrics.Metric),
        "test_lsd": MagicMock(spec=torchmetrics.Metric),
        "test_chamfer": MagicMock(spec=torchmetrics.Metric),
        "test_lad": MagicMock(spec=torchmetrics.Metric),
        "log": MagicMock(),
    }
    module.val_lsd = mocks["val_lsd"]
    module.val_chamfer = mocks["val_chamfer"]
    module.test_lsd = mocks["test_lsd"]
    module.test_chamfer = mocks["test_chamfer"]
    module.test_lad = mocks["test_lad"]
    module.log = mocks["log"]  # type: ignore[method-assign]
    return mocks


def test_validation_step_calls_lsd_with_preds_inputs_synthfn(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
) -> None:
    """val_lsd receives (preds, inputs, synth_fn) — the unusual 3-arg metric signature."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    mocks = _patch_metrics_and_log(module)
    x, y, synth_fn = batch
    module.validation_step(batch, 0)  # type: ignore[arg-type]
    assert mocks["val_lsd"].call_count == 1
    args, kwargs = mocks["val_lsd"].call_args
    assert kwargs == {}
    preds_arg, inputs_arg, synth_arg = args
    assert torch.equal(inputs_arg, x)
    assert synth_arg is synth_fn
    assert preds_arg.shape == y.shape


def test_validation_step_calls_chamfer_with_preds_targets(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
) -> None:
    """val_chamfer receives (preds, targets) — the standard 2-arg metric signature."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    mocks = _patch_metrics_and_log(module)
    _x, y, _synth_fn = batch
    module.validation_step(batch, 0)  # type: ignore[arg-type]
    assert mocks["val_chamfer"].call_count == 1
    args, kwargs = mocks["val_chamfer"].call_args
    assert kwargs == {}
    preds_arg, target_arg = args
    assert torch.equal(target_arg, y)
    assert preds_arg.shape == y.shape


def test_test_step_calls_lsd_chamfer_lad_and_logs_param_mse(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
) -> None:
    """test_step calls all three test metrics and logs ``test/param_mse``."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    mocks = _patch_metrics_and_log(module)
    x, y, synth_fn = batch
    module.test_step(batch, 0)  # type: ignore[arg-type]
    assert mocks["test_lsd"].call_count == 1
    args, _ = mocks["test_lsd"].call_args
    preds_arg, inputs_arg, synth_arg = args
    assert torch.equal(inputs_arg, x)
    assert synth_arg is synth_fn
    assert preds_arg.shape == y.shape
    assert mocks["test_chamfer"].call_count == 1
    chamfer_preds, chamfer_target = mocks["test_chamfer"].call_args.args
    assert torch.equal(chamfer_target, y)
    assert chamfer_preds.shape == y.shape
    assert mocks["test_lad"].call_count == 1
    lad_preds, lad_target = mocks["test_lad"].call_args.args
    assert torch.equal(lad_target, y)
    assert lad_preds.shape == y.shape

    log_calls = mocks["log"].call_args_list
    logged_names = [call.args[0] for call in log_calls]
    assert "test/param_mse" in logged_names
    param_mse_call = next(c for c in log_calls if c.args[0] == "test/param_mse")
    expected_param_mse = (preds_arg - y).square().mean()
    assert torch.allclose(param_mse_call.args[1], expected_param_mse)


def test_on_train_start_resets_val_metrics(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
) -> None:
    """on_train_start resets val_lsd and val_chamfer so sanity-check residuals don't leak."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    val_lsd_mock = MagicMock(spec=torchmetrics.Metric)
    val_chamfer_mock = MagicMock(spec=torchmetrics.Metric)
    module.val_lsd = val_lsd_mock
    module.val_chamfer = val_chamfer_mock
    module.on_train_start()
    assert val_lsd_mock.reset.call_count == 1
    assert val_chamfer_mock.reset.call_count == 1


def test_predictions_invariant_to_batch_row_permutation(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    batch_size: int,
) -> None:
    """Permuting batch rows permutes predictions identically (MT9 row-independence invariance)."""
    module = _make_module(net=tiny_net, optimizer=opt_partial)
    module.eval()
    x = batch[0]
    perm = torch.tensor([2, 0, 1])
    assert perm.shape == (batch_size,)
    with torch.no_grad():
        baseline = module(x)
        permuted = module(x[perm])
    assert torch.allclose(permuted, baseline[perm], atol=1e-6)


# --------------------------------------------------------------------------- #
# Section E — configure_optimizers (MT25)
# --------------------------------------------------------------------------- #


def test_configure_optimizers_no_scheduler(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
) -> None:
    """With scheduler=None, configure_optimizers returns a single-key dict with the optimizer."""
    module = _make_module(net=tiny_net, optimizer=opt_partial, scheduler=None)
    module.trainer = types.SimpleNamespace(model=module)  # type: ignore[assignment]
    result = module.configure_optimizers()
    assert set(result) == {"optimizer"}
    assert isinstance(result["optimizer"], torch.optim.Optimizer)


def test_configure_optimizers_with_scheduler(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    sched_partial: Callable[..., torch.optim.lr_scheduler.LRScheduler],
) -> None:
    """With a scheduler, configure_optimizers returns the optimizer plus an lr_scheduler block."""
    module = _make_module(net=tiny_net, optimizer=opt_partial, scheduler=sched_partial)
    module.trainer = types.SimpleNamespace(model=module)  # type: ignore[assignment]
    result = module.configure_optimizers()
    assert set(result) == {"optimizer", "lr_scheduler"}
    lr_block = result["lr_scheduler"]
    assert lr_block["monitor"] == "val/loss"
    assert lr_block["interval"] == "epoch"
    assert lr_block["frequency"] == 1


# --------------------------------------------------------------------------- #
# Section F — setup("fit") compile gating
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("compile_flag", "stage", "expect_called"),
    [
        (False, "fit", False),
        (True, "validate", False),
        (True, "fit", True),
    ],
)
def test_setup_compiles_only_when_compile_true_and_stage_fit(  # noqa: DOC101,DOC103
    tiny_net: nn.Module,
    opt_partial: Callable[..., torch.optim.Optimizer],
    monkeypatch: pytest.MonkeyPatch,
    compile_flag: bool,
    stage: str,
    expect_called: bool,
) -> None:
    """torch.compile is invoked exactly when compile=True AND stage=='fit'."""
    module = _make_module(net=tiny_net, optimizer=opt_partial, compile=compile_flag)
    original_net = module.net
    compile_mock = MagicMock(return_value=original_net)
    monkeypatch.setattr(torch, "compile", compile_mock)
    module.setup(stage)
    if expect_called:
        compile_mock.assert_called_once_with(original_net)
    else:
        compile_mock.assert_not_called()


# --------------------------------------------------------------------------- #
# Section G — Overfit single batch (MT5, slow)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_overfit_single_batch_with_mse_loss(  # noqa: DOC101,DOC103
    synth_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """A slightly larger net can drive MSE loss on a fixed batch to near-zero (MT5)."""
    torch.manual_seed(0)
    net = ResidualMLPBlock(in_dim=_SIGNAL_LENGTH, hidden_dim=64, out_dim=_NUM_PARAMS)
    optimizer = functools.partial(torch.optim.Adam, lr=3e-4)
    module = _make_module(net=net, loss_fn="mse", optimizer=optimizer)
    x = torch.randn(_BATCH_SIZE, _SIGNAL_LENGTH)
    y = torch.rand(_BATCH_SIZE, _NUM_PARAMS)
    batch = (x, y, synth_fn)
    opt = torch.optim.Adam(module.parameters(), lr=3e-4)

    with torch.no_grad():
        initial_loss = module.criterion(module(x), y).item()

    final_loss = float("inf")
    for _ in range(300):
        opt.zero_grad()
        loss = module.training_step(batch, 0)  # type: ignore[arg-type]
        loss.backward()
        opt.step()
        final_loss = loss.item()

    assert final_loss < 0.01, (
        f"failed to overfit single batch: final_loss={final_loss:.4f}, "
        f"initial_loss={initial_loss:.4f}"
    )
    assert final_loss < 0.1 * initial_loss, (
        f"loss did not decrease by 10x: final_loss={final_loss:.4f}, "
        f"initial_loss={initial_loss:.4f}"
    )
