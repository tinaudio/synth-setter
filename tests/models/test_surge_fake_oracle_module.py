"""Behavioral tests for :mod:`synth_setter.models.surge_fake_oracle_module`.

The fake-oracle module is a drop-in replacement for
:class:`synth_setter.models.surge_ff_module.VSTFeedForwardModule` that returns
``batch["params"]`` as its prediction. The tests pin the oracle contract
(perfect inversion, zero loss, grad-capable) and the four Lightning step
shapes that downstream callbacks depend on (``PredictionWriter`` unpacks
``predict_step``'s tuple; ``LogPerParamMSE`` reads ``per_param_mse`` from the
val/test step dicts).

Shapes are deliberately tiny — the oracle ignores both ``mel_spec`` shape and
the AST-style production sizes — so the tests run in milliseconds.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from synth_setter.models.surge_fake_oracle_module import (
    FakeOracleNet,
    VSTFakeOracleModule,
)

_NUM_PARAMS = 8
_MEL_CHANNELS = 2
_MEL_N_MELS = 4
_MEL_N_FRAMES = 5


def _make_batch(batch_size: int, *, mel_seed: int = 0) -> dict[str, torch.Tensor]:
    """Build a synthetic batch with deterministic ``params`` and seedable ``mel_spec``.

    :param batch_size: First-axis length for every tensor in the batch.
    :param mel_seed: Seed for the mel-spec RNG so callers can request batches that
        share ``params`` but differ in ``mel_spec`` (used to verify mel-independence
        of the oracle's predictions).

    :return: Batch dict with the keys the Lightning module's step functions consume.
    :rtype: dict[str, torch.Tensor]
    """
    params = torch.arange(batch_size * _NUM_PARAMS, dtype=torch.float32).reshape(
        batch_size, _NUM_PARAMS
    )
    generator = torch.Generator().manual_seed(mel_seed)
    mel_spec = torch.randn(
        batch_size,
        _MEL_CHANNELS,
        _MEL_N_MELS,
        _MEL_N_FRAMES,
        generator=generator,
    )
    audio = torch.zeros(batch_size, _MEL_CHANNELS, 16)
    return {"params": params, "mel_spec": mel_spec, "audio": audio}


def _make_module() -> VSTFakeOracleModule:
    """Build a fresh :class:`VSTFakeOracleModule` with a partial Adam optimizer.

    :return: Module ready for direct step-method calls (no Trainer attached).
    :rtype: VSTFakeOracleModule
    """
    net = FakeOracleNet(d_out=_NUM_PARAMS)
    optimizer = partial(torch.optim.Adam, lr=1e-4)
    return VSTFakeOracleModule(net=net, optimizer=optimizer, scheduler=None)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_predict_step_returns_params_and_batch_tuple(batch_size: int) -> None:
    """``predict_step`` returns ``(batch["params"], batch)`` so PredictionWriter can unpack.

    :param batch_size: Parametrized batch dimension exercised by this test.
    """
    module = _make_module()
    batch = _make_batch(batch_size)

    preds, returned_batch = module.predict_step(batch, batch_idx=0)

    assert torch.equal(preds, batch["params"])
    assert returned_batch is batch


@pytest.mark.parametrize("batch_size", [1, 4])
def test_predict_step_ignores_mel_spec(batch_size: int) -> None:
    """Different mel inputs with identical params produce identical predictions.

    :param batch_size: Parametrized batch dimension exercised by this test.
    """
    module = _make_module()
    batch_a = _make_batch(batch_size, mel_seed=0)
    batch_b = _make_batch(batch_size, mel_seed=1)

    preds_a, _ = module.predict_step(batch_a, batch_idx=0)
    preds_b, _ = module.predict_step(batch_b, batch_idx=0)

    assert not torch.equal(batch_a["mel_spec"], batch_b["mel_spec"])
    assert torch.equal(preds_a, preds_b)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_model_step_returns_four_tuple_with_oracle_preds(batch_size: int) -> None:
    """``model_step`` returns ``(loss, preds, targets, mel_spec)`` with preds == targets == params.

    :param batch_size: Parametrized batch dimension exercised by this test.
    """
    module = _make_module()
    batch = _make_batch(batch_size)

    loss, preds, targets, mel_spec = module.model_step(batch)

    assert loss.shape == ()
    assert preds.shape == (batch_size, _NUM_PARAMS)
    assert targets.shape == (batch_size, _NUM_PARAMS)
    assert mel_spec.shape == (batch_size, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES)
    assert torch.equal(preds, batch["params"])
    assert torch.equal(targets, batch["params"])


@pytest.mark.parametrize("batch_size", [1, 4])
def test_model_step_loss_is_exactly_zero(batch_size: int) -> None:
    """Oracle loss is zero — no parameter regression error on a perfect prediction.

    :param batch_size: Parametrized batch dimension exercised by this test.
    """
    module = _make_module()
    batch = _make_batch(batch_size)

    loss, *_ = module.model_step(batch)

    assert loss.item() == 0.0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_model_step_loss_requires_grad(batch_size: int) -> None:
    """Loss must carry grad so Lightning's ``loss.backward()`` succeeds with a real optimizer.

    :param batch_size: Parametrized batch dimension exercised by this test.
    """
    module = _make_module()
    batch = _make_batch(batch_size)

    loss, *_ = module.model_step(batch)

    assert loss.requires_grad


@pytest.mark.parametrize("batch_size", [1, 4])
def test_training_step_returns_zero_loss_with_grad(batch_size: int) -> None:
    """``training_step`` returns a zero, grad-bearing loss that can backpropagate.

    :param batch_size: Parametrized batch dimension exercised by this test.
    """
    module = _make_module()
    batch = _make_batch(batch_size)

    loss = module.training_step(batch, batch_idx=0)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    # The dummy parameter is the *only* reason loss carries a grad path — if a
    # refactor drops the `self.net(mel_spec)` call from model_step, loss.backward()
    # would still pass on a detached zero tensor, but no grad would land on dummy.
    assert module.net.dummy.grad is not None


@pytest.mark.parametrize("batch_size", [1, 4])
def test_validation_step_returns_zero_per_param_mse(batch_size: int) -> None:
    """``validation_step`` returns the dict shape ``LogPerParamMSE`` reads, with zero MSE.

    :param batch_size: Parametrized batch dimension exercised by this test.
    """
    module = _make_module()
    batch = _make_batch(batch_size)

    outputs = module.validation_step(batch, batch_idx=0)

    assert set(outputs.keys()) == {"param_mse", "per_param_mse"}
    assert outputs["param_mse"].shape == ()
    assert outputs["per_param_mse"].shape == (_NUM_PARAMS,)
    assert outputs["param_mse"].item() == 0.0
    assert torch.equal(outputs["per_param_mse"], torch.zeros(_NUM_PARAMS))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_test_step_returns_zero_per_param_mse(batch_size: int) -> None:
    """``test_step`` mirrors ``validation_step``'s contract — same dict, zero values.

    :param batch_size: Parametrized batch dimension exercised by this test.
    """
    module = _make_module()
    batch = _make_batch(batch_size)

    outputs = module.test_step(batch, batch_idx=0)

    assert set(outputs.keys()) == {"param_mse", "per_param_mse"}
    assert outputs["param_mse"].shape == ()
    assert outputs["per_param_mse"].shape == (_NUM_PARAMS,)
    assert outputs["param_mse"].item() == 0.0
    assert torch.equal(outputs["per_param_mse"], torch.zeros(_NUM_PARAMS))


def test_fake_oracle_net_exposes_d_out_and_has_grad_parameter() -> None:
    """``FakeOracleNet`` stores ``d_out`` and exposes at least one trainable parameter."""
    net = FakeOracleNet(d_out=_NUM_PARAMS)

    assert net.d_out == _NUM_PARAMS
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert len(trainable) >= 1
