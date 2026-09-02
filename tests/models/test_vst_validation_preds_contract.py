"""Pin the validation outputs consumed by VST callbacks.

``ValAudioProbe`` stages ``outputs["preds"]`` and ``LogPerParamMSE`` consumes
``outputs["per_param_mse"]``. Each module is built at tiny sizes and driven
without mocks so missing or malformed callback inputs fail here.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.components.vae import VAEOutput
from synth_setter.models.vst_fake_oracle_module import FakeOracleNet, VSTFakeOracleModule
from synth_setter.models.vst_ff_module import VSTFeedForwardModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.models.vst_flowvae_module import VSTFlowVAEModule
from synth_setter.utils.callbacks import LogPerParamMSE

type _VstModule = (
    VSTFeedForwardModule | VSTFakeOracleModule | VSTFlowMatchingModule | VSTFlowVAEModule
)

_NUM_PARAMS = 7
_BATCH = 3
_MEL_CHANNELS = 2
_MEL_N_MELS = 4
_MEL_N_FRAMES = 5


def _batch() -> dict[str, torch.Tensor]:
    """Return a batch carrying the keys every VST module's step functions read.

    :returns: Batch dict with ``params`` and ``mel``.
    """
    return {
        "params": torch.rand(_BATCH, _NUM_PARAMS),
        "mel": torch.rand(_BATCH, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES),
    }


class _TinyNet(torch.nn.Module):
    """Flattening linear net mapping a mel spec to ``_NUM_PARAMS`` predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_MEL_CHANNELS * _MEL_N_MELS * _MEL_N_FRAMES, _NUM_PARAMS)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Map ``mel`` to a ``(B, _NUM_PARAMS)`` prediction.

        :param mel: Batch of mel spectrograms.
        :returns: Predicted parameter tensor.
        """
        return self.linear(mel.flatten(start_dim=1))


class _TinyFlowVAENet(_TinyNet):
    """Produce a complete Flow-VAE output from a tiny learned projection."""

    def forward(self, mel_spec: torch.Tensor) -> VAEOutput:
        """Map ``mel_spec`` to predictions and valid latent-loss tensors.

        :param mel_spec: Batch of mel spectrograms.
        :returns: Flow-VAE output consumed by the production loss function.
        """
        x_hat = super().forward(mel_spec)
        zeros = torch.zeros_like(x_hat)
        return VAEOutput(
            y_hat=mel_spec,
            x_hat=x_hat,
            z_0=x_hat,
            z_k=x_hat,
            mu=x_hat,
            log_var=zeros,
            log_det_jacobian=zeros[:, 0],
        )


def _feed_forward_module() -> VSTFeedForwardModule:
    """Build a tiny real feed-forward module.

    :returns: Module wired for the test batch shapes.
    """
    return VSTFeedForwardModule(
        net=_TinyNet(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
    )


def _fake_oracle_module() -> VSTFakeOracleModule:
    """Build a tiny real fake-oracle module.

    :returns: Module wired for the test batch shapes.
    """
    return VSTFakeOracleModule(
        net=FakeOracleNet(d_out=_NUM_PARAMS),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
    )


def _flow_vae_module() -> VSTFlowVAEModule:
    """Build a tiny Flow-VAE module with production loss computation.

    :returns: Module wired for the test batch shapes and surge_4 parameter spec.
    """
    return VSTFlowVAEModule(
        net=_TinyFlowVAENet(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        param_spec="surge_4",
    )


def _flow_matching_module() -> VSTFlowMatchingModule:
    """Build a tiny real flow-matching module with a 2-step validation sampler.

    :returns: Module wired for the test batch shapes.
    """
    vector_field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=16,
            d_token=16,
            num_params=_NUM_PARAMS,
            num_tokens=4,
            initial_ffn=True,
            final_ffn=False,
        ),
        num_layers=1,
        d_model=16,
        conditioning_dim=16,
        num_heads=2,
        d_ff=16,
        num_tokens=4,
        learn_projection=True,
        time_encoding="sinusoidal",
        zero_init=False,
    )
    return VSTFlowMatchingModule(
        encoder=_TinyEncoder(),
        vector_field=vector_field,
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_NUM_PARAMS,
        validation_sample_steps=2,
        validation_cfg_strength=1.0,
    )


class _TinyEncoder(torch.nn.Module):
    """Conditioning encoder mapping a mel spec to a ``(B, 1, 16)`` conditioning token."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.linear = torch.nn.Linear(_MEL_CHANNELS * _MEL_N_MELS * _MEL_N_FRAMES, 16)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Map ``mel`` to a single conditioning token per sample.

        :param mel: Batch of mel spectrograms.
        :returns: Conditioning tensor of shape ``(B, 1, 16)``.
        """
        self.calls += 1
        return self.linear(mel.flatten(start_dim=1)).unsqueeze(1)


@pytest.mark.parametrize(
    "build_module",
    [_feed_forward_module, _fake_oracle_module, _flow_matching_module, _flow_vae_module],
    ids=["feed_forward", "fake_oracle", "flow_matching", "flow_vae"],
)
def test_validation_step_returns_callback_metrics_shaped_like_target_params(
    build_module: Callable[[], _VstModule],
) -> None:
    """Every VST module returns callback inputs shaped like the target parameters.

    :param build_module: Factory for the module under test.
    """
    module = build_module()
    batch = _batch()

    outputs = module.validation_step(batch, batch_idx=0)

    assert "preds" in outputs
    assert outputs["preds"].shape == batch["params"].shape
    assert outputs["per_param_mse"].shape == batch["params"].shape[1:]
    # Finiteness, not range: raw predictions are unbounded by design (linear/flow
    # outputs); decode_model_output owns the mapping into parameter space.
    assert torch.isfinite(outputs["preds"]).all()


def test_flow_matching_test_step_returns_predictions_and_flow_loss() -> None:
    """Flow test output carries the terminal patch and fully conditioned flow loss."""
    module = _flow_matching_module()
    batch = _batch()
    batch["noise"] = torch.randn_like(batch["params"])

    outputs = module.test_step(batch, batch_idx=0)

    assert outputs["preds"].shape == batch["params"].shape
    assert outputs["flow_loss"].ndim == 0
    assert torch.isfinite(outputs["flow_loss"])


def test_flow_matching_weighted_flow_loss_averages_rows_and_coordinates() -> None:
    """Flow loss uses the mean coordinate error and mean row reduction."""
    module = _flow_matching_module()
    prediction = torch.zeros(3, 7)
    target = torch.tensor([[1.0] * 7, [2.0] * 7, [3.0] * 7])
    t = torch.tensor([[0.1], [0.5], [0.9]])

    loss = module._weighted_flow_loss(prediction, target, t)

    torch.testing.assert_close(loss, torch.tensor(14.0 / 3.0))


def test_flow_matching_test_step_encodes_conditioning_once() -> None:
    """Sampling and flow-loss evaluation share one conditioning encoder result."""
    module = _flow_matching_module()
    batch = _batch()
    batch["noise"] = torch.randn_like(batch["params"])

    module.test_step(batch, batch_idx=0)

    assert isinstance(module.encoder, _TinyEncoder)
    assert module.encoder.calls == 1


def test_flow_matching_test_prediction_ignores_target_matched_batch_noise() -> None:
    """Terminal inference starts from an independent prior rather than OT noise."""
    module = _flow_matching_module()
    first = _batch()
    second = {**first, "noise": torch.full_like(first["params"], 100.0)}
    first["noise"] = torch.full_like(first["params"], -100.0)

    torch.manual_seed(41)
    first_outputs = module.test_step(first, batch_idx=0)
    torch.manual_seed(41)
    second_outputs = module.test_step(second, batch_idx=0)

    torch.testing.assert_close(first_outputs["preds"], second_outputs["preds"])
    assert not torch.equal(first_outputs["flow_loss"], second_outputs["flow_loss"])


def test_flow_vae_validation_step_returns_per_param_mse() -> None:
    """Flow-VAE validation exposes the per-parameter metric consumed by callbacks."""
    module = _flow_vae_module()
    batch = _batch()
    offsets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).repeat(_BATCH, 1)
    batch["params"] = module.net(batch["mel"]).x_hat.detach() + offsets

    outputs = module.validation_step(batch, batch_idx=0)

    torch.testing.assert_close(
        outputs["per_param_mse"], torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])
    )


def test_flow_vae_validation_loop_emits_per_param_metric() -> None:
    """Lightning dispatches Flow-VAE outputs through the default callback contract."""
    trainer = Trainer(
        accelerator="cpu",
        callbacks=[LogPerParamMSE("surge_4")],
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )

    dataloader = DataLoader(cast(Dataset[dict[str, torch.Tensor]], [_batch()]), batch_size=None)
    trainer.validate(_flow_vae_module(), dataloaders=dataloader)

    assert torch.isfinite(trainer.callback_metrics["per_param_mse/a_amp_eg_attack"])


def test_validation_step_preds_are_the_feed_forward_nets_predictions() -> None:
    """The feed-forward module's ``preds`` is its net's output, not a placeholder."""
    module = _feed_forward_module()
    batch = _batch()

    outputs = module.validation_step(batch, batch_idx=0)

    expected = module.net(batch["mel"])
    assert torch.allclose(outputs["preds"], expected)


def test_validation_step_preds_are_the_oracle_targets() -> None:
    """The oracle predicts ``batch["params"]`` verbatim, so ``preds`` equals the targets."""
    module = _fake_oracle_module()
    batch = _batch()

    outputs = module.validation_step(batch, batch_idx=0)

    assert torch.equal(outputs["preds"], batch["params"])


def test_validation_step_preds_depend_on_input() -> None:
    """Different mel inputs produce different feed-forward predictions.

    Guards against a module that returns constants (e.g. a detached head or a dead input path)
    while still passing the shape and finiteness pins.
    """
    torch.manual_seed(0)
    module = _feed_forward_module()
    batch_a = _batch()
    batch_b = {**batch_a, "mel": batch_a["mel"] + 1.0}

    preds_a = module.validation_step(batch_a, batch_idx=0)["preds"]
    preds_b = module.validation_step(batch_b, batch_idx=0)["preds"]

    assert not torch.allclose(preds_a, preds_b)


def test_flow_matching_validation_preds_vary_with_sampling_noise() -> None:
    """The flow-matching sampler is stochastic by design: fresh noise, fresh preds.

    Pins that validation sampling draws new noise per call rather than caching a
    trajectory, while the shape/finiteness contract holds for every draw.
    """
    torch.manual_seed(0)
    module = _flow_matching_module()
    batch = _batch()

    preds_a = module.validation_step(batch, batch_idx=0)["preds"]
    preds_b = module.validation_step(batch, batch_idx=0)["preds"]

    assert preds_a.shape == preds_b.shape == batch["params"].shape
    assert torch.isfinite(preds_a).all() and torch.isfinite(preds_b).all()
    assert not torch.equal(preds_a, preds_b)
