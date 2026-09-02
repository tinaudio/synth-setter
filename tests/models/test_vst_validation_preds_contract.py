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

from synth_setter.data.vst import param_specs
from synth_setter.data.vst.param_spec import ContinuousParameter, ParamSpec
from synth_setter.metrics import midi_pitch_residuals
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


def _flow_matching_module(
    *,
    validation_noise_seed: int | None = None,
    validation_cfg_strength: float = 1.0,
    param_spec: str | None = "surge_4",
) -> VSTFlowMatchingModule:
    """Build a tiny real flow-matching module with a 2-step validation sampler.

    :param validation_noise_seed: Optional deterministic validation-noise seed.
    :param validation_cfg_strength: Content guidance scale used by validation.
    :param param_spec: Registered spec used by structured validation metrics.
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
        param_spec=param_spec,
        validation_sample_steps=2,
        validation_cfg_strength=validation_cfg_strength,
        validation_noise_seed=validation_noise_seed,
    )


def _zero_velocity_flow_matching_module(
    *, validation_noise_seed: int, validation_cfg_strength: float = 1.0
) -> VSTFlowMatchingModule:
    """Build a flow whose sample equals its initial noise.

    :param validation_noise_seed: Deterministic validation-noise seed.
    :param validation_cfg_strength: Content guidance scale used by validation.
    :returns: Module with a zero velocity field.
    """
    module = _flow_matching_module(
        validation_noise_seed=validation_noise_seed,
        validation_cfg_strength=validation_cfg_strength,
    )
    for parameter in module.vector_field.parameters():
        torch.nn.init.zeros_(parameter)
    return module


class _TinyEncoder(torch.nn.Module):
    """Conditioning encoder mapping a mel spec to a ``(B, 1, 16)`` conditioning token."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_MEL_CHANNELS * _MEL_N_MELS * _MEL_N_FRAMES, 16)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Map ``mel`` to a single conditioning token per sample.

        :param mel: Batch of mel spectrograms.
        :returns: Conditioning tensor of shape ``(B, 1, 16)``.
        """
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
    """The default validation sampler draws fresh noise for every call."""
    torch.manual_seed(0)
    module = _flow_matching_module()
    batch = _batch()

    preds_a = module.validation_step(batch, batch_idx=0)["preds"]
    preds_b = module.validation_step(batch, batch_idx=0)["preds"]

    assert preds_a.shape == preds_b.shape == batch["params"].shape
    assert torch.isfinite(preds_a).all() and torch.isfinite(preds_b).all()
    assert not torch.equal(preds_a, preds_b)


def test_flow_matching_validation_seed_reuses_noise_across_cfg_strengths() -> None:
    """A validation seed holds initial flow noise fixed across CFG comparisons."""
    cfg_zero = _zero_velocity_flow_matching_module(
        validation_noise_seed=3018, validation_cfg_strength=0.0
    )
    cfg_two = _zero_velocity_flow_matching_module(
        validation_noise_seed=3018, validation_cfg_strength=2.0
    )
    batch = _batch()

    cfg_zero_preds = cfg_zero.validation_step(batch, batch_idx=0)["preds"]
    cfg_two_preds = cfg_two.validation_step(batch, batch_idx=0)["preds"]

    assert torch.equal(cfg_zero_preds, cfg_two_preds)


def test_flow_matching_validation_seed_changes_each_batch_stream() -> None:
    """Successive batch indices receive distinct deterministic initial states."""
    module = _zero_velocity_flow_matching_module(validation_noise_seed=3018)
    batch = _batch()

    first_batch_preds = module.validation_step(batch, batch_idx=0)["preds"]
    second_batch_preds = module.validation_step(batch, batch_idx=1)["preds"]

    assert not torch.equal(first_batch_preds, second_batch_preds)


def test_flow_matching_validation_seed_repeats_within_rank_and_differs_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed batch stream repeats locally while each DDP rank receives distinct noise.

    :param monkeypatch: Rank-property replacement scoped to this test.
    """
    module = _zero_velocity_flow_matching_module(validation_noise_seed=3018)
    batch = _batch()

    rank_zero_first = module.validation_step(batch, batch_idx=0)["preds"]
    rank_zero_second = module.validation_step(batch, batch_idx=0)["preds"]
    monkeypatch.setattr(
        VSTFlowMatchingModule,
        "global_rank",
        property(lambda _: 1),
    )
    rank_one = module.validation_step(batch, batch_idx=0)["preds"]

    assert torch.equal(rank_zero_first, rank_zero_second)
    assert not torch.equal(rank_zero_first, rank_one)


def test_flow_matching_validation_seed_outside_distinct_cpu_range_raises_value_error() -> None:
    """Seeds that alias a supported CPU stream are rejected at construction."""
    with pytest.raises(ValueError, match="distinct CPU seed range"):
        _zero_velocity_flow_matching_module(validation_noise_seed=4294970314)


def test_flow_matching_validation_batch_stream_seed_overflow_raises_value_error() -> None:
    """Derived batch/rank streams fail instead of wrapping onto another CPU stream."""
    module = _zero_velocity_flow_matching_module(validation_noise_seed=4294967295)

    with pytest.raises(ValueError, match="validation stream seed exceeds"):
        module.validation_step(_batch(), batch_idx=1)


def test_flow_matching_validation_seed_change_changes_predictions() -> None:
    """Changing the validation seed changes the sampled initial state."""
    first_module = _zero_velocity_flow_matching_module(validation_noise_seed=3018)
    second_module = _zero_velocity_flow_matching_module(validation_noise_seed=3019)
    batch = _batch()

    first_preds = first_module.validation_step(batch, batch_idx=0)["preds"]
    second_preds = second_module.validation_step(batch, batch_idx=0)["preds"]

    assert not torch.equal(first_preds, second_preds)


def test_flow_matching_validation_without_scalar_pitch_skips_pitch_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered spec without scalar MIDI pitch retains core validation metrics.

    :param monkeypatch: Registry projection replacement scoped to this test.
    """
    import synth_setter.data.vst as vst

    spec_without_pitch = ParamSpec(
        synth_params=[ContinuousParameter(f"param_{index}") for index in range(_NUM_PARAMS)],
        note_params=[],
    )
    monkeypatch.setattr(
        vst, "param_specs", {**vst.param_specs, "without_pitch": spec_without_pitch}
    )
    module = _flow_matching_module(param_spec="without_pitch")

    outputs = module.validation_step(_batch(), batch_idx=0)

    assert torch.isfinite(outputs["param_mse"])


def test_flow_matching_validation_loop_logs_signed_pitch_residuals() -> None:
    """Lightning aggregates all signed pitch residual diagnostics over validation."""
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    dataloader = DataLoader(cast(Dataset[dict[str, torch.Tensor]], [_batch()]), batch_size=None)

    trainer.validate(
        _flow_matching_module(validation_noise_seed=3018),
        dataloaders=dataloader,
    )

    assert torch.isfinite(trainer.callback_metrics["val/pitch_residual_continuous_mean_semitones"])
    assert torch.isfinite(trainer.callback_metrics["val/pitch_residual_floor_mean_semitones"])
    assert torch.isfinite(trainer.callback_metrics["val/pitch_residual_nearest_mean_semitones"])


def test_flow_matching_validation_loop_row_weights_signed_pitch_residuals() -> None:
    """Lightning averages signed pitch residuals over rows, not over batches."""
    module = _flow_matching_module(validation_noise_seed=3018)
    full_batch = _batch()
    first_batch = {name: values[:2] for name, values in full_batch.items()}
    second_batch = {name: values[2:] for name, values in full_batch.items()}
    first_predictions = module.validation_step(first_batch, batch_idx=0)["preds"]
    second_predictions = module.validation_step(second_batch, batch_idx=1)["preds"]
    expected = torch.cat(
        (
            midi_pitch_residuals(first_predictions, first_batch["params"], param_specs["surge_4"])[
                "continuous"
            ],
            midi_pitch_residuals(
                second_predictions, second_batch["params"], param_specs["surge_4"]
            )["continuous"],
        )
    ).mean()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    dataloader = DataLoader(
        cast(Dataset[dict[str, torch.Tensor]], [first_batch, second_batch]),
        batch_size=None,
    )

    trainer.validate(module, dataloaders=dataloader)

    torch.testing.assert_close(
        trainer.callback_metrics["val/pitch_residual_continuous_mean_semitones"], expected
    )
