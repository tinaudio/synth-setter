"""Pin global and number-group-constrained parameter swap metrics.

Global best-swap remains unconditional. Selecting a ParamSpec also logs the
structured middle bound ``best_swap <= number_group_swap <= param_mse``.
"""

from __future__ import annotations

import math
from functools import partial

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset

from synth_setter.data.vst import param_specs
from synth_setter.metrics import BestSwapParamMSE, NumberGroupSwapParamMSE
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.utils.callbacks import LogPerParamMSE

_MEL_CHANNELS = 2
_MEL_N_MELS = 4
_MEL_N_FRAMES = 5
_D_MODEL = 16


class _TinyEncoder(torch.nn.Module):
    """Conditioning encoder mapping a mel spec to a ``(B, 1, _D_MODEL)`` token."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_MEL_CHANNELS * _MEL_N_MELS * _MEL_N_FRAMES, _D_MODEL)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Map ``mel`` to a single conditioning token per sample.

        :param mel: Batch of mel spectrograms.
        :returns: Conditioning tensor of shape ``(B, 1, _D_MODEL)``.
        """
        return self.linear(mel.flatten(start_dim=1)).unsqueeze(1)


class _FakeBatchDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed random samples shaped like the VST datamodule's batches."""

    def __init__(self, num_params: int) -> None:
        """Materialize the fixed samples.

        :param num_params: Width of each ``params`` row.
        """
        generator = torch.Generator().manual_seed(0)
        self._params = torch.rand(4, num_params, generator=generator)
        self._mels = torch.rand(4, _MEL_CHANNELS, _MEL_N_MELS, _MEL_N_FRAMES, generator=generator)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample carrying the keys the module's step functions read.

        :param index: Sample index.
        :returns: ``params`` / ``mel`` sample dict.
        """
        return {"params": self._params[index], "mel": self._mels[index]}


def _flow_module(num_params: int, *, param_spec: str | None = None) -> VSTFlowMatchingModule:
    """Build a tiny real flow-matching module with a 1-step sampler.

    :param num_params: Parameter-vector width.
    :param param_spec: Optional registered spec enabling structured swap metrics.
    :returns: Module wired for the fake batch shapes.
    """
    vector_field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=_D_MODEL,
            d_token=_D_MODEL,
            num_params=num_params,
            num_tokens=4,
            initial_ffn=True,
            final_ffn=False,
        ),
        num_layers=1,
        d_model=_D_MODEL,
        conditioning_dim=_D_MODEL,
        num_heads=2,
        d_ff=_D_MODEL,
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
        num_params=num_params,
        param_spec=param_spec,
        validation_sample_steps=1,
        validation_cfg_strength=1.0,
        test_sample_steps=1,
        test_cfg_strength=1.0,
    )


def _tiny_trainer(*, callbacks: list[Callback] | None = None) -> Trainer:
    """Build a minimal CPU trainer for one validation/test batch.

    :param callbacks: Optional callbacks to exercise with the loop.
    :returns: Silent single-batch CPU trainer.
    """
    return Trainer(
        callbacks=callbacks,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=1,
        limit_test_batches=1,
    )


def test_ctor_instantiates_best_swap_metrics_unconditionally() -> None:
    """Both loop metrics exist without any spec plumbing."""
    module = _flow_module(6)

    assert isinstance(module.val_param_mse_best_swap, BestSwapParamMSE)
    assert isinstance(module.test_param_mse_best_swap, BestSwapParamMSE)


def test_ctor_instantiates_number_group_swap_metrics_with_param_spec() -> None:
    """Selecting a ParamSpec enables both structured loop metrics."""
    spec = param_specs["surge_simple"]

    module = _flow_module(spec.encoded_width, param_spec="surge_simple")

    assert isinstance(module.val_param_mse_number_group_swap, NumberGroupSwapParamMSE)
    assert isinstance(module.test_param_mse_number_group_swap, NumberGroupSwapParamMSE)


def test_validation_loop_logs_best_swap_alongside_param_mse() -> None:
    """``val/param_mse_best_swap`` lands beside ``val/param_mse`` and never exceeds it."""
    module = _flow_module(6)
    loader = DataLoader(_FakeBatchDataset(6), batch_size=2)

    metrics = _tiny_trainer().validate(module, dataloaders=loader)[0]

    assert "val/param_mse_best_swap" in metrics
    assert "val/param_mse" in metrics
    assert metrics["val/param_mse_best_swap"] <= metrics["val/param_mse"] + 1e-6


def test_validation_loop_logs_number_group_swap_metrics() -> None:
    """Structured swap MSE lands between global best-swap and fixed-order MSE."""
    spec = param_specs["surge_simple"]
    module = _flow_module(spec.encoded_width, param_spec="surge_simple")
    loader = DataLoader(_FakeBatchDataset(spec.encoded_width), batch_size=2)
    trainer = _tiny_trainer(callbacks=[LogPerParamMSE("surge_simple")])

    metrics = trainer.validate(module, dataloaders=loader)[0]

    assert "val/param_mse_number_group_swap" in metrics
    assert "val/per_param_mse_number_group_swap/a_osc_1_pitch" in metrics
    assert metrics["val/param_mse_best_swap"] <= metrics["val/param_mse_number_group_swap"]
    assert metrics["val/param_mse_number_group_swap"] <= metrics["val/param_mse"] + 1e-6


def test_validation_loop_logs_per_param_best_swap() -> None:
    """The callback publishes best-swap errors under target parameter names."""
    module = _flow_module(7)
    loader = DataLoader(_FakeBatchDataset(7), batch_size=2)
    trainer = _tiny_trainer(callbacks=[LogPerParamMSE("surge_4")])

    metrics = trainer.validate(module, dataloaders=loader)[0]

    assert "val/per_param_mse_best_swap/note_start_and_end" in metrics


def test_validation_loop_logs_spec_quantized_metrics() -> None:
    """The callback publishes scalar and per-parameter rendered-value errors."""
    spec = param_specs["surge_4"]
    module = _flow_module(spec.encoded_width, param_spec="surge_4")
    loader = DataLoader(_FakeBatchDataset(spec.encoded_width), batch_size=2)
    trainer = _tiny_trainer(callbacks=[LogPerParamMSE("surge_4")])

    metrics = trainer.validate(module, dataloaders=loader)[0]

    assert math.isfinite(metrics["val/param_mse_spec_quantized"])
    assert math.isfinite(metrics["val/per_param_mse_spec_quantized/a_amp_eg_attack"])


def test_pyfdn_validation_loop_logs_all_per_param_metric_families() -> None:
    """PyFDN validation publishes each per-parameter metric family under ``val``."""
    spec_name = "pyfdn_n8_mono_householder"
    spec = param_specs[spec_name]
    module = _flow_module(spec.encoded_width, param_spec=spec_name)
    loader = DataLoader(_FakeBatchDataset(spec.encoded_width), batch_size=2)

    metrics = _tiny_trainer(callbacks=[LogPerParamMSE(spec_name)]).validate(
        module, dataloaders=loader
    )[0]

    assert "val/per_param_mse/delays" in metrics
    assert "val/per_param_mse_best_swap/delays" in metrics
    assert "val/per_param_mse_number_group_swap/delays" in metrics
    assert "val/per_param_mse_spec_quantized/delays" in metrics


def test_pyfdn_test_loop_logs_all_per_param_metric_families() -> None:
    """PyFDN test publishes each per-parameter metric family under ``test``."""
    spec_name = "pyfdn_n8_mono_householder"
    spec = param_specs[spec_name]
    module = _flow_module(spec.encoded_width, param_spec=spec_name)
    loader = DataLoader(_FakeBatchDataset(spec.encoded_width), batch_size=2)

    metrics = _tiny_trainer(callbacks=[LogPerParamMSE(spec_name)]).test(
        module, dataloaders=loader
    )[0]

    assert "test/per_param_mse/delays" in metrics
    assert "test/per_param_mse_best_swap/delays" in metrics
    assert "test/per_param_mse_number_group_swap/delays" in metrics
    assert "test/per_param_mse_spec_quantized/delays" in metrics


def test_test_loop_logs_number_group_swap() -> None:
    """The test loop emits the structured scalar metric when a spec is selected."""
    spec = param_specs["surge_simple"]
    module = _flow_module(spec.encoded_width, param_spec="surge_simple")
    loader = DataLoader(_FakeBatchDataset(spec.encoded_width), batch_size=2)
    trainer = _tiny_trainer(callbacks=[LogPerParamMSE("surge_simple")])

    metrics = trainer.test(module, dataloaders=loader)[0]

    assert "test/param_mse_number_group_swap" in metrics
    assert "test/per_param_mse/a_osc_1_pitch" in metrics
    assert "test/per_param_mse_best_swap/a_osc_1_pitch" in metrics
    assert "test/per_param_mse_number_group_swap/a_osc_1_pitch" in metrics
    assert "test/per_param_mse_spec_quantized/a_osc_1_pitch" in metrics
    assert "test/param_mse_spec_quantized" in metrics
    assert metrics["test/param_mse_best_swap"] <= metrics["test/param_mse_number_group_swap"]
    assert metrics["test/param_mse_number_group_swap"] <= metrics["test/param_mse"] + 1e-6


def test_test_loop_logs_best_swap() -> None:
    """``test/param_mse_best_swap`` is logged by the test loop."""
    module = _flow_module(6)
    loader = DataLoader(_FakeBatchDataset(6), batch_size=2)

    metrics = _tiny_trainer().test(module, dataloaders=loader)[0]

    assert "test/param_mse_best_swap" in metrics
