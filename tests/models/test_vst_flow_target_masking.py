"""Opt-in target masking drops coordinates the selected categorical makes inaudible.

The synthetic spec encodes a one-hot ``fx_type`` (columns 0-2), ``slot1`` (3, audible under
A and B), ``slot2`` (4, A only), a one-hot ``mode`` (5-6, C only), and an ungated ``shared``
(7), so a row selecting A has 6 active columns and a row selecting B has 5.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from synth_setter.data.vst import param_spec_registry
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    ParamSpec,
    SelectorGate,
)
from synth_setter.models.components.transformer import ApproxEquivTransformer, LearntProjection
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.utils.callbacks import LogPerParamMSE

_GATED = "target_masking_gated"
_UNGATED = "target_masking_ungated"
_WIDTH = 8
_CONDITIONING_WIDTH = 4
_SIGNAL_LENGTH = 6
_MODE_COLUMNS = slice(5, 7)
# Row 0 selects A (mode inert); row 1 selects B (slot2 and mode inert).
_TARGET = torch.tensor(
    [
        [1.0, -1.0, -1.0, 0.5, 0.5, 0.9, 0.9, 0.0],
        [-1.0, 1.0, -1.0, 0.0, 0.9, 0.9, 0.9, 0.5],
    ]
)
# Squared active entries against a zero prediction: 3 + 0.25 + 0.25 + 0 over 6 columns,
# and 3 + 0 + 0.25 over 5 columns.
_ROW_A_ACTIVE_MEAN = 3.5 / 6
_ROW_B_ACTIVE_MEAN = 3.25 / 5


def _synth_params() -> list:
    return [
        CategoricalParameter(name="fx_type", values=["A", "B", "C"], encoding="onehot"),
        ContinuousParameter(name="slot1", min=0.0, max=1.0),
        ContinuousParameter(name="slot2", min=0.0, max=1.0),
        CategoricalParameter(name="mode", values=["x", "y"], encoding="onehot"),
        ContinuousParameter(name="shared", min=0.0, max=1.0),
    ]


@pytest.fixture(autouse=True)
def _register_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register gated and ungated variants of the synthetic spec.

    :param monkeypatch: Restores the registry after each test.
    """
    gate = SelectorGate(
        selector="fx_type",
        blocks={"A": ("slot1", "slot2"), "B": ("slot1",), "C": ("mode",)},
    )
    monkeypatch.setitem(
        param_spec_registry._param_specs,
        ParamSpecName(_GATED),
        ParamSpec(_synth_params(), [], gate=gate),
    )
    monkeypatch.setitem(
        param_spec_registry._param_specs,
        ParamSpecName(_UNGATED),
        ParamSpec(_synth_params(), []),
    )


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_SIGNAL_LENGTH, _CONDITIONING_WIDTH)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode raw audio.

        :param audio: Audio batch.
        :returns: One conditioning token per row.
        """
        return self.linear(audio).unsqueeze(1)


class _ConstantField(torch.nn.Module):
    """Emit one trainable row for every input, so predictions are known exactly.

    .. attribute :: d_model

       Conditioning width the module reads off its field.
    """

    d_model = _CONDITIONING_WIDTH

    def __init__(self, row: torch.Tensor | None = None) -> None:
        """Hold the emitted row.

        :param row: Initial row; zeros by default.
        """
        super().__init__()
        self.row = torch.nn.Parameter(torch.zeros(_WIDTH) if row is None else row.clone())

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Broadcast the row.

        :param x: Parameter state.
        :param t: Flow time.
        :param conditioning: Ignored conditioning.
        :returns: The shared row per input row.
        """
        return self.row.expand(x.shape[0], -1)

    def apply_dropout(
        self, conditioning: torch.Tensor, rate: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep every conditioning row.

        :param conditioning: Conditioning batch.
        :param rate: Ignored rate.
        :returns: Unchanged conditioning and an all-true keep mask.
        """
        return conditioning, torch.ones(conditioning.shape[0], dtype=torch.bool)


def _module(
    vector_field: torch.nn.Module | None = None,
    *,
    param_spec: str | None = _GATED,
    mask_inert_targets: bool = False,
    endpoint_loss: str = "mse",
) -> VSTFlowMatchingModule:
    """Build a raw-audio flow module over the synthetic spec.

    :param vector_field: Field under test; defaults to a zero constant row.
    :param param_spec: Registered spec name, or ``None`` for no spec.
    :param mask_inert_targets: Whether inert target coordinates are masked.
    :param endpoint_loss: Endpoint objective forwarded to the module.
    :returns: Module with one-step sampling.
    """
    torch.manual_seed(0)
    return VSTFlowMatchingModule(
        encoder=_Encoder(),
        vector_field=vector_field if vector_field is not None else _ConstantField(),
        optimizer=partial(torch.optim.SGD, lr=0.1),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_WIDTH,
        param_spec=param_spec,
        conditioning="audio",
        mask_inert_targets=mask_inert_targets,
        endpoint_loss=endpoint_loss,  # pyright: ignore[reportArgumentType]
        parameterization="endpoint" if endpoint_loss == "mixed" else "velocity",
        validation_sample_steps=1,
        validation_cfg_strength=1.0,
        test_sample_steps=1,
        test_cfg_strength=1.0,
    )


def _batch(target: torch.Tensor = _TARGET) -> dict[str, torch.Tensor]:
    """Return a batch whose zero noise makes the velocity target equal ``target``.

    :param target: Model-space parameter rows.
    :returns: ``params`` / ``noise`` / ``audio`` batch.
    """
    return {
        "params": target.clone(),
        "noise": torch.zeros_like(target),
        "audio": torch.ones(target.shape[0], _SIGNAL_LENGTH),
    }


def _fixed_time(module: VSTFlowMatchingModule) -> VSTFlowMatchingModule:
    """Pin flow time to 0.5 so repeated steps see identical paths.

    :param module: Module to patch.
    :returns: The same module.
    """
    module._sample_time = lambda n, device: torch.full((n, 1), 0.5, device=device)  # type: ignore[method-assign]
    return module


def _inert_perturbed(target: torch.Tensor) -> torch.Tensor:
    """Move only coordinates inert in both rows: ``mode`` everywhere, ``slot2`` on row B.

    :param target: Rows selecting A then B.
    :returns: A copy differing only on inert coordinates.
    """
    perturbed = target.clone()
    perturbed[:, _MODE_COLUMNS] = -0.3
    perturbed[1, 4] = -0.7
    return perturbed


def test_train_step_masking_off_with_gate_matches_ungated_spec_bitwise() -> None:
    """Declaring a gate without opting in leaves loss and train metrics bit-identical."""
    row = torch.linspace(-0.4, 0.6, _WIDTH)
    gated = _fixed_time(_module(_ConstantField(row), param_spec=_GATED))
    ungated = _fixed_time(_module(_ConstantField(row), param_spec=_UNGATED))

    gated_out = gated._train_step(_batch())
    ungated_out = ungated._train_step(_batch())

    assert torch.equal(gated_out.loss, ungated_out.loss)
    assert torch.equal(gated_out.per_param_flow_mse, ungated_out.per_param_flow_mse)
    assert torch.equal(gated_out.per_param_endpoint_mse, ungated_out.per_param_endpoint_mse)


def test_train_step_masking_off_divides_by_full_width() -> None:
    """Without masking, every row averages over all eight columns, inert ones included."""
    module = _fixed_time(_module(mask_inert_targets=False))

    loss = module._train_step(_batch()).loss

    expected = (_TARGET.square().sum(dim=-1) / _WIDTH).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_train_step_masking_on_divides_by_active_count() -> None:
    """Each row averages over its own active columns, then rows average equally."""
    module = _fixed_time(_module(mask_inert_targets=True))

    loss = module._train_step(_batch()).loss

    assert loss.item() == pytest.approx((_ROW_A_ACTIVE_MEAN + _ROW_B_ACTIVE_MEAN) / 2, rel=1e-6)


def test_train_step_masking_on_inert_target_perturbation_leaves_loss_unchanged() -> None:
    """Changing an inert target moves neither the loss nor the gradient it produces."""
    field = _ConstantField()
    module = _fixed_time(_module(field, mask_inert_targets=True))

    reference = module._train_step(_batch()).loss
    perturbed = module._train_step(_batch(_inert_perturbed(_TARGET))).loss
    (reference_grad,) = torch.autograd.grad(reference, field.row)
    (perturbed_grad,) = torch.autograd.grad(perturbed, field.row)

    assert torch.equal(perturbed, reference)
    assert torch.equal(perturbed_grad, reference_grad)
    assert torch.count_nonzero(reference_grad[_MODE_COLUMNS]) == 0


def test_eval_outputs_masking_off_with_gate_match_ungated_spec_bitwise() -> None:
    """Without opting in, the evaluation outputs carry the same keys and values."""
    predicted = torch.linspace(-1.0, 1.0, 2 * _WIDTH).reshape(2, _WIDTH)

    gated = _module(param_spec=_GATED)._per_param_mse_outputs(predicted, _TARGET, None)
    ungated = _module(param_spec=_UNGATED)._per_param_mse_outputs(predicted, _TARGET, None)

    assert gated.keys() == ungated.keys()
    assert all(torch.equal(gated[key], ungated[key]) for key in gated)


def test_eval_outputs_masking_on_report_active_means() -> None:
    """The scalar is the per-row active mean; per-column values average active rows only."""
    module = _module(mask_inert_targets=True)

    outputs = module._per_param_mse_outputs(torch.zeros_like(_TARGET), _TARGET, None)

    assert outputs["param_mse_active"].item() == pytest.approx(
        (_ROW_A_ACTIVE_MEAN + _ROW_B_ACTIVE_MEAN) / 2, rel=1e-6
    )
    # slot2 is active on row A only (0.25); mode on neither row; shared on both (0, 0.25).
    expected_columns = torch.tensor([1.0, 1.0, 1.0, 0.125, 0.25, 0.0, 0.0, 0.125])
    torch.testing.assert_close(outputs["per_param_mse_active"], expected_columns)
    assert outputs["param_mse"].item() == pytest.approx(_TARGET.square().mean().item(), rel=1e-6)


def test_eval_outputs_masking_on_inert_perturbation_leaves_active_metrics_unchanged() -> None:
    """Moving inert targets or inert predictions changes no active metric."""
    module = _module(mask_inert_targets=True)
    predicted = torch.linspace(-1.0, 1.0, 2 * _WIDTH).reshape(2, _WIDTH)

    reference = module._per_param_mse_outputs(predicted, _TARGET, None)
    moved_target = module._per_param_mse_outputs(predicted, _inert_perturbed(_TARGET), None)
    moved_prediction = module._per_param_mse_outputs(_inert_perturbed(predicted), _TARGET, None)

    for outputs in (moved_target, moved_prediction):
        assert torch.equal(outputs["param_mse_active"], reference["param_mse_active"])
        assert torch.equal(outputs["per_param_mse_active"], reference["per_param_mse_active"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"param_spec": None}, "requires param_spec"),
        ({"param_spec": _UNGATED}, "no selector gate"),
        ({"endpoint_loss": "mixed"}, "endpoint_loss='mixed'"),
    ],
)
def test_module_masking_on_unsupported_configuration_raises(
    kwargs: dict[str, object], message: str
) -> None:
    """Masking needs a gated spec and the flat objective; anything else fails at build time.

    :param kwargs: Module overrides that cannot support masking.
    :param message: Fragment the error must name.
    """
    with pytest.raises(ValueError, match=message):
        _module(mask_inert_targets=True, **kwargs)  # type: ignore[arg-type]


class _RecordingModule:
    """Stand-in LightningModule capturing the callback's ``log_dict`` call."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def log_dict(self, metrics: dict[str, float]) -> None:
        """Record emitted metrics.

        :param metrics: Metric name to value mapping.
        """
        self.logged.update(metrics)


def test_log_per_param_mse_active_outputs_weight_columns_by_active_rows() -> None:
    """Each column averages over the rows where it was active, across ragged batches."""
    callback = LogPerParamMSE(param_spec=_GATED)
    recorder = _RecordingModule()
    trainer = cast(Trainer, None)
    pl_module = cast(LightningModule, recorder)
    module = _module(mask_inert_targets=True)
    # Batch 1 rows select A and B; batch 2 is one A row with a different slot2 error.
    batches = [_TARGET, _TARGET[:1].clone()]
    batches[1][0, 4] = 0.1

    callback.on_validation_epoch_start(trainer, pl_module)
    for index, target in enumerate(batches):
        outputs = module._per_param_mse_outputs(torch.zeros_like(target), target, None)
        callback.on_validation_batch_end(trainer, pl_module, outputs, {"params": target}, index)
    callback.on_validation_epoch_end(trainer, pl_module)

    # slot2: rows A (0.25) and A (0.01) -> 0.13; slot1: A, B, A (0.25, 0, 0.25) -> 1/6.
    assert recorder.logged["val/per_param_mse_active/slot2"] == pytest.approx(0.13, rel=1e-6)
    assert recorder.logged["val/per_param_mse_active/slot1"] == pytest.approx(1 / 6, rel=1e-6)
    assert "val/per_param_mse_active/mode" not in recorder.logged
    assert "val/per_param_mse/mode" in recorder.logged


class _GatedDataset(Dataset[dict[str, torch.Tensor]]):
    """Rows cycling through every selector value with random continuous targets."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self._params = torch.rand(6, _WIDTH, generator=generator) * 2 - 1
        self._params[:, :3] = -1.0
        self._params[torch.arange(6), torch.arange(6) % 3] = 1.0
        self._noise = torch.randn(6, _WIDTH, generator=generator)
        self._audio = torch.randn(6, _SIGNAL_LENGTH, generator=generator)

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample.

        :param index: Sample index.
        :returns: ``params`` / ``noise`` / ``audio`` sample.
        """
        return {
            "params": self._params[index],
            "noise": self._noise[index],
            "audio": self._audio[index],
        }


def test_fit_and_validate_masking_on_logs_finite_active_metrics() -> None:
    """A real transformer trains and validates with masking on through Lightning."""
    field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=_CONDITIONING_WIDTH,
            d_token=_CONDITIONING_WIDTH,
            num_params=_WIDTH,
            num_tokens=2,
            initial_ffn=True,
            final_ffn=False,
        ),
        num_layers=1,
        d_model=_CONDITIONING_WIDTH,
        conditioning_dim=_CONDITIONING_WIDTH,
        num_heads=1,
        d_ff=_CONDITIONING_WIDTH,
        num_tokens=2,
        learn_projection=True,
        time_encoding="sinusoidal",
        zero_init=False,
    )
    module = _module(field, mask_inert_targets=True)
    loader = DataLoader(_GatedDataset(), batch_size=3)
    trainer = Trainer(
        accelerator="cpu",
        max_steps=2,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[LogPerParamMSE(param_spec=_GATED)],
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    metrics = trainer.callback_metrics
    assert torch.isfinite(metrics["train/loss"])
    assert torch.isfinite(metrics["val/param_mse_active"])
    assert torch.isfinite(metrics["val/per_param_mse_active/slot1"])
