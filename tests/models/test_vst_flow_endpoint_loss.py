"""Endpoint losses for mixed numerical and one-hot parameter spans."""

from __future__ import annotations

import math
from pathlib import Path

import lightning
import pytest
import torch
from omegaconf import OmegaConf

from synth_setter.data.vst.cardinal_param_spec import CARDINAL_PARAM_SPEC
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    ParamSpec,
)
from synth_setter.models.vst_flow_matching_module import (
    ControlTokenBranches,
    VSTFlowMatchingModule,
    endpoint_prediction_to_model,
    mixed_endpoint_row_loss,
)

_BATCH = 2
_CONDITIONING_DIM = 4
_SIGNAL_LENGTH = 8
_WIDTH = CARDINAL_PARAM_SPEC.encoded_width


class _WaveformEncoder(torch.nn.Module):
    """Minimal trainable raw-audio encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_SIGNAL_LENGTH, _CONDITIONING_DIM)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode raw audio into conditioning.

        :param audio: Audio batch.
        :returns: Flat conditioning batch.
        """
        return self.linear(audio)


class _ConstantField(torch.nn.Module):
    """Field exposing one trainable output row."""

    def __init__(self, row: torch.Tensor) -> None:
        """Store the field output shared by every row.

        :param row: Initial field output.
        """
        super().__init__()
        self.row = torch.nn.Parameter(row.clone())

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Broadcast the trainable row across the batch.

        :param x: Parameter state.
        :param t: Flow time.
        :param conditioning: Ignored conditioning.
        :returns: Shared trainable field output.
        """
        return self.row.expand(x.shape[0], -1)

    def apply_dropout(
        self, conditioning: torch.Tensor, rate: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep all conditioning rows.

        :param conditioning: Conditioning batch.
        :param rate: Ignored dropout rate.
        :returns: Unchanged conditioning and an all-true keep mask.
        """
        return conditioning, torch.ones(
            conditioning.shape[0], dtype=torch.bool, device=conditioning.device
        )


class _ConditionedLogitField(_ConstantField):
    """Emit opposite categorical logits for conditional and unconditional CFG branches."""

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Choose the branch-specific row from conditioning presence.

        :param x: Parameter state.
        :param t: Flow time.
        :param conditioning: Present only for the conditional branch.
        :returns: Conditional or unconditional logits.
        """
        row = self.row.clone()
        row[8:10] = torch.tensor(
            [2.0, 0.0] if conditioning is not None else [0.0, 2.0], device=x.device
        )
        return row.expand(x.shape[0], -1)


class _ConditionedControlLogitField(_ConstantField):
    """Emit distinct logits for all three multi-CFG branches."""

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        *,
        control_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Choose logits from content and control-token presence.

        :param x: Parameter state.
        :param t: Flow time.
        :param conditioning: Present only for the content-plus-sketch branch.
        :param control_tokens: Full-sketch or unconditional control tokens.
        :returns: Branch-specific logits.
        """
        row = self.row.clone()
        if conditioning is not None:
            logits = [2.0, 0.0]
        elif torch.count_nonzero(control_tokens):
            logits = [0.0, 2.0]
        else:
            logits = [-1.0, 1.0]
        row[8:10] = torch.tensor(logits, device=x.device)
        return row.expand(x.shape[0], -1)


class _RecordingAudioLoss(torch.nn.Module):
    """Record the endpoint handed to audio feedback."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint: torch.Tensor | None = None

    def forward(
        self,
        theta_hat: torch.Tensor,
        t: torch.Tensor,
        target_audio: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture the endpoint while retaining its gradient graph.

        :param theta_hat: Model-space endpoint estimate.
        :param t: Flow time.
        :param target_audio: Ignored target audio.
        :param keep: Ignored row keep mask.
        :returns: Zero scalar connected to ``theta_hat``.
        """
        self.endpoint = theta_hat.detach().clone()
        return theta_hat.sum() * 0.0


def _module(
    *,
    endpoint_loss: str = "mse",
    parameterization: str = "endpoint",
    param_spec: str | None = "cardinal",
    rectified_sigma_min: float = 0.0,
    row: torch.Tensor | None = None,
    audio_loss: torch.nn.Module | None = None,
    vector_field: torch.nn.Module | None = None,
) -> VSTFlowMatchingModule:
    """Build a tiny endpoint module using the registered Cardinal spec.

    :param endpoint_loss: Endpoint objective selection.
    :param parameterization: Field output parameterization.
    :param param_spec: Registered ParamSpec name, or ``None``.
    :param rectified_sigma_min: Residual source-noise scale.
    :param row: Constant field output when no explicit field is supplied.
    :param audio_loss: Optional recording audio loss.
    :param vector_field: Explicit field override.
    :returns: Configured flow module.
    """
    prediction = torch.zeros(_WIDTH) if row is None else row
    return VSTFlowMatchingModule(
        encoder=_WaveformEncoder(),
        vector_field=vector_field or _ConstantField(prediction),
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_WIDTH,
        param_spec=param_spec,
        conditioning="audio",
        audio_loss=audio_loss,  # pyright: ignore[reportArgumentType]
        cfg_dropout_rate=0.0,
        compile=False,
        endpoint_loss=endpoint_loss,  # pyright: ignore[reportArgumentType]
        parameterization=parameterization,  # pyright: ignore[reportArgumentType]
        rectified_sigma_min=rectified_sigma_min,
    )


def _target() -> torch.Tensor:
    """Build two model-space rows with opposite Cardinal classes.

    :returns: Cardinal-width endpoint targets.
    """
    target = torch.zeros(_BATCH, _WIDTH)
    target[0, 8:10] = torch.tensor([1.0, -1.0])
    target[1, 8:10] = torch.tensor([-1.0, 1.0])
    return target


def _batch(target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the direct train-step batch contract.

    :param target: Clean endpoint rows.
    :returns: Batch with endpoint, noise, and audio tensors.
    """
    return {
        "params": target,
        "noise": torch.zeros_like(target),
        "audio": torch.zeros(target.shape[0], _SIGNAL_LENGTH),
    }


def test_module_endpoint_loss_defaults_to_flat_mse() -> None:
    """Omitting the opt-in retains coordinate-wise endpoint MSE."""
    module = _module(param_spec=None)
    target = torch.ones(_BATCH, _WIDTH)

    outputs = module._train_step(_batch(target))  # noqa: SLF001

    assert module.hparams["endpoint_loss"] == "mse"
    assert outputs.loss.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("endpoint_loss", "parameterization", "param_spec", "message"),
    [
        pytest.param("cross_entropy", "endpoint", "cardinal", "endpoint_loss", id="unknown"),
        pytest.param("mixed", "velocity", "cardinal", "parameterization", id="velocity"),
        pytest.param("mixed", "endpoint", None, "param_spec", id="missing-spec"),
    ],
)
def test_module_invalid_endpoint_loss_configuration_raises(
    endpoint_loss: str, parameterization: str, param_spec: str | None, message: str
) -> None:
    """Invalid or underspecified mixed objectives fail at construction.

    :param endpoint_loss: Objective selection under test.
    :param parameterization: Field parameterization under test.
    :param param_spec: Optional registered ParamSpec name.
    :param message: Expected error fragment.
    """
    with pytest.raises(ValueError, match=message):
        _module(
            endpoint_loss=endpoint_loss,
            parameterization=parameterization,
            param_spec=param_spec,
        )


@pytest.mark.parametrize("sigma", [-0.01, 1.0, 4.0])
def test_module_invalid_rectified_sigma_min_raises(sigma: float) -> None:
    """Residual source-noise scales outside ``[0, 1)`` are rejected.

    :param sigma: Invalid residual noise scale.
    """
    with pytest.raises(ValueError, match="rectified_sigma_min"):
        _module(rectified_sigma_min=sigma)


def test_mixed_endpoint_row_loss_averages_each_logical_parameter_once() -> None:
    """A two-column category and scalar numerical span each contribute one term."""
    spec = ParamSpec(
        [
            CategoricalParameter("mode", values=["a", "b"], encoding="onehot"),
            ContinuousParameter("level"),
        ],
        [],
    )
    prediction = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, -1.0, 0.0]])

    row_loss = mixed_endpoint_row_loss(prediction, target, spec)
    row_loss.backward()

    assert row_loss.shape == (1, 1)
    assert row_loss.item() == pytest.approx((math.log(2.0) + 4.0) / 2.0)
    torch.testing.assert_close(
        prediction.grad,
        torch.tensor([[-0.25, 0.25, 2.0]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_mixed_endpoint_row_loss_treats_onehot_discrete_literal_as_classification() -> None:
    """One-hot integer literals use CE while scalar ordered integers remain numerical."""
    spec = ParamSpec(
        [],
        [
            DiscreteLiteralParameter("switch", min=1, max=3, encoding="onehot"),
            DiscreteLiteralParameter("pitch", min=48, max=72),
        ],
    )
    prediction = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    target = torch.tensor([[-1.0, 1.0, -1.0, 0.0]])

    row_loss = mixed_endpoint_row_loss(prediction, target, spec)

    assert row_loss.item() == pytest.approx((math.log(3.0) + 1.0) / 2.0)


def test_train_step_mixed_loss_keeps_row_weights_paired_with_rows() -> None:
    """A ``(batch, 1)`` weight scales only the matching row objective."""
    field = _ConstantField(torch.zeros(_WIDTH))
    module = _module(endpoint_loss="mixed", vector_field=field)
    with torch.no_grad():
        field.row[0] = 1.0
    target = _target()
    target[1, 0] = 3.0
    module._weight_time = lambda t: torch.tensor(  # pyright: ignore[reportAttributeAccessIssue]
        [[1.0], [3.0]], device=t.device
    )

    outputs = module._train_step(_batch(target))  # noqa: SLF001

    expected_first = (1.0 + math.log(2.0)) / 11.0
    expected_second = (4.0 + math.log(2.0)) / 11.0
    assert outputs.loss.item() == pytest.approx((expected_first + 3.0 * expected_second) / 2.0)
    torch.testing.assert_close(outputs.per_param_flow_mse[8:10], torch.full((2,), 2.0))
    torch.testing.assert_close(outputs.per_param_endpoint_mse[8:10], torch.ones(2))
    outputs.loss.backward()
    assert field.row.grad is not None
    assert torch.count_nonzero(field.row.grad[8:10]).item() == 2


def test_train_step_mse_keeps_row_weights_paired_with_rows() -> None:
    """Legacy MSE reduces to one row before applying its ``(batch, 1)`` weight."""
    module = _module(param_spec=None)
    module._weight_time = lambda t: torch.tensor(  # pyright: ignore[reportAttributeAccessIssue]
        [[1.0], [3.0]], device=t.device
    )
    target = torch.stack((torch.ones(_WIDTH), torch.full((_WIDTH,), 2.0)))

    outputs = module._train_step(_batch(target))  # noqa: SLF001

    assert outputs.loss.item() == pytest.approx(6.5)
    torch.testing.assert_close(outputs.per_param_flow_mse, torch.full((_WIDTH,), 6.5))
    torch.testing.assert_close(outputs.per_param_endpoint_mse, torch.full((_WIDTH,), 2.5))


def test_velocity_endpoint_diagnostic_nonzero_sigma_is_exact_for_perfect_field() -> None:
    """A perfect velocity has zero endpoint error on a sigma-bearing path."""
    module = _module(parameterization="velocity", param_spec=None, row=torch.full((_WIDTH,), 2.0))
    module.hparams["rectified_sigma_min"] = 0.25
    target = torch.ones(_BATCH, _WIDTH)
    batch = _batch(target)
    batch["noise"] = torch.full_like(target, -1.0)

    outputs = module._train_step(batch)  # noqa: SLF001

    torch.testing.assert_close(outputs.per_param_endpoint_mse, torch.zeros(_WIDTH))


def test_train_step_mse_and_mixed_share_unweighted_endpoint_diagnostic() -> None:
    """Endpoint diagnostics compare typed model-space predictions under both objectives."""
    target = _target()
    mse = _module(endpoint_loss="mse")._train_step(_batch(target))  # noqa: SLF001
    mixed = _module(endpoint_loss="mixed")._train_step(_batch(target))  # noqa: SLF001

    torch.testing.assert_close(mse.per_param_endpoint_mse, mixed.per_param_endpoint_mse)


def test_fixed_time_velocity_endpoint_mse_reports_ten_bins_and_equal_bin_mean() -> None:
    """Held-out diagnostics score fixed centers and average bins equally."""
    module = _module(parameterization="velocity", param_spec=None)
    target = torch.ones(_BATCH, _WIDTH)

    metrics = module._fixed_time_endpoint_mse(_batch(target))  # noqa: SLF001

    expected = torch.tensor(
        [0.9025, 0.7225, 0.5625, 0.4225, 0.3025, 0.2025, 0.1225, 0.0625, 0.0225, 0.0025]
    )
    actual = torch.stack(
        [metrics[f"velocity_endpoint_mse/t_{index:02d}"] for index in range(5, 100, 10)]
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        metrics["velocity_endpoint_mse/equal_bin_mean"], torch.tensor(0.3325)
    )


def test_fixed_time_endpoint_mse_uses_direct_endpoint_namespace() -> None:
    """Direct endpoint diagnostics report exact bins without a velocity prefix."""
    module = _module(parameterization="endpoint", param_spec=None)

    metrics = module._fixed_time_endpoint_mse(_batch(torch.ones(_BATCH, _WIDTH)))  # noqa: SLF001

    assert metrics.keys() == {
        *(f"endpoint_mse/t_{index:02d}" for index in range(5, 100, 10)),
        "endpoint_mse/equal_bin_mean",
    }
    torch.testing.assert_close(metrics["endpoint_mse/equal_bin_mean"], torch.tensor(1.0))


def test_endpoint_prediction_to_model_softmaxes_only_onehot_spans() -> None:
    """Categorical logits become ``2p - 1`` while ordered pitch and numerics stay unchanged."""
    prediction = torch.arange(_WIDTH, dtype=torch.float32).unsqueeze(0)
    prediction[0, 8:10] = torch.tensor([2.0, 3.0])

    endpoint = endpoint_prediction_to_model(prediction, CARDINAL_PARAM_SPEC)

    probabilities = torch.softmax(torch.tensor([2.0, 3.0]), dim=0)
    torch.testing.assert_close(endpoint[0, 8:10], 2 * probabilities - 1)
    torch.testing.assert_close(endpoint[0, :8], prediction[0, :8])
    torch.testing.assert_close(endpoint[0, 10:], prediction[0, 10:])


def test_mixed_endpoint_cfg_converts_guided_logits_to_valid_endpoint() -> None:
    """CFG logits convert to a valid model-space categorical endpoint."""
    module = _module(
        endpoint_loss="mixed",
        vector_field=_ConditionedLogitField(torch.zeros(_WIDTH)),
    )
    x = torch.zeros(1, _WIDTH)
    t = torch.full((1, 1), 0.5)
    conditioning = torch.ones(1, _CONDITIONING_DIM)

    velocity = module._velocity_field(conditioning, 2.0, None)(x, t)  # noqa: SLF001

    expected_endpoint = 2 * torch.softmax(torch.tensor([4.0, -2.0]), dim=0) - 1
    torch.testing.assert_close(velocity[0, 8:10], expected_endpoint / 0.5)
    assert torch.all(expected_endpoint.abs() <= 1.0)


def test_mixed_endpoint_multi_cfg_converts_guided_logits_once() -> None:
    """Three CFG branches combine as logits before one endpoint conversion."""
    module = _module(
        endpoint_loss="mixed",
        vector_field=_ConditionedControlLogitField(torch.zeros(_WIDTH)),
    )
    x = torch.zeros(1, _WIDTH)
    t = torch.full((1, 1), 0.5)
    conditioning = torch.ones(1, _CONDITIONING_DIM)
    control_tokens = ControlTokenBranches(
        conditional=torch.ones(1, 1, 1),
        unconditional=torch.zeros(1, 1, 1),
    )

    velocity = module._velocity_field(  # noqa: SLF001
        conditioning,
        2.0,
        control_tokens,
        sketch_cfg_strength=0.5,
    )(x, t)

    expected_endpoint = 2 * torch.softmax(torch.tensor([3.5, -2.5]), dim=0) - 1
    torch.testing.assert_close(velocity[0, 8:10], expected_endpoint / 0.5)


def test_sample_mixed_endpoint_finishes_on_typed_endpoint_without_sampling_classes() -> None:
    """The direct final step lands on deterministic probabilities in model space."""
    logits = torch.linspace(-0.5, 0.5, _WIDTH)
    logits[8:10] = torch.tensor([2.0, 3.0])
    module = _module(endpoint_loss="mixed", row=logits)
    noise = torch.randn(_BATCH, _WIDTH, generator=torch.Generator().manual_seed(0))

    sample = module._sample(None, noise, steps=4, cfg_strength=1.0)  # noqa: SLF001

    expected = endpoint_prediction_to_model(logits.expand(_BATCH, -1), CARDINAL_PARAM_SPEC)
    torch.testing.assert_close(sample, expected, atol=1e-5, rtol=0.0)


def test_audio_feedback_mixed_endpoint_receives_typed_endpoint() -> None:
    """The one-step audio estimate converts logits before rendering."""
    logits = torch.zeros(_WIDTH)
    logits[8:10] = torch.tensor([2.0, 3.0])
    recorder = _RecordingAudioLoss()
    module = _module(endpoint_loss="mixed", row=logits, audio_loss=recorder)

    module._train_step(_batch(_target()))  # noqa: SLF001

    assert recorder.endpoint is not None
    expected = endpoint_prediction_to_model(logits.expand(_BATCH, -1), CARDINAL_PARAM_SPEC)
    torch.testing.assert_close(recorder.endpoint, expected)


def _save_checkpoint(module: VSTFlowMatchingModule, path: Path, *, legacy: bool = False) -> Path:
    """Write a loadable checkpoint, optionally without endpoint-loss metadata.

    :param module: Module supplying state and hyperparameters.
    :param path: Checkpoint destination.
    :param legacy: Whether to remove endpoint-loss metadata.
    :returns: Checkpoint path.
    """
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    module.on_save_checkpoint(checkpoint)
    if legacy:
        checkpoint.pop("endpoint_loss")
        checkpoint["hyper_parameters"].pop("endpoint_loss", None)
    torch.save(checkpoint, path)
    return path


def test_load_checkpoint_with_other_endpoint_loss_raises(tmp_path: Path) -> None:
    """Same-shaped endpoint heads cannot cross raw-value and logit semantics.

    :param tmp_path: Checkpoint directory.
    """
    path = _save_checkpoint(_module(endpoint_loss="mixed"), tmp_path / "mixed.ckpt")

    with pytest.raises(ValueError, match="endpoint_loss"):
        VSTFlowMatchingModule.load_from_checkpoint(
            path,
            encoder=_WaveformEncoder(),
            endpoint_loss="mse",
            weights_only=False,
        )


def test_load_legacy_checkpoint_without_endpoint_loss_counts_as_mse(tmp_path: Path) -> None:
    """Checkpoints without endpoint-loss metadata default to the MSE objective.

    :param tmp_path: Checkpoint directory.
    """
    path = _save_checkpoint(_module(endpoint_loss="mse"), tmp_path / "legacy.ckpt", legacy=True)

    loaded = VSTFlowMatchingModule.load_from_checkpoint(
        path, encoder=_WaveformEncoder(), weights_only=False
    )

    assert loaded.hparams["endpoint_loss"] == "mse"


def test_vst_flow_config_defaults_endpoint_loss_to_mse() -> None:
    """The shipped flow config defaults endpoint_loss to MSE."""
    path = Path(__file__).parents[2] / "src/synth_setter/configs/model/vst_flow.yaml"

    config = OmegaConf.load(path)

    assert config.endpoint_loss == "mse"
