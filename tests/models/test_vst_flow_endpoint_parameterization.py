"""Endpoint parameterization of the flow module: predict ``x1`` instead of ``x1 - x0``."""

from __future__ import annotations

from pathlib import Path

import lightning
import pytest
import torch

from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule

_BATCH = 3
_WIDTH = 6
_SIGNAL_LENGTH = 16
_CONDITIONING_DIM = 4


class _WaveformEncoder(torch.nn.Module):
    """Minimal raw-audio conditioning encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_SIGNAL_LENGTH, _CONDITIONING_DIM)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Map a waveform batch to a flat conditioning vector.

        :param audio: Audio shaped ``(batch, _SIGNAL_LENGTH)``.
        :returns: Conditioning shaped ``(batch, _CONDITIONING_DIM)``.
        """
        return self.linear(audio)


class _ConstantField(torch.nn.Module):
    """Network whose prediction is one fixed row regardless of state, time, or conditioning."""

    def __init__(self, row: torch.Tensor) -> None:
        """Pin the row every forward pass returns.

        :param row: Prediction shaped ``(_WIDTH,)``.
        """
        super().__init__()
        self.row = torch.nn.Parameter(row.clone(), requires_grad=False)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Broadcast the fixed row over the batch.

        :param x: Parameter state shaped ``(batch, _WIDTH)``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param conditioning: Ignored.
        :returns: The fixed row repeated ``batch`` times.
        """
        return self.row.expand(x.shape[0], -1)

    def apply_dropout(
        self, z: torch.Tensor, rate: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep every conditioning row; this field ignores conditioning anyway.

        :param z: Conditioning rows.
        :param rate: Ignored.
        :returns: ``z`` unchanged and an all-True keep mask.
        """
        return z, torch.ones(z.shape[0], dtype=torch.bool, device=z.device)


class _DisplacementEndpointField(_ConstantField):
    """Endpoint predictor ``x_t + (1 - t) * row``: state- and time-dependent, velocity ``row``."""

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the endpoint the straight line through ``x`` at velocity ``row`` reaches.

        :param x: Parameter state shaped ``(batch, _WIDTH)``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param conditioning: Ignored.
        :returns: ``x + (1 - t) * row``.
        """
        return x + (1 - t) * self.row


class _RecordingAudioLoss(torch.nn.Module):
    """Audio term that keeps the estimate it was handed and contributes nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.theta_hat: torch.Tensor | None = None

    def forward(
        self,
        theta_hat: torch.Tensor,
        t: torch.Tensor,
        target_audio: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record ``theta_hat`` and return a zero scalar.

        :param theta_hat: One-step parameter estimate handed to the audio term.
        :param t: Flow time.
        :param target_audio: Target audio.
        :param keep: Optional keep mask.
        :returns: Zero scalar attached to the estimate's graph.
        """
        self.theta_hat = theta_hat.detach().clone()
        return theta_hat.sum() * 0.0


def _module(
    parameterization: str,
    field_row: torch.Tensor | None = None,
    audio_loss: torch.nn.Module | None = None,
    vector_field: torch.nn.Module | None = None,
) -> VSTFlowMatchingModule:
    """Build a tiny raw-audio flow module around a stub field.

    :param parameterization: ``"velocity"`` or ``"endpoint"``.
    :param field_row: Row the constant field always predicts; zeros when omitted.
    :param audio_loss: Optional audio term to attach.
    :param vector_field: Field replacing the constant one built from ``field_row``.
    :returns: Configured module with guidance dropout disabled.
    """
    row = torch.zeros(_WIDTH) if field_row is None else field_row
    return VSTFlowMatchingModule(
        encoder=_WaveformEncoder(),
        vector_field=vector_field or _ConstantField(row),
        optimizer=torch.optim.Adam,  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_WIDTH,
        conditioning="audio",
        audio_loss=audio_loss,  # pyright: ignore[reportArgumentType]
        cfg_dropout_rate=0.0,
        compile=False,
        parameterization=parameterization,  # pyright: ignore[reportArgumentType]
    )


def _batch(params_value: float, noise_value: float) -> dict[str, torch.Tensor]:
    """Build a batch whose params and noise are constant rows.

    :param params_value: Value filling every clean parameter coordinate.
    :param noise_value: Value filling every source-noise coordinate.
    :returns: Batch keyed as the flow module consumes it.
    """
    torch.manual_seed(0)
    return {
        "params": torch.full((_BATCH, _WIDTH), params_value),
        "noise": torch.full((_BATCH, _WIDTH), noise_value),
        "audio": torch.randn(_BATCH, _SIGNAL_LENGTH),
    }


@pytest.mark.parametrize(
    ("parameterization", "expected_loss"),
    [
        # Endpoint targets the clean row.
        pytest.param("endpoint", 4.0, id="endpoint-scores-x1"),
        # Velocity targets the source-to-clean displacement.
        pytest.param("velocity", 1.0, id="velocity-scores-x1-minus-x0"),
    ],
)
def test_train_step_zero_prediction_scores_against_the_parameterization_target(
    parameterization: str, expected_loss: float
) -> None:
    """The flow loss compares the prediction with ``x1`` or ``x1 - x0`` per the flag.

    :param parameterization: Flag under test.
    :param expected_loss: Hardcoded mean squared error a zero prediction incurs.
    """
    module = _module(parameterization)

    outputs = module._train_step(_batch(params_value=2.0, noise_value=1.0))  # noqa: SLF001

    assert outputs.loss.item() == pytest.approx(expected_loss)


def test_sample_endpoint_parameterization_lands_on_the_predicted_endpoint() -> None:
    """Integrating a constant endpoint prediction from any noise arrives at that endpoint."""
    endpoint = torch.tensor([0.5, -0.25, 1.0, 0.0, -1.0, 0.75])
    module = _module("endpoint", field_row=endpoint)
    torch.manual_seed(0)
    noise = torch.randn(_BATCH, _WIDTH)

    sample = module._sample(None, noise, steps=8, cfg_strength=1.0)  # noqa: SLF001

    torch.testing.assert_close(sample, endpoint.expand(_BATCH, -1), atol=1e-3, rtol=0.0)


def test_sample_endpoint_parameterization_integrates_a_state_dependent_field() -> None:
    """A field predicting ``x_t + (1 - t) * c`` is the velocity ``c``, so noise moves by ``c``.

    Unlike a constant endpoint, every RK4 stage sees a different state and time here, so a
    wrong intermediate conversion would change where the trajectory lands.
    """
    displacement = torch.tensor([0.5, -0.25, 1.0, 0.0, -1.0, 0.75])
    module = _module("endpoint", vector_field=_DisplacementEndpointField(displacement))
    torch.manual_seed(0)
    noise = torch.randn(_BATCH, _WIDTH)

    sample = module._sample(None, noise, steps=8, cfg_strength=1.0)  # noqa: SLF001

    torch.testing.assert_close(sample, noise + displacement, atol=1e-5, rtol=0.0)


def test_sample_velocity_parameterization_integrates_the_predicted_velocity() -> None:
    """Under the default flag a constant prediction is a velocity, so noise moves by it."""
    velocity = torch.tensor([0.5, -0.25, 1.0, 0.0, -1.0, 0.75])
    module = _module("velocity", field_row=velocity)
    torch.manual_seed(0)
    noise = torch.randn(_BATCH, _WIDTH)

    sample = module._sample(None, noise, steps=8, cfg_strength=1.0)  # noqa: SLF001

    torch.testing.assert_close(sample, noise + velocity, atol=1e-5, rtol=0.0)


def test_module_unknown_parameterization_raises() -> None:
    """An unknown flag is refused at construction instead of silently training velocity."""
    with pytest.raises(ValueError, match="parameterization"):
        _module("midpoint")


# Endpoint MSE the best row-independent predictor reaches on rows targeting +0.5, -0.5, 0.0:
# the variance of those targets. Beating it proves the field reads its per-row input.
_CONSTANT_PREDICTOR_FLOOR = 1 / 6


def test_train_step_endpoint_parameterization_fits_distinct_rows_on_one_fixed_batch() -> None:
    """Fitting one batch with distinct per-row targets beats any input-blind predictor."""
    from synth_setter.models.components.vector_field import VectorField

    torch.manual_seed(0)
    module = _module(
        "endpoint",
        vector_field=VectorField(
            field_dim=_WIDTH, hidden_dim=32, conditioning_dim=_CONDITIONING_DIM, num_blocks=2
        ),
    )
    batch = _batch(params_value=0.0, noise_value=0.0)
    batch["params"] = torch.tensor([0.5, -0.5, 0.0]).unsqueeze(1).expand(_BATCH, _WIDTH).clone()
    batch["noise"] = torch.randn(_BATCH, _WIDTH)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    initial = module._train_step(batch).loss.item()  # noqa: SLF001
    for _ in range(300):
        loss = module._train_step(batch).loss  # noqa: SLF001
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final = module._train_step(batch).loss.item()  # noqa: SLF001

    assert final < initial * 0.1
    assert final < _CONSTANT_PREDICTOR_FLOOR * 0.1


def test_audio_term_endpoint_parameterization_receives_the_raw_prediction() -> None:
    """The audio term scores the predicted endpoint itself, not ``x_t + (1 - t) * v``."""
    recorder = _RecordingAudioLoss()
    module = _module("endpoint", audio_loss=recorder)

    module._train_step(_batch(params_value=2.0, noise_value=1.0))  # noqa: SLF001

    assert recorder.theta_hat is not None
    torch.testing.assert_close(recorder.theta_hat, torch.zeros(_BATCH, _WIDTH))


def _save_checkpoint(module: VSTFlowMatchingModule, path: Path, *, legacy: bool) -> Path:
    """Persist a Lightning checkpoint, optionally omitting its parameterization stamp.

    :param module: Module whose weights and hyperparameters are written.
    :param path: Destination file.
    :param legacy: Whether to omit the parameterization stamp.
    :returns: ``path``.
    """
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    if not legacy:
        module.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, path)
    return path


def test_load_checkpoint_with_other_parameterization_raises(tmp_path: Path) -> None:
    """A velocity checkpoint refuses to load as an endpoint module rather than mis-sampling.

    :param tmp_path: Directory for the checkpoint.
    """
    path = _save_checkpoint(_module("velocity"), tmp_path / "velocity.ckpt", legacy=False)

    with pytest.raises(ValueError, match="parameterization"):
        VSTFlowMatchingModule.load_from_checkpoint(
            path, encoder=_WaveformEncoder(), parameterization="endpoint", weights_only=False
        )


def test_load_legacy_checkpoint_without_flag_counts_as_velocity(tmp_path: Path) -> None:
    """A checkpoint written before the flag existed loads as velocity and refuses endpoint.

    :param tmp_path: Directory for the checkpoint.
    """
    path = _save_checkpoint(_module("velocity"), tmp_path / "legacy.ckpt", legacy=True)

    loaded = VSTFlowMatchingModule.load_from_checkpoint(
        path, encoder=_WaveformEncoder(), weights_only=False
    )
    assert loaded.hparams["parameterization"] == "velocity"
    with pytest.raises(ValueError, match="parameterization"):
        VSTFlowMatchingModule.load_from_checkpoint(
            path, encoder=_WaveformEncoder(), parameterization="endpoint", weights_only=False
        )
