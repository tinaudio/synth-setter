"""Behaviour tests for the torchsynth audio-feedback loss and its runtime guards."""

import pytest
import torch

from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
)
from synth_setter.models.components.audio_feedback import (
    AudioFeedbackLoss,
    gradient_balance,
    validate_audio_feedback_runtime,
)

_SAMPLE_RATE = 44_100
_SIGNAL_LENGTH = 4_410
_MIDI_PITCH = 60
_BATCH = 4
_NUM_PARAMS = 76


def _render(params01: torch.Tensor) -> torch.Tensor:
    """Render a parameter batch without gradients.

    :param params01: Parameters in ``[0, 1]``.
    :returns: Rendered audio.
    """
    with torch.no_grad():
        return render_torchsynth_grad(
            params01,
            sample_rate=_SAMPLE_RATE,
            signal_length=_SIGNAL_LENGTH,
            midi_pitch=_MIDI_PITCH,
        )


def _loss(**kwargs: object) -> AudioFeedbackLoss:
    """Build an audio-feedback loss with test-sized render settings.

    :param **kwargs: Overrides forwarded to :class:`AudioFeedbackLoss`.
    :returns: Configured loss module.
    """
    settings = {
        "lambda_audio": 1.0,
        "t_min": 0.8,
        "sample_rate": _SAMPLE_RATE,
        "signal_length": _SIGNAL_LENGTH,
        "midi_pitch": _MIDI_PITCH,
    }
    return AudioFeedbackLoss(**(settings | kwargs))


def _linear_encoder(scale: float = 1.0) -> torch.nn.Module:
    """Build a deterministic encoder whose output magnitude is set by ``scale``.

    :param scale: Multiplier applied to every weight and bias.
    :returns: Encoder mapping ``(batch, signal_length)`` to ``(batch, 8)``.
    """
    torch.manual_seed(0)
    encoder = torch.nn.Linear(_SIGNAL_LENGTH, 8)
    with torch.no_grad():
        encoder.weight.mul_(scale)
        encoder.bias.mul_(scale)
    return encoder


def test_differentiable_decode_in_range_input_matches_affine_map() -> None:
    """An in-range model-space value maps to ``(theta + 1) / 2``."""
    decoded = differentiable_decode(torch.tensor([[-0.5, 0.0, 0.5]]))
    assert torch.allclose(decoded, torch.tensor([[0.25, 0.5, 0.75]]))


def test_differentiable_decode_saturated_input_keeps_gradient() -> None:
    """Saturated entries keep gradient so the loss can pull them back into range."""
    theta = torch.tensor([[-4.0, 4.0]], requires_grad=True)
    (gradient,) = torch.autograd.grad(differentiable_decode(theta).sum(), theta)
    assert torch.allclose(gradient, torch.tensor([[0.5, 0.5]]))


def test_differentiable_decode_saturated_input_clamps_forward_value() -> None:
    """The rendered value stays strictly inside the renderer's open interval."""
    decoded = differentiable_decode(torch.tensor([[-4.0, 4.0]]))
    assert (decoded > 0.0).all()
    assert (decoded < 1.0).all()


def test_audio_weight_below_t_min_is_zero() -> None:
    """The audio term is inactive before the feedback window opens."""
    weight = _loss().audio_weight(torch.tensor([[0.0], [0.5], [0.79]]))
    assert torch.all(weight == 0.0)


def test_audio_weight_at_final_time_equals_lambda() -> None:
    """The ramp reaches the configured weight at t=1."""
    weight = _loss(lambda_audio=0.25).audio_weight(torch.tensor([[1.0]]))
    assert torch.allclose(weight, torch.tensor([[0.25]]))


def test_grad_render_output_matches_the_documented_audio_contract() -> None:
    """The differentiable render emits finite float32 ``(batch, signal_length)`` audio in range."""
    audio = _render(torch.rand(_BATCH, _NUM_PARAMS, generator=torch.Generator().manual_seed(0)))

    assert audio.shape == (_BATCH, _SIGNAL_LENGTH)
    assert audio.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert audio.abs().max() <= 1.0


def test_latent_loss_backprops_gradient_through_the_encoder() -> None:
    """The latent distance differentiates through both the render and the encoder."""
    torch.manual_seed(0)
    target_audio = _render(torch.rand(_BATCH, _NUM_PARAMS))
    theta = (torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1).requires_grad_(True)

    value = _loss().forward(
        theta, torch.full((_BATCH, 1), 0.9), target_audio, encoder=_linear_encoder()
    )
    (gradient,) = torch.autograd.grad(value, theta)

    assert value.item() > 0.0
    assert torch.isfinite(gradient).all()
    assert (gradient != 0).any()


def test_latent_loss_is_invariant_to_encoder_output_scale() -> None:
    """Cosine geometry: scaling the encoder's outputs must not move the distance."""
    torch.manual_seed(0)
    params = torch.rand(_BATCH, _NUM_PARAMS)
    target_audio = _render(params)
    theta = torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1
    t = torch.full((_BATCH, 1), 0.9)

    unscaled = _loss().forward(theta, t, target_audio, encoder=_linear_encoder(scale=1.0))
    scaled = _loss().forward(theta, t, target_audio, encoder=_linear_encoder(scale=10.0))

    assert torch.allclose(unscaled, scaled, atol=1e-5)


def test_latent_loss_of_a_perfect_estimate_is_zero() -> None:
    """An estimate that decodes to the target's own parameters renders an exact match."""
    torch.manual_seed(0)
    params = torch.rand(_BATCH, _NUM_PARAMS).clamp(0.01, 0.99)
    target_audio = _render(params)

    value = _loss().forward(
        params * 2 - 1, torch.full((_BATCH, 1), 0.9), target_audio, encoder=_linear_encoder()
    )

    assert value.item() == pytest.approx(0.0, abs=1e-6)


def test_latent_loss_with_a_sequence_encoder_reduces_to_a_scalar() -> None:
    """Token-emitting encoders must still yield one distance per sample, not per token."""

    class _TokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(_SIGNAL_LENGTH, 24)

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            return self.linear(audio).reshape(audio.shape[0], 3, 8)

    torch.manual_seed(0)
    target_audio = _render(torch.rand(_BATCH, _NUM_PARAMS))
    theta = torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1

    value = _loss().forward(
        theta, torch.full((_BATCH, 1), 0.9), target_audio, encoder=_TokenEncoder()
    )

    assert value.ndim == 0
    assert torch.isfinite(value)


def test_latent_loss_leaves_encoder_weights_and_stats_untouched() -> None:
    """The latent space is frozen: no weight gradients, no BatchNorm stat drift."""
    torch.manual_seed(0)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(_SIGNAL_LENGTH, 8), torch.nn.BatchNorm1d(8), torch.nn.GELU()
    )
    encoder.train()
    batch_norm = encoder[1]
    assert isinstance(batch_norm, torch.nn.BatchNorm1d)
    assert batch_norm.running_mean is not None
    stats_before = batch_norm.running_mean.clone()
    target_audio = _render(torch.rand(_BATCH, _NUM_PARAMS))
    theta = (torch.rand(_BATCH, _NUM_PARAMS) * 2 - 1).requires_grad_(True)

    value = _loss().forward(theta, torch.full((_BATCH, 1), 0.9), target_audio, encoder=encoder)
    value.backward()

    assert theta.grad is not None and (theta.grad != 0).any()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert torch.equal(batch_norm.running_mean, stats_before)
    assert encoder.training


def test_zero_lambda_is_rejected_rather_than_silently_rendering() -> None:
    """A zero weight means the control arm, which must not pay for renders."""
    with pytest.raises(ValueError, match="lambda_audio"):
        _loss(lambda_audio=0.0)


def test_gradient_balance_of_a_term_against_itself_is_unit_ratio_and_alignment() -> None:
    """Identical terms contribute identically: ratio 1, cosine 1."""
    shared = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    term = shared.square().sum()

    ratio, cosine = gradient_balance(flow_loss=term, audio_term=term, shared=shared)

    assert ratio.item() == pytest.approx(1.0)
    assert cosine.item() == pytest.approx(1.0)


def test_gradient_balance_scaled_audio_term_scales_the_ratio_only() -> None:
    """The ratio tracks relative gradient magnitude; the cosine ignores it."""
    shared = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    flow = shared.square().sum()

    ratio, cosine = gradient_balance(flow_loss=flow, audio_term=3.0 * flow, shared=shared)

    assert ratio.item() == pytest.approx(3.0)
    assert cosine.item() == pytest.approx(1.0)


def test_gradient_balance_opposed_terms_reports_negative_cosine() -> None:
    """A term pulling against the flow loss is the conflict signal we want to see."""
    shared = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    flow = shared.square().sum()

    ratio, cosine = gradient_balance(flow_loss=flow, audio_term=-flow, shared=shared)

    assert ratio.item() == pytest.approx(1.0)
    assert cosine.item() == pytest.approx(-1.0)


def test_gradient_balance_with_a_detached_flow_loss_is_finite() -> None:
    """A zero flow gradient must not divide by zero."""
    shared = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)

    ratio, cosine = gradient_balance(
        flow_loss=(shared.detach() * shared.detach()).sum() + 0.0 * shared.sum(),
        audio_term=shared.square().sum(),
        shared=shared,
    )

    assert torch.isfinite(ratio)
    assert torch.isfinite(cosine)


def test_validate_runtime_without_drop_last_raises() -> None:
    """A trailing partial batch would silently miss the renderer's batch-keyed cache."""
    with pytest.raises(ValueError, match="drop_last"):
        validate_audio_feedback_runtime(drop_last=False, compiled=False, world_size=1)


def test_validate_runtime_with_torch_compile_raises() -> None:
    """Compiling over the functional_call render graph-breaks or miscompiles."""
    with pytest.raises(ValueError, match="compile"):
        validate_audio_feedback_runtime(drop_last=True, compiled=True, world_size=1)


def test_validate_runtime_with_multiple_devices_raises() -> None:
    """The renderer is process-local and single-device only."""
    with pytest.raises(ValueError, match="world_size"):
        validate_audio_feedback_runtime(drop_last=True, compiled=False, world_size=2)


def test_validate_runtime_accepts_a_supported_configuration() -> None:
    """A single-device, uncompiled, drop-last run is the supported configuration."""
    validate_audio_feedback_runtime(drop_last=True, compiled=False, world_size=1)
