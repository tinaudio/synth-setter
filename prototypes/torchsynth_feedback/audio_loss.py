"""Audio-loss finetuning of the torchsynth flow.

Stage A pretrains the flow with the conditional flow-matching loss
(:mod:`step_b_pretrain`); this module adds a differentiable-render term on the
one-step estimate so spectral error backpropagates into the flow's own weights.
Contrast the #2553 spike, where the render gradient was detached and fed to a
separate control field as a conditioning feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prototypes.torchsynth_feedback.flow import (
    MIDI_PITCH,
    SAMPLE_RATE,
    SIGNAL_LENGTH,
    SpectrumEncoder,
    sample_batch,
)
from prototypes.torchsynth_feedback.grad_render import (
    multi_scale_log_mel_distance,
    render_torchsynth_grad,
)
from synth_setter.data.torchsynth_datamodule import _PARAM_CLAMP_EPS
from synth_setter.models.components.vector_field import VectorField

GRAD_CLIP_NORM = 1.0


@dataclass
class FlowBatch:
    """One conditional-flow-matching training tuple.

    .. attribute :: params

       Target params in model space ``[-1, 1]``, shaped ``(batch, NUM_PARAMS)``.

    .. attribute :: target_audio

       Audio rendered from ``params``, shaped ``(batch, SIGNAL_LENGTH)``.

    .. attribute :: x0

       Noise endpoint of the probability path, shaped like ``params``.

    .. attribute :: t

       Flow time shaped ``(batch, 1)``.
    """

    params: torch.Tensor
    target_audio: torch.Tensor
    x0: torch.Tensor
    t: torch.Tensor


@dataclass
class AudioLossConfig:
    """Weighting of the render term against the flow-matching term.

    .. attribute :: lambda_audio

       Audio-loss weight at t=1. Zero skips the render entirely, giving the
       flow-matching-only ablation arm.

    .. attribute :: t_min

       Flow time at which the audio term switches on. Below it the one-step
       estimate is still near noise, so its render carries no usable signal.
    """

    lambda_audio: float = 1.0
    t_min: float = 0.8


@dataclass
class FinetuneConfig:
    """Stage-B optimization settings.

    .. attribute :: steps

       Optimizer steps; each one renders the batch twice (target and estimate).

    .. attribute :: batch_size

       Rows per step; fixed for the whole loop (renderer cache, #1820).

    .. attribute :: learning_rate

       AdamW learning rate for the flow.

    .. attribute :: lambda_audio

       Audio-loss weight at t=1; see :class:`AudioLossConfig`.

    .. attribute :: t_min

       Flow time at which the audio term switches on, and the lower bound of
       the finetuning time window.

    .. attribute :: seed

       RNG seed for the online data stream.
    """

    steps: int = 6_000
    batch_size: int = 128
    learning_rate: float = 3e-4
    lambda_audio: float = 1.0
    t_min: float = 0.8
    seed: int = 7


def differentiable_decode(theta: torch.Tensor) -> torch.Tensor:
    """Map model space ``[-1, 1]`` to renderable ``[eps, 1 - eps]``, keeping gradient.

    The forward value is the clamp the renderer needs; the backward pass sees
    the identity, so saturated entries still receive gradient and the audio
    loss can pull them back into range. A plain ``clamp`` zeroes them instead.

    :param theta: Params in model space shaped ``(batch, NUM_PARAMS)``.
    :returns: Params in torchsynth space, strictly inside ``(0, 1)``.
    """
    params01 = (theta + 1) / 2
    clamped = params01.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS)
    return params01 + (clamped - params01).detach()


def audio_weight(t: torch.Tensor, lambda_audio: float, t_min: float) -> torch.Tensor:
    """Ramp the audio-loss weight from zero at ``t_min`` to ``lambda_audio`` at t=1.

    :param t: Flow time shaped ``(batch, 1)``.
    :param lambda_audio: Weight at t=1.
    :param t_min: Flow time at which the ramp starts.
    :returns: Per-sample weight shaped ``(batch, 1)``.
    :raises ValueError: ``t_min`` outside ``[0, 1)``.
    """
    if not 0.0 <= t_min < 1.0:
        raise ValueError(f"t_min must lie in [0, 1), got {t_min}")
    return lambda_audio * ((t - t_min) / (1 - t_min)).clamp(min=0.0)


def combined_loss(
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    batch: FlowBatch,
    config: AudioLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Flow-matching loss plus a weighted spectral distance on the rendered estimate.

    :param encoder: Conditioning encoder (frozen by the caller during stage B).
    :param vector_field: Flow being trained.
    :param batch: Training tuple to score.
    :param config: Audio-term weighting.
    :returns: Scalar loss and its per-term values for logging.
    """
    t = batch.t
    x_t = batch.x0 * (1 - t) + batch.params * t
    prediction = vector_field(x_t, t, encoder(batch.target_audio))
    cfm_loss = (prediction - (batch.params - batch.x0)).square().mean()
    if config.lambda_audio == 0.0:
        return cfm_loss, {"loss": cfm_loss.item(), "cfm_loss": cfm_loss.item(), "audio_loss": 0.0}

    theta_hat = x_t + (1 - t) * prediction
    rendered = render_torchsynth_grad(
        differentiable_decode(theta_hat),
        sample_rate=SAMPLE_RATE,
        signal_length=SIGNAL_LENGTH,
        midi_pitch=MIDI_PITCH,
    )
    distance = multi_scale_log_mel_distance(rendered, batch.target_audio, SAMPLE_RATE)
    audio_loss = (audio_weight(t, config.lambda_audio, config.t_min).squeeze(-1) * distance).mean()
    loss = cfm_loss + audio_loss
    return loss, {
        "loss": loss.item(),
        "cfm_loss": cfm_loss.item(),
        "audio_loss": audio_loss.item(),
    }


def per_param_grad_norms(theta: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
    """Report each synth parameter's audio-gradient magnitude across the batch.

    Diagnostic for the known spread in raw render gradients: a parameter whose
    norm is orders below the rest is effectively untrained by the audio term.

    :param theta: Params in model space shaped ``(batch, NUM_PARAMS)``.
    :param target_audio: Observed audio shaped ``(batch, SIGNAL_LENGTH)``.
    :returns: Per-parameter gradient norm shaped ``(NUM_PARAMS,)``.
    """
    leaf = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        rendered = render_torchsynth_grad(
            differentiable_decode(leaf),
            sample_rate=SAMPLE_RATE,
            signal_length=SIGNAL_LENGTH,
            midi_pitch=MIDI_PITCH,
        )
        distance = multi_scale_log_mel_distance(rendered, target_audio, SAMPLE_RATE)
        (grad,) = torch.autograd.grad(distance.sum(), leaf)
    return grad.norm(dim=0)


def finetune_audio_loss(
    encoder: SpectrumEncoder,
    vector_field: VectorField,
    device: str,
    config: FinetuneConfig,
) -> list[dict[str, float]]:
    """Train the flow with the combined loss, holding the encoder frozen.

    :param encoder: Pretrained conditioning encoder; frozen in place.
    :param vector_field: Pretrained flow, updated in place.
    :param device: Torch device string.
    :param config: Optimization settings.
    :returns: One loss record per step, in step order.
    """
    encoder.eval()
    encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(vector_field.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    loss_config = AudioLossConfig(lambda_audio=config.lambda_audio, t_min=config.t_min)
    history: list[dict[str, float]] = []
    for _ in range(config.steps):
        params, target_audio = sample_batch(config.batch_size, device, generator)
        # Train only inside the feedback window; below t_min the estimate is still noise.
        t = config.t_min + (1 - config.t_min) * torch.rand(config.batch_size, 1, device=device)
        batch = FlowBatch(params, target_audio, torch.randn_like(params), t)
        loss, metrics = combined_loss(encoder, vector_field, batch, loss_config)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(vector_field.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        history.append(metrics)
    return history


__all__ = [
    "AudioLossConfig",
    "FinetuneConfig",
    "FlowBatch",
    "audio_weight",
    "combined_loss",
    "differentiable_decode",
    "finetune_audio_loss",
    "per_param_grad_norms",
]
