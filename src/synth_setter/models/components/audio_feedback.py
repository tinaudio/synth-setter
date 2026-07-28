"""Audio-domain feedback loss that backpropagates through a differentiable render.

The flow's one-step parameter estimate is rendered with autograd connected and scored
against the target audio in a frozen encoder's embedding space, so perceptual error reaches
the vector field's own weights. The term is gated to late flow times: below ``t_min`` the
estimate is still near noise and its render carries no usable signal.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
)

# Guards the gradient-ratio denominator when the flow loss contributes no gradient.
_GRAD_NORM_EPS = 1e-12


def validate_audio_feedback_runtime(*, drop_last: bool, compiled: bool, world_size: int) -> None:
    """Reject runtime configurations the differentiable renderer cannot serve.

    Each condition fails loudly rather than degrading silently — see
    https://github.com/tinaudio/synth-setter/issues/2585.

    :param drop_last: Whether the train loader drops a trailing partial batch.
    :param compiled: Whether the module is wrapped by ``torch.compile``.
    :param world_size: Number of distributed training processes.
    :raises ValueError: Any unsupported condition holds.
    """
    if not drop_last:
        # The renderer caches per (sample_rate, signal_length, batch, device) — #1820. A
        # trailing partial batch changes the batch dim and silently misses that cache.
        raise ValueError(
            "audio feedback requires drop_last=True on the train dataloader; the torchsynth "
            "renderer is cached per batch size and a partial final batch misses the cache "
            "(see https://github.com/tinaudio/synth-setter/issues/2585)"
        )
    if compiled:
        # torch.compile traces through functional_call into torchsynth's Voice, where it
        # graph-breaks or miscompiles; wrong gradients are worse than no run.
        raise ValueError(
            "audio feedback is incompatible with torch.compile; set model.compile=false "
            "(see https://github.com/tinaudio/synth-setter/issues/2585)"
        )
    if world_size > 1:
        # The render runs inside the loss on the training device behind a process-local
        # lock; distributed training is unvalidated and must not run silently.
        raise ValueError(
            f"audio feedback is single-device only, got world_size={world_size} "
            "(see https://github.com/tinaudio/synth-setter/issues/2585)"
        )


def gradient_balance(
    *, flow_loss: torch.Tensor, audio_term: torch.Tensor, shared: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure how two loss terms contribute gradient at a tensor they both reach.

    Loss magnitude is a poor proxy for gradient magnitude, so a weight tuned against loss curves
    does not transfer. EnCodec's balancer instead reads the weight as a fraction of total gradient,
    sampled at one shared intermediate tensor rather than at every parameter. The cosine is REPA's
    conflict diagnostic: it turns negative once the auxiliary term starts fighting the primary
    objective. Both gradients retain the graph, so the caller's own backward pass is unaffected.

    :param flow_loss: Scalar flow-matching loss.
    :param audio_term: Scalar weighted audio loss.
    :param shared: Tensor both terms backpropagate through.
    :returns: Audio-to-flow gradient-norm ratio, and the cosine between the two gradients.
    """
    (flow_grad,) = torch.autograd.grad(flow_loss, shared, retain_graph=True)
    (audio_grad,) = torch.autograd.grad(audio_term, shared, retain_graph=True)
    ratio = audio_grad.norm() / flow_grad.norm().clamp_min(_GRAD_NORM_EPS)
    cosine = functional.cosine_similarity(audio_grad.flatten(), flow_grad.flatten(), dim=0)
    return ratio, cosine


def _frozen_latent_distance(
    encoder: nn.Module, rendered: torch.Tensor, target_audio: torch.Tensor
) -> torch.Tensor:
    """Per-sample cosine distance in an encoder's embedding space, holding the space fixed.

    The encoder's parameters are detached via ``functional_call`` and it runs in eval mode,
    so the term cannot shrink by collapsing the (jointly trained) conditioning encoder or by
    drifting its BatchNorm running stats; gradient still flows to the rendered estimate
    through the frozen weights. Cosine rather than raw MSE because embedding norm carries no
    fixed meaning while the encoder trains, which would let the term's magnitude drift with
    activation scale alone.

    :param encoder: Encoder defining the latent space; left in its original mode.
    :param rendered: Rendered estimate shaped ``(batch, samples)``.
    :param target_audio: Observed audio, same shape.
    :returns: Per-sample distance in ``[0, 2]`` shaped ``(batch,)``.
    """
    frozen_state = {
        name: tensor.detach()
        for name, tensor in (*encoder.named_parameters(), *encoder.named_buffers())
    }
    was_training = encoder.training
    encoder.eval()
    try:
        # flatten(1) collapses token/layer axes so sequence encoders also reduce to a
        # per-sample scalar instead of broadcasting against the (batch,) weight.
        embeddings = (
            torch.func.functional_call(encoder, frozen_state, (signal,)).flatten(start_dim=1)
            for signal in (rendered, target_audio)
        )
        pred_embedding, target_embedding = embeddings
        return 1.0 - functional.cosine_similarity(pred_embedding, target_embedding, dim=-1)
    finally:
        encoder.train(was_training)


class AudioFeedbackLoss(nn.Module):
    """Weighted latent-space audio distance on the flow's rendered one-step estimate."""

    def __init__(
        self,
        *,
        lambda_audio: float,
        t_min: float,
        sample_rate: int,
        signal_length: int,
        midi_pitch: int,
    ) -> None:
        """Configure the render geometry and the term's weighting.

        :param lambda_audio: Audio-term weight at t=1; must be positive.
        :param t_min: Flow time at which the term switches on, in ``[0, 1)``.
        :param sample_rate: Render sample rate in Hz.
        :param signal_length: Rendered samples per row.
        :param midi_pitch: Fixed MIDI note rendered for every row.
        :raises ValueError: Non-positive ``lambda_audio`` or out-of-range ``t_min``.
        """
        super().__init__()
        if lambda_audio <= 0.0:
            raise ValueError(
                f"lambda_audio must be positive, got {lambda_audio}; omit the audio loss "
                "entirely for the no-render control arm"
            )
        if not 0.0 <= t_min < 1.0:
            raise ValueError(f"t_min must lie in [0, 1), got {t_min}")
        self.lambda_audio = lambda_audio
        self.t_min = t_min
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.midi_pitch = midi_pitch

    def audio_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Ramp the weight from zero at ``t_min`` to ``lambda_audio`` at t=1.

        :param t: Flow time shaped ``(batch, 1)``.
        :returns: Per-sample weight shaped ``(batch, 1)``.
        """
        return self.lambda_audio * ((t - self.t_min) / (1 - self.t_min)).clamp(min=0.0)

    def forward(
        self,
        theta_hat: torch.Tensor,
        t: torch.Tensor,
        target_audio: torch.Tensor,
        encoder: nn.Module,
        keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render the estimate and return the weighted latent distance to the target.

        :param theta_hat: One-step parameter estimate in model space ``[-1, 1]``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param target_audio: Observed audio shaped ``(batch, signal_length)``.
        :param encoder: Encoder defining the latent space the distance is measured in.
        :param keep: Optional per-row mask shaped ``(batch,)``; rows at ``False`` are
            zero-weighted (CFG-dropped rows must not train the unconditional branch on
            row-specific targets).
        :returns: Scalar weighted audio loss.
        """
        rendered = render_torchsynth_grad(
            differentiable_decode(theta_hat),
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            midi_pitch=self.midi_pitch,
        )
        # The stored target was hard-clamped by render_torchsynth; a straight-through
        # clamp matches that contract without zeroing gradient on clipped samples.
        rendered = rendered + (rendered.clamp(-1.0, 1.0) - rendered).detach()
        distance = _frozen_latent_distance(encoder, rendered, target_audio)
        weight = self.audio_weight(t).squeeze(-1)
        if keep is not None:
            weight = weight * keep
        return (weight * distance).mean()
