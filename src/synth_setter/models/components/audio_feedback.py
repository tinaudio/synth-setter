"""Audio-domain feedback loss that backpropagates through a differentiable render.

The flow's one-step parameter estimate is scored against target audio in a frozen encoder's
embedding space. The term is gated to late flow times, where the estimate carries usable signal.

Typical usage:
    audio_term = AudioFeedbackLoss(**audio_loss_config)(
        theta_hat, t, target_audio, encoder
    )
"""

import math
from collections.abc import Mapping

import torch
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch import Tensor, nn
from torch.nn import functional

from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
    validate_torchsynth_params,
)

# Guards the gradient-ratio denominator when the flow loss contributes no gradient.
_GRAD_NORM_EPS = 1e-12
_ANY_SHAPE = "..."
_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_PARAMS_SHAPE = "batch params"
_BATCH_SHAPE = "batch"
_BATCH_TIME_SHAPE = "batch 1"
_SCALAR_SHAPE = ""


@jaxtyped(typechecker=beartype)
def _drop_last_metadata(train_dataloader: object) -> tuple[bool | None, ...]:
    """Collect drop-last metadata from every loader leaf.

    :param train_dataloader: Plain or nested loader structure to inspect.
    :returns: One explicit boolean or indeterminate marker per leaf.
    """
    if isinstance(train_dataloader, CombinedLoader):
        return _drop_last_metadata(train_dataloader.iterables)
    if isinstance(train_dataloader, Mapping):
        return tuple(
            status
            for loader in train_dataloader.values()
            for status in _drop_last_metadata(loader)
        )
    if isinstance(train_dataloader, (list, tuple)):
        return tuple(
            status for loader in train_dataloader for status in _drop_last_metadata(loader)
        )
    drop_last = getattr(train_dataloader, "drop_last", None)
    return (drop_last if isinstance(drop_last, bool) else None,)


@jaxtyped(typechecker=beartype)
def validate_audio_feedback_runtime(
    *, train_dataloader: object | None = None, compiled: bool, world_size: int
) -> None:
    """Reject runtime configurations the differentiable renderer cannot serve.

    Each condition fails loudly rather than degrading silently — see
    https://github.com/tinaudio/synth-setter/issues/2585.

    :param train_dataloader: Trainer loader metadata, or ``None`` before trainer attachment.
    :param compiled: Whether the module is wrapped by ``torch.compile``.
    :param world_size: Number of distributed training processes.
    :raises ValueError: Any unsupported condition holds.
    """
    if train_dataloader is not None:
        drop_last_metadata = _drop_last_metadata(train_dataloader)
        if False in drop_last_metadata:
            # The renderer caches per (sample_rate, signal_length, batch, device) — #1820. A
            # trailing partial batch changes the batch dim and silently misses that cache.
            raise ValueError(
                "audio feedback requires drop_last=True on every train dataloader leaf; found "
                "drop_last=False (see https://github.com/tinaudio/synth-setter/issues/2585)"
            )
        if not drop_last_metadata or None in drop_last_metadata:
            raise ValueError(
                "audio feedback could not determine drop_last for every train dataloader leaf"
            )
    if compiled:
        # torch.compile traces through functional_call into torchsynth's Voice, where it
        # graph-breaks or miscompiles; wrong gradients are worse than no run.
        raise ValueError(
            "audio feedback is incompatible with torch.compile; set model.compile=false "
            "(see https://github.com/tinaudio/synth-setter/issues/2585)"
        )
    if world_size > 1:
        # The render mutates one cached Voice per process, so multi-rank behaviour is
        # unvalidated — https://github.com/tinaudio/synth-setter/issues/2659.
        raise ValueError(
            f"audio feedback is single-device only, got world_size={world_size} "
            "(see https://github.com/tinaudio/synth-setter/issues/2585)"
        )


@jaxtyped(typechecker=beartype)
def gradient_balance(
    *,
    flow_loss: Float[Tensor, _SCALAR_SHAPE],
    audio_term: Float[Tensor, _SCALAR_SHAPE],
    shared: Float[Tensor, _ANY_SHAPE],
) -> tuple[Float[Tensor, _SCALAR_SHAPE], Float[Tensor, _SCALAR_SHAPE]]:
    """Measure how two loss terms contribute gradient at a tensor they both reach.

    Loss magnitude is a poor proxy for gradient magnitude, so tune ``lambda_audio`` against the
    ratio and read the cosine as a conflict signal — rationale and citations in
    https://github.com/tinaudio/synth-setter/issues/2628#issuecomment-5111367681. Both gradients
    retain the graph, so the caller's own backward pass is unaffected.

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


@jaxtyped(typechecker=beartype)
def _frozen_latent_distance(
    encoder: nn.Module,
    rendered: Float[Tensor, _BATCH_AUDIO_SHAPE],
    target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
) -> Float[Tensor, _BATCH_SHAPE]:
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

    @jaxtyped(typechecker=beartype)
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

        :param lambda_audio: Finite positive audio-term weight at t=1.
        :param t_min: Flow time at which the term switches on, in ``[0, 1)``.
        :param sample_rate: Render sample rate in Hz.
        :param signal_length: Rendered samples per row.
        :param midi_pitch: Fixed MIDI note rendered for every row.
        :raises ValueError: Non-finite/non-positive ``lambda_audio`` or out-of-range ``t_min``.
        """
        super().__init__()
        if not math.isfinite(lambda_audio) or lambda_audio <= 0.0:
            raise ValueError(
                f"lambda_audio must be finite and positive, got {lambda_audio}; omit the audio loss "
                "entirely for the no-render control arm"
            )
        if not 0.0 <= t_min < 1.0:
            raise ValueError(f"t_min must lie in [0, 1), got {t_min}")
        self.lambda_audio = lambda_audio
        self.t_min = t_min
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.midi_pitch = midi_pitch

    @jaxtyped(typechecker=beartype)
    def audio_weight(
        self, t: Float[Tensor, _BATCH_TIME_SHAPE]
    ) -> Float[Tensor, _BATCH_TIME_SHAPE]:
        """Ramp the weight from zero at ``t_min`` to ``lambda_audio`` at t=1.

        :param t: Flow time shaped ``(batch, 1)``.
        :returns: Per-sample weight shaped ``(batch, 1)``.
        """
        return self.lambda_audio * ((t - self.t_min) / (1 - self.t_min)).clamp(min=0.0)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        theta_hat: Float[Tensor, _BATCH_PARAMS_SHAPE],
        t: Float[Tensor, _BATCH_TIME_SHAPE],
        target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
        encoder: nn.Module,
        keep: Shaped[Tensor, _BATCH_SHAPE] | None = None,
    ) -> Float[Tensor, _SCALAR_SHAPE]:
        """Render the estimate and return the weighted latent distance to the target.

        :param theta_hat: One-step parameter estimate in model space ``[-1, 1]``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param target_audio: Observed audio shaped ``(batch, signal_length)``.
        :param encoder: Encoder defining the latent space the distance is measured in.
        :param keep: Optional CFG keep mask shaped ``(batch,)``; rows at ``False`` are
            zero-weighted because their estimate is drawn from the marginal, making the
            residual against that row's own audio near-arbitrary.
        :returns: Scalar weighted audio loss.
        """
        params = differentiable_decode(theta_hat)
        validate_torchsynth_params(params)
        weight = self.audio_weight(t).squeeze(-1)
        if keep is not None:
            weight = weight * keep
        if torch.count_nonzero(weight).item() == 0:
            return theta_hat.sum() * 0.0

        rendered = render_torchsynth_grad(
            params,
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            midi_pitch=self.midi_pitch,
        )
        # The stored target was hard-clamped by render_torchsynth; a straight-through
        # clamp matches that contract without zeroing gradient on clipped samples.
        rendered = rendered + (rendered.clamp(-1.0, 1.0) - rendered).detach()
        distance = _frozen_latent_distance(encoder, rendered, target_audio)
        return (weight * distance).mean()
