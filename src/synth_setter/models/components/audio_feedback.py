"""Audio-domain feedback loss that backpropagates through a differentiable render.

The flow's one-step parameter estimate is scored against target audio in a frozen encoder's
embedding space. Gradient reaching the network scales as ``(t - t_min) * (1 - t)``: zero at
``t_min``, zero again at t=1 where the estimate is trivially correct, peaking midway — see
https://github.com/tinaudio/synth-setter/issues/2665.

Typical usage:
    audio_term = AudioFeedbackLoss(**audio_loss_config)(
        theta_hat, t, target_audio, encoder
    )
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor, nn
from torch.nn import functional

from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
    validate_torchsynth_params,
)

# Guards the gradient-ratio denominator when the flow loss contributes no gradient.
_GRAD_NORM_EPS = 1e-12
_TIME_BUCKETS = 4
_BATCH_ANY_SHAPE = "batch ..."
_BUCKETS_SHAPE = "buckets"
_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_PARAMS_SHAPE = "batch params"
_BATCH_SHAPE = "batch"
_BATCH_TIME_SHAPE = "batch 1"
_SCALAR_SHAPE = ""

type EmbeddingTap = Callable[[Float[Tensor, _BATCH_AUDIO_SHAPE]], Float[Tensor, _BATCH_ANY_SHAPE]]


@jaxtyped(typechecker=beartype)
def validate_audio_feedback_runtime(*, compiled: bool, world_size: int) -> None:
    """Reject runtime configurations the differentiable renderer cannot serve.

    Each condition fails loudly rather than degrading silently — see
    https://github.com/tinaudio/synth-setter/issues/2585.

    :param compiled: Whether the module is wrapped by ``torch.compile``.
    :param world_size: Number of distributed training processes.
    :raises ValueError: Any unsupported condition holds.
    """
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


@dataclass(frozen=True)
class GradientBalance:
    """How the audio term's gradient compares to the flow loss's at a shared tensor.

    .. attribute :: ratio

       Audio-to-flow gradient-norm ratio; the scale ``lambda_audio`` is tuned against.

    .. attribute :: cosine

       Cosine between the two gradients; negative once the terms conflict.

    .. attribute :: audio_row_norms

       Per-row audio gradient norm, un-reduced so it can be bucketed by flow time.
    """

    ratio: Tensor
    cosine: Tensor
    audio_row_norms: Tensor


@jaxtyped(typechecker=beartype)
def gradient_balance(
    *,
    flow_loss: Float[Tensor, _SCALAR_SHAPE],
    audio_term: Float[Tensor, _SCALAR_SHAPE],
    shared: Float[Tensor, _BATCH_ANY_SHAPE],
) -> GradientBalance:
    """Measure how two loss terms contribute gradient at a tensor they both reach.

    Loss magnitude is a poor proxy for gradient magnitude, so tune ``lambda_audio`` against the
    ratio and read the cosine as a conflict signal — rationale and citations in
    https://github.com/tinaudio/synth-setter/issues/2628#issuecomment-5111367681. Both gradients
    retain the graph, so the caller's own backward pass is unaffected.

    :param flow_loss: Scalar flow-matching loss.
    :param audio_term: Scalar weighted audio loss.
    :param shared: Batch-first tensor both terms backpropagate through.
    :returns: The two aggregate diagnostics plus the un-reduced per-row audio norms.
    """
    (flow_grad,) = torch.autograd.grad(flow_loss, shared, retain_graph=True)
    (audio_grad,) = torch.autograd.grad(audio_term, shared, retain_graph=True)
    return GradientBalance(
        ratio=audio_grad.norm() / flow_grad.norm().clamp_min(_GRAD_NORM_EPS),
        cosine=functional.cosine_similarity(audio_grad.flatten(), flow_grad.flatten(), dim=0),
        audio_row_norms=audio_grad.flatten(start_dim=1).norm(dim=-1),
    )


@jaxtyped(typechecker=beartype)
def time_bucket_means(
    values: Float[Tensor, _BATCH_SHAPE],
    t: Float[Tensor, _BATCH_TIME_SHAPE],
    num_buckets: int = _TIME_BUCKETS,
) -> Float[Tensor, _BUCKETS_SHAPE]:
    """Average per-row values inside equal-width flow-time buckets spanning ``[0, 1]``.

    :param values: One value per row.
    :param t: Flow time shaped ``(batch, 1)``.
    :param num_buckets: Number of equal-width buckets.
    :returns: Per-bucket mean shaped ``(num_buckets,)``; NaN where no row landed.
    """
    bucket = (t.squeeze(-1) * num_buckets).long().clamp(0, num_buckets - 1)
    totals = torch.zeros(num_buckets, device=values.device, dtype=values.dtype)
    counts = torch.zeros_like(totals)
    totals.index_add_(0, bucket, values)
    counts.index_add_(0, bucket, torch.ones_like(values))
    # 0/0 marks an empty bucket as NaN, which the caller skips rather than logging a zero.
    return totals / counts


@jaxtyped(typechecker=beartype)
def metric_tap(encoder: nn.Module) -> EmbeddingTap:
    """Resolve the embedding the audio loss measures distance in.

    Encoders with a frozen backbone expose ``embed``; ones without are their own tap. The
    metric never taps a trainable head, which would reintroduce the between-step drift the
    frozen backbone exists to remove.

    :param encoder: Conditioning encoder.
    :returns: Callable mapping a waveform batch to its metric-space embedding.
    """
    return cast(EmbeddingTap, getattr(encoder, "embed", encoder))


@jaxtyped(typechecker=beartype)
def _frozen_embedder(encoder: nn.Module) -> EmbeddingTap:
    """Build a tap that cannot be reshaped by the gradient it carries.

    A frozen backbone is already stationary. A jointly trained encoder is instead called
    through ``functional_call`` on detached parameters, so the term cannot shrink by
    collapsing the encoder while gradient still reaches the rendered estimate.

    :param encoder: Conditioning encoder, expected to be in eval mode already.
    :returns: Callable mapping a waveform batch to its metric-space embedding.
    """
    tap = metric_tap(encoder)
    if tap is not encoder:
        return tap
    frozen_state = {
        name: tensor.detach()
        for name, tensor in (*encoder.named_parameters(), *encoder.named_buffers())
    }
    return lambda signal: torch.func.functional_call(encoder, frozen_state, (signal,))


@jaxtyped(typechecker=beartype)
def _frozen_latent_distance(
    encoder: nn.Module,
    rendered: Float[Tensor, _BATCH_AUDIO_SHAPE],
    target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
    target_embedding: Float[Tensor, _BATCH_ANY_SHAPE] | None = None,
) -> Float[Tensor, _BATCH_SHAPE]:
    """Per-sample cosine distance in an encoder's embedding space, holding the space fixed.

    The encoder runs in eval mode so BatchNorm running stats cannot drift, and its space is
    held fixed either by a frozen backbone or by ``functional_call`` on detached parameters.
    Cosine rather than raw MSE because embedding norm carries no fixed meaning while the
    encoder trains, which would let the term's magnitude drift with activation scale alone.

    :param encoder: Encoder defining the latent space; left in its original mode.
    :param rendered: Rendered estimate shaped ``(batch, samples)``.
    :param target_audio: Observed audio, same shape.
    :param target_embedding: Embedding of ``target_audio`` the caller already computed;
        ``None`` recomputes it.
    :returns: Per-sample distance in ``[0, 2]`` shaped ``(batch,)``.
    """
    was_training = encoder.training
    encoder.eval()
    try:
        embed = _frozen_embedder(encoder)
        if target_embedding is None:
            target_embedding = embed(target_audio)
        # flatten(1) collapses token/layer axes so sequence encoders also reduce to a
        # per-sample scalar instead of broadcasting against the (batch,) weight.
        return 1.0 - functional.cosine_similarity(
            embed(rendered).flatten(start_dim=1),
            target_embedding.flatten(start_dim=1),
            dim=-1,
        )
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
        render_batch_size: int,
    ) -> None:
        """Configure the render geometry and the term's weighting.

        :param lambda_audio: Finite positive audio-term weight at t=1.
        :param t_min: Flow time at which the term switches on, in ``[0, 1)``.
        :param sample_rate: Render sample rate in Hz.
        :param signal_length: Rendered samples per row.
        :param render_batch_size: Rows the renderer's voice holds; must cover the
            training batch size, which shorter batches pad up to.
        :raises ValueError: Non-finite/non-positive ``lambda_audio``, out-of-range
            ``t_min``, or non-positive ``render_batch_size``.
        """
        super().__init__()
        if not math.isfinite(lambda_audio) or lambda_audio <= 0.0:
            raise ValueError(
                f"lambda_audio must be finite and positive, got {lambda_audio}; omit the audio loss "
                "entirely for the no-render control arm"
            )
        if not 0.0 <= t_min < 1.0:
            raise ValueError(f"t_min must lie in [0, 1), got {t_min}")
        if render_batch_size <= 0:
            raise ValueError(f"render_batch_size must be positive, got {render_batch_size}")
        self.lambda_audio = lambda_audio
        self.t_min = t_min
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.render_batch_size = render_batch_size

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
        target_embedding: Float[Tensor, _BATCH_ANY_SHAPE] | None = None,
    ) -> Float[Tensor, _SCALAR_SHAPE]:
        """Render the estimate and return the weighted latent distance to the target.

        :param theta_hat: One-step parameter estimate in model space ``[-1, 1]``.
        :param t: Flow time shaped ``(batch, 1)``.
        :param target_audio: Observed audio shaped ``(batch, signal_length)``.
        :param encoder: Encoder defining the latent space the distance is measured in.
        :param keep: Optional CFG keep mask shaped ``(batch,)``; rows at ``False`` are
            zero-weighted because their estimate is drawn from the marginal, making the
            residual against that row's own audio near-arbitrary.
        :param target_embedding: Metric-space embedding of ``target_audio`` the caller
            already computed for conditioning; ``None`` recomputes it.
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
            render_batch_size=self.render_batch_size,
        )
        # The stored target was hard-clamped by render_torchsynth; a straight-through
        # clamp matches that contract without zeroing gradient on clipped samples.
        rendered = rendered + (rendered.clamp(-1.0, 1.0) - rendered).detach()
        distance = _frozen_latent_distance(encoder, rendered, target_audio, target_embedding)
        return (weight * distance).mean()
