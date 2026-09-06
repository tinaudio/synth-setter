"""Black-box reward scoring sampled parameters by re-rendering them against target audio."""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
)

_BATCH_SHAPE = "batch"
_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_PARAMS_SHAPE = "batch params"


class RenderedAudioReward(nn.Module):
    """Negative audio distance between a rendered parameter row and its target audio."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        distance: nn.Module,
        sample_rate: int,
        signal_length: int,
        render_batch_size: int,
    ) -> None:
        """Configure the render geometry and the distance the reward negates.

        :param distance: Module mapping ``(rendered, target)`` to a per-row distance.
        :param sample_rate: Render sample rate in Hz.
        :param signal_length: Rendered samples per row; must match the target audio.
        :param render_batch_size: Rows the renderer's voice holds; wider batches render in
            chunks of this size.
        :raises ValueError: ``render_batch_size`` is not positive.
        """
        super().__init__()
        if render_batch_size <= 0:
            raise ValueError(f"render_batch_size must be positive, got {render_batch_size}")
        self.distance = distance
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.render_batch_size = render_batch_size

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def forward(
        self,
        theta: Float[Tensor, _BATCH_PARAMS_SHAPE],
        target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Render every row and score it against its own target.

        :param theta: Sampled parameters in model space ``[-1, 1]``.
        :param target_audio: Observed audio shaped ``(batch, signal_length)``.
        :returns: Per-row reward, higher for renders closer to the target.
        """
        params = differentiable_decode(theta.detach())
        # The grad renderer, not the dataset one: it aligns every row to the noise chunk the
        # targets were rendered with, so a row's reward cannot depend on its batch position.
        rendered = torch.cat(
            [
                render_torchsynth_grad(
                    chunk,
                    sample_rate=self.sample_rate,
                    signal_length=self.signal_length,
                    render_batch_size=self.render_batch_size,
                )
                for chunk in params.split(self.render_batch_size)
            ]
        )
        return -self.distance(rendered, target_audio)
