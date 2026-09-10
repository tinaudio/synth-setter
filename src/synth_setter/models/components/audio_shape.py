"""Shared waveform shape normalization."""

from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_CHANNEL_AUDIO_SHAPE = "batch channels samples"


@jaxtyped(typechecker=beartype)
def canonical_audio(
    audio: Float[Tensor, _BATCH_AUDIO_SHAPE] | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
) -> Float[Tensor, _BATCH_AUDIO_SHAPE] | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE]:
    """Remove only a singleton channel axis, preserving every multichannel waveform.

    :param audio: Batched rendered or target waveform, optionally carrying a channel axis.
    :returns: The unchanged channels, or a flat batch when exactly one channel is present.
    """
    return audio[:, 0] if audio.ndim == 3 and audio.shape[1] == 1 else audio
