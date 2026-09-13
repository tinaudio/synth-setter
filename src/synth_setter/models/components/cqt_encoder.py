"""Checkpoint-free online CQT conditioning encoder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as functional
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor, nn

from synth_setter.cqt import (
    CQT_BINS_PER_OCTAVE,
    CQT_EMBEDDING_DIM,
    CQT_MODE,
    CQT_NUM_OCTAVES,
    cqt_num_frames,
)

_BATCH_AUDIO_SHAPE: Final = "batch *channels samples"
_BATCH_CQT_SHAPE: Final = "batch cqt_bins frames"


@runtime_checkable
class _CqtTransform(Protocol):
    """Upstream CQT operation used by the online encoder."""

    @jaxtyped(typechecker=beartype)
    def fwd(self, x: Float[Tensor, "batch 1 samples"]) -> object:
        """Return matrix-mode CQT coefficients.

        :param x: Mono float32 waveform batch.
        :returns: Upstream matrix-mode result.
        """
        ...


class CqtAudioEncoder(nn.Module):
    """Extract canonical CQT features from mono or channel-first waveforms.

    .. attribute :: out_dim

       Number of log-frequency bins emitted per frame.
    """

    out_dim: Final = CQT_EMBEDDING_DIM

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, sample_rate: int, max_batch_size: int = 32) -> None:
        """Configure fixed-rate, memory-bounded CQT extraction.

        :param sample_rate: Input waveform sample rate in Hz.
        :param max_batch_size: Maximum rows processed by one upstream transform call.
        :raises ValueError: Either configuration value is not positive.
        """
        super().__init__()
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self.sample_rate = sample_rate
        self.max_batch_size = max_batch_size
        self._transforms: dict[tuple[torch.device, int], _CqtTransform] = {}

    @jaxtyped(typechecker=beartype)
    def _apply(
        self,
        fn: Callable[[Shaped[Tensor, ...]], Shaped[Tensor, ...]],
        recurse: bool = True,
    ) -> CqtAudioEncoder:
        """Apply a module conversion after releasing device-bound CQT state.

        :param fn: Tensor conversion used by :class:`torch.nn.Module`.
        :param recurse: Whether to convert child modules recursively.
        :returns: This converted module.
        """
        self._transforms.clear()
        return super()._apply(fn, recurse)

    @jaxtyped(typechecker=beartype)
    def _transform(self, device: torch.device, num_samples: int) -> _CqtTransform:
        """Return the upstream transform for one device and waveform length.

        :param device: Device holding the waveform batch.
        :param num_samples: Fixed waveform length consumed by the transform.
        :returns: Cached checkpoint-free CQT transform.
        """
        key = device, num_samples
        transform = self._transforms.get(key)
        if transform is None:
            from cqt_nsgt_pytorch import CQT_nsgt

            transform = cast(
                _CqtTransform,
                CQT_nsgt(
                    CQT_NUM_OCTAVES,
                    CQT_BINS_PER_OCTAVE,
                    mode=CQT_MODE,
                    fs=self.sample_rate,
                    audio_len=num_samples,
                    device=str(device),
                    dtype=torch.float32,
                    verbose=False,
                ),
            )
            self._transforms[key] = transform
        return transform

    @jaxtyped(typechecker=beartype)
    def forward(self, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]) -> Float[Tensor, _BATCH_CQT_SHAPE]:
        """Return detached float32 log-magnitude CQT features.

        :param audio: Waveforms shaped ``(batch, samples)`` or
            ``(batch, channels, samples)``; channels are averaged.
        :returns: Features shaped ``(batch, 256, frames)`` on the input device.
        :raises ValueError: Audio has invalid rank or an empty dimension.
        :raises TypeError: The upstream transform returns an unsupported value.
        """
        if audio.ndim not in (2, 3):
            raise ValueError(f"expected audio shape (B, [C,] T), got {tuple(audio.shape)}")
        if audio.shape[0] == 0:
            raise ValueError("audio batch cannot be empty")
        if audio.ndim == 3 and audio.shape[1] == 0:
            raise ValueError("audio channels cannot be empty")
        if audio.shape[-1] == 0:
            raise ValueError("audio samples cannot be empty")

        channel_audio = audio.unsqueeze(1) if audio.ndim == 2 else audio
        output_frames = cqt_num_frames(channel_audio.shape[-1], self.sample_rate)
        chunks = []
        with torch.no_grad():
            for chunk in channel_audio.split(self.max_batch_size):
                waveform = chunk.detach().to(dtype=torch.float32).mean(dim=1, keepdim=True)
                coefficients = self._transform(chunk.device, chunk.shape[-1]).fwd(waveform)
                if not isinstance(coefficients, Tensor):
                    raise TypeError(f"CQT {CQT_MODE} mode returned {type(coefficients).__name__}")
                chunks.append(
                    functional.adaptive_avg_pool1d(
                        torch.log1p(coefficients.abs()).squeeze(1), output_frames
                    )
                )
        return torch.cat(chunks).to(dtype=audio.dtype)
