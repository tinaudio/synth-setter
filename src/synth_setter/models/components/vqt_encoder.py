"""Checkpoint-free online nnAudio2 VQT conditioning encoder."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

import torch
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from torch import Tensor, nn
from torch.nn import functional

_BATCH_AUDIO_SHAPE = "batch *channels samples"
_BATCH_VQT_SHAPE = "batch vqt_bins frames"
_SUPPORTED_PAD_MODES = frozenset({"constant", "reflect"})


@runtime_checkable
class _VqtTransform(Protocol):
    """Protocol for the nnAudio2 VQT operation used by the online encoder."""

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "batch samples"],
        *,
        output_format: str,
        normalization_type: str,
    ) -> object:
        """Return VQT coefficients.

        :param x: Mono float32 waveform batch.
        :param output_format: nnAudio2 coefficient representation.
        :param normalization_type: nnAudio2 normalization policy.
        :returns: Upstream VQT result.
        """
        ...


class VqtAudioEncoder(nn.Module):
    """Extract fixed-policy VQT features from channel-first waveforms."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        sample_rate: int,
        hop_length: int,
        fmin: float,
        n_bins: int,
        bins_per_octave: int,
        gamma: float,
        max_batch_size: int = 32,
        pad_mode: str = "reflect",
    ) -> None:
        """Configure frozen, memory-bounded VQT extraction.

        :param sample_rate: Input waveform sample rate in Hz.
        :param hop_length: Samples between adjacent output frames.
        :param fmin: Center frequency of the lowest VQT bin in Hz.
        :param n_bins: Number of log-frequency bins.
        :param bins_per_octave: Number of bins spanning one octave.
        :param gamma: Positive variable-Q bandwidth offset.
        :param max_batch_size: Maximum rows per transform call, or ``-1`` for the full batch.
        :param pad_mode: Boundary padding mode accepted by nnAudio2.
        :raises ValueError: A transform parameter is outside its supported range.
        """
        super().__init__()
        positive_values = {
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "fmin": fmin,
            "n_bins": n_bins,
            "bins_per_octave": bins_per_octave,
            "gamma": gamma,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if max_batch_size != -1 and max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive or -1, got {max_batch_size}")
        if pad_mode not in _SUPPORTED_PAD_MODES:
            raise ValueError(
                f"pad_mode must be one of {sorted(_SUPPORTED_PAD_MODES)}, got {pad_mode!r}"
            )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.gamma = gamma
        self.max_batch_size = max_batch_size
        self.pad_mode = pad_mode
        self.out_dim = n_bins
        self._transforms: dict[torch.device, _VqtTransform] = {}

    @jaxtyped(typechecker=beartype)
    def _apply(
        self,
        fn: Callable[[Shaped[Tensor, ...]], Shaped[Tensor, ...]],
        recurse: bool = True,
    ) -> VqtAudioEncoder:
        """Apply a module conversion after releasing device-bound VQT state.

        :param fn: Tensor conversion used by :class:`torch.nn.Module`.
        :param recurse: Whether to convert child modules recursively.
        :returns: This converted module.
        """
        self._transforms.clear()
        return super()._apply(fn, recurse)

    @jaxtyped(typechecker=beartype)
    def _transform(self, device: torch.device) -> _VqtTransform:
        """Return the frozen nnAudio2 transform for one device.

        :param device: Device holding the waveform batch.
        :returns: Cached VQT transform.
        """
        transform = self._transforms.get(device)
        if transform is None:
            from nnAudio2.features.vqt import VQT

            vqt_factory = cast(Callable[..., nn.Module], VQT)
            transform = cast(
                _VqtTransform,
                vqt_factory(
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    fmin=self.fmin,
                    n_bins=self.n_bins,
                    bins_per_octave=self.bins_per_octave,
                    gamma=self.gamma,
                    pad_mode=self.pad_mode,
                    trainable=False,
                    output_format="Magnitude",
                    verbose=False,
                ).to(device=device, dtype=torch.float32),
            )
            self._transforms[device] = transform
        return transform

    @jaxtyped(typechecker=beartype)
    def forward(self, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]) -> Float[Tensor, _BATCH_VQT_SHAPE]:
        """Return detached float32-computed log-magnitude VQT features.

        :param audio: Waveforms shaped ``(batch, samples)`` or
            ``(batch, channels, samples)``; channels are averaged.
        :returns: Features shaped ``(batch, n_bins, frames)`` on the input device.
        :raises ValueError: Audio has invalid rank or an empty dimension.
        :raises TypeError: nnAudio2 returns an unsupported value.
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
        output_frames = 1 + channel_audio.shape[-1] // self.hop_length
        batch_size = len(channel_audio) if self.max_batch_size == -1 else self.max_batch_size
        chunks = []
        with torch.no_grad():
            for chunk in channel_audio.split(batch_size):
                waveform = chunk.detach().to(dtype=torch.float32).mean(dim=1)
                coefficients = self._transform(chunk.device)(
                    waveform,
                    output_format="Magnitude",
                    normalization_type="librosa",
                )
                if not isinstance(coefficients, Tensor):
                    raise TypeError(f"nnAudio2 VQT returned {type(coefficients).__name__}")
                chunks.append(
                    functional.adaptive_avg_pool1d(torch.log1p(coefficients), output_frames)
                )
        return torch.cat(chunks).to(dtype=audio.dtype)
