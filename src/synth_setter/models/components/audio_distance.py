"""Pluggable per-sample distances the audio-feedback term can measure a render in.

Each distance maps a rendered and a target waveform batch to one non-negative scalar per row, so
the caller can weight rows by flow time. Spectral distances carry no parameters and so hold their
space fixed by construction; embedding distances need a frozen encoder.
"""

from __future__ import annotations

from typing import Final

import torch
import torchaudio
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional

# Window/hop in milliseconds with the mel-bin count, mirroring the reported MSS metric's
# ``MEL_PARAMS``. Duplicated rather than imported because that module pulls in librosa,
# which must not enter the training import path (plugin dlopen hazard, #2549).
MEL_SCALES: Final[tuple[tuple[int, int, int], ...]] = ((10, 5, 32), (25, 10, 64), (100, 50, 128))
# Stable Audio 3's ``loss_norm_eps``, guarding the variance divisor of a constant target.
_LOSS_NORM_EPS: Final = 1e-6
# librosa's ``power_to_db`` floor and its 10*log10 power convention.
_AMIN: Final = 1e-10
_POWER_TO_DB: Final = 10.0

_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_SHAPE = "batch"
_BATCH_ANY_SHAPE = "batch ..."


class MultiScaleSpectralDistance(nn.Module):
    """Mean absolute log-mel difference across three resolutions, per sample."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, sample_rate: int) -> None:
        """Build one mel transform per configured resolution.

        Slaney norm and scale rather than torchaudio's htk defaults, which would weight the upper
        bands differently from the reported metric.

        :param sample_rate: Waveform rate the window sizes are derived from.
        :raises ValueError: A configured window rounds to fewer than two samples at this rate.
        """
        super().__init__()
        transforms = []
        for window_ms, hop_ms, n_mels in MEL_SCALES:
            n_fft = int(window_ms * sample_rate / 1000.0)
            hop_length = int(hop_ms * sample_rate / 1000.0)
            if n_fft < 2 or hop_length < 1:
                raise ValueError(
                    f"sample_rate {sample_rate} yields n_fft={n_fft} hop={hop_length} for the "
                    f"{window_ms}ms scale; too small to transform"
                )
            transforms.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    window_fn=torch.hann_window,
                    power=2.0,
                    norm="slaney",
                    mel_scale="slaney",
                )
            )
        self.transforms = nn.ModuleList(transforms)

    @jaxtyped(typechecker=beartype)
    def _log_mel(
        self, transform: nn.Module, audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
    ) -> Float[Tensor, "batch mels frames"]:
        """Return one resolution's dB-scaled mel relative to each waveform's own peak.

        Peak-relative like the reported metric's ``ref=np.max``, which makes the distance
        invariant to overall gain.

        :param transform: Mel transform for one resolution.
        :param audio: Waveform batch.
        :returns: Peak-referenced decibel mel spectrogram.
        """
        power = torch.clamp(transform(audio), min=_AMIN)
        decibels = _POWER_TO_DB * torch.log10(power)
        return decibels - decibels.amax(dim=(-2, -1), keepdim=True)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        rendered: Float[Tensor, _BATCH_AUDIO_SHAPE],
        target: Float[Tensor, _BATCH_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Average the absolute per-bin decibel gap over bins and resolutions.

        :param rendered: Rendered estimate shaped ``(batch, samples)``.
        :param target: Observed audio, same shape.
        :returns: Per-sample distance shaped ``(batch,)``.
        """
        per_scale = [
            (self._log_mel(transform, rendered) - self._log_mel(transform, target))
            .abs()
            .mean(dim=(-2, -1))
            for transform in self.transforms
        ]
        return torch.stack(per_scale, dim=0).mean(dim=0)


class _FrozenEncoderDistance(nn.Module):
    """Shared frozen-encoder custody for distances measured in a learned space."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, encoder: nn.Module) -> None:
        """Adopt an already-frozen waveform encoder as the metric space.

        :param encoder: Frozen waveform-in module defining the space.
        :raises ValueError: The encoder has trainable parameters, which would move the space.
        """
        super().__init__()
        trainable = [name for name, p in encoder.named_parameters() if p.requires_grad]
        if trainable:
            raise ValueError(
                f"encoder must be frozen; {len(trainable)} trainable parameter(s) {trainable} "
                "would move the space the distance is measured in"
            )
        self.encoder = encoder

    @jaxtyped(typechecker=beartype)
    def train(self, mode: bool = True) -> _FrozenEncoderDistance:
        """Keep the encoder in eval mode so normalization statistics cannot drift.

        :param mode: Training mode requested for this module's own children.
        :returns: This module.
        """
        super().train(mode)
        self.encoder.eval()
        return self


class CosineEmbeddingDistance(_FrozenEncoderDistance):
    """Cosine distance in a frozen encoder's embedding space, per sample."""

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        rendered: Float[Tensor, _BATCH_AUDIO_SHAPE],
        target: Float[Tensor, _BATCH_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Embed both waveforms and return one minus their cosine similarity.

        Cosine rather than raw distance because embedding norm carries no fixed meaning, and
        ``flatten(1)`` collapses token axes so sequence encoders also reduce per sample.

        :param rendered: Rendered estimate shaped ``(batch, samples)``.
        :param target: Observed audio, same shape.
        :returns: Per-sample distance in ``[0, 2]`` shaped ``(batch,)``.
        """
        embedded: list[Float[Tensor, _BATCH_ANY_SHAPE]] = [
            self.encoder(signal).flatten(start_dim=1) for signal in (rendered, target)
        ]
        return 1.0 - functional.cosine_similarity(embedded[0], embedded[1].detach(), dim=-1)


class LatentMseDistance(_FrozenEncoderDistance):
    """Magnitude-normalized squared error in a frozen encoder's latent space, per sample."""

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        rendered: Float[Tensor, _BATCH_AUDIO_SHAPE],
        target: Float[Tensor, _BATCH_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Return the target-variance-normalized latent error of each row.

        Normalizing by the target's own detached variance — Stable Audio 3's ``sample``-mode
        loss normalization — keeps high-magnitude latents from swamping quiet rows.

        :param rendered: Rendered estimate shaped ``(batch, samples)``.
        :param target: Observed audio, same shape.
        :returns: Per-sample distance shaped ``(batch,)``.
        """
        rendered_latents = self.encoder(rendered).flatten(start_dim=1)
        target_latents = self.encoder(target).flatten(start_dim=1).detach()
        magnitude = target_latents.var(dim=-1, keepdim=True, unbiased=False) + _LOSS_NORM_EPS
        return ((rendered_latents - target_latents) ** 2 / magnitude).mean(dim=-1)
