"""Pluggable per-sample distances the audio-feedback term can measure a render in.

Each distance maps a rendered and a target waveform batch to one non-negative scalar per row, so
the caller can weight rows by flow time. Spectral distances carry no parameters and so hold their
space fixed by construction; embedding distances need a frozen encoder.
"""

from __future__ import annotations

import math
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
# DiffVox MLDR integration scales and torchcomp's millisecond-to-coefficient numerator.
_LDR_SCALES_MS: Final[tuple[tuple[float, float], ...]] = ((50.0, 1000.0), (100.0, 2000.0))
_LDR_ENERGY_FLOOR: Final = 1e-8
_TORCHCOMP_MS_TO_COEF: Final = 2200.0

_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_CHANNEL_AUDIO_SHAPE = "batch channels samples"
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


@jaxtyped(typechecker=beartype)
def _one_pole_average(
    energy: Float[Tensor, "rows samples"], *, time_ms: float, sample_rate: int
) -> Float[Tensor, "rows samples"]:
    """Apply torchcomp's causal running-average envelope along the sample axis.

    :param energy: Non-negative signal rows.
    :param time_ms: Integration time in milliseconds.
    :param sample_rate: Sample rate in Hz.
    :returns: Smoothed signal with the same shape and dtype.
    """
    coefficient = 1.0 - torch.exp(
        energy.new_tensor(-_TORCHCOMP_MS_TO_COEF / (time_ms * sample_rate))
    )
    numerator = torch.stack((coefficient, energy.new_zeros(())))
    denominator = torch.stack((energy.new_ones(()), coefficient - 1.0))
    return torchaudio.functional.lfilter(
        energy, denominator, numerator, clamp=False, batching=False
    )


@jaxtyped(typechecker=beartype)
def _loudness_dynamic_range(
    energy: Float[Tensor, "rows samples"],
    *,
    short_ms: float,
    long_ms: float,
    sample_rate: int,
) -> Float[Tensor, "rows samples"]:
    """Align short and long energy envelopes and return their log ratio.

    :param energy: Floored squared signal rows.
    :param short_ms: Short integration time in milliseconds.
    :param long_ms: Long integration time in milliseconds.
    :param sample_rate: Sample rate in Hz.
    :returns: Log loudness-dynamic-range signal.
    """
    align_shift = int(sample_rate * (long_ms - short_ms) / 2000.0)
    short_envelope = _one_pole_average(energy, time_ms=short_ms, sample_rate=sample_rate)
    long_envelope = torch.roll(
        _one_pole_average(energy, time_ms=long_ms, sample_rate=sample_rate),
        -align_shift,
        dims=-1,
    )
    return short_envelope.log() - long_envelope.log()


@jaxtyped(typechecker=beartype)
def _mldr_corresponding_channels(
    rendered: Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    target: Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    *,
    sample_rate: int,
) -> Float[Tensor, _BATCH_SHAPE]:
    """Return DiffVox MLDR averaged over corresponding channels for each batch row.

    :param rendered: Predicted channelized audio.
    :param target: Detached target audio with matching geometry.
    :param sample_rate: Sample rate governing envelope time constants.
    :returns: Per-row distance in natural-log units.
    """
    batch, channels, samples = rendered.shape
    rendered_energy = rendered.square().clamp_min(_LDR_ENERGY_FLOOR).reshape(-1, samples)
    target_energy = target.square().clamp_min(_LDR_ENERGY_FLOOR).reshape(-1, samples)
    distance = rendered.new_zeros(batch)
    for short_ms, long_ms in _LDR_SCALES_MS:
        rendered_ldr = _loudness_dynamic_range(
            rendered_energy,
            short_ms=short_ms,
            long_ms=long_ms,
            sample_rate=sample_rate,
        )
        target_ldr = _loudness_dynamic_range(
            target_energy,
            short_ms=short_ms,
            long_ms=long_ms,
            sample_rate=sample_rate,
        )
        distance = distance + (rendered_ldr - target_ldr).abs().reshape(
            batch, channels, samples
        ).mean(dim=(1, 2))
    return distance


class MultichannelAudioDistance(nn.Module):
    """Weighted spectral, channel-envelope, and spatial-envelope audio distance."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        sample_rate: int,
        spectral_weight: float,
        channel_mldr_weight: float,
        pair_mldr_weight: float,
    ) -> None:
        """Configure one channel-aware objective without an external channel adapter.

        :param sample_rate: Waveform rate in Hz.
        :param spectral_weight: Corresponding-channel MSS weight.
        :param channel_mldr_weight: Corresponding-channel MLDR weight.
        :param pair_mldr_weight: Within-signal channel-pair MLDR weight.
        :raises ValueError: Weights are non-finite, negative, or all zero.
        """
        super().__init__()
        weights = (spectral_weight, channel_mldr_weight, pair_mldr_weight)
        if not all(math.isfinite(weight) for weight in weights):
            raise ValueError(f"component weights must be finite, got {weights}")
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"component weights must be nonnegative, got {weights}")
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one component weight must be positive")
        self.sample_rate = sample_rate
        self.spectral_weight = spectral_weight
        self.channel_mldr_weight = channel_mldr_weight
        self.pair_mldr_weight = pair_mldr_weight
        self.spectral = MultiScaleSpectralDistance(sample_rate=sample_rate)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        rendered: Float[Tensor, ...],
        target: Float[Tensor, ...],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Return one distance per row for exact matching mono or channelized audio.

        :param rendered: Predicted audio shaped ``(B,T)`` or ``(B,C,T)``.
        :param target: Target audio with exactly the same geometry.
        :returns: Weighted per-row distance shaped ``(B,)``.
        """
        rendered, target = self._validated_channel_audio(rendered, target)
        target = target.detach()
        batch, channels, samples = rendered.shape

        spectral = rendered.new_zeros(batch)
        if self.spectral_weight > 0.0:
            spectral = (
                self.spectral(
                    rendered.reshape(batch * channels, samples),
                    target.reshape(batch * channels, samples),
                )
                .reshape(batch, channels)
                .mean(dim=1)
            )

        channel_mldr = rendered.new_zeros(batch)
        if self.channel_mldr_weight > 0.0:
            channel_mldr = _mldr_corresponding_channels(
                rendered, target, sample_rate=self.sample_rate
            )

        pair_mldr = self._pair_mldr(rendered, target)
        return (
            self.spectral_weight * spectral
            + self.channel_mldr_weight * channel_mldr
            + self.pair_mldr_weight * pair_mldr
        )

    @jaxtyped(typechecker=beartype)
    def _validated_channel_audio(
        self,
        rendered: Float[Tensor, ...],
        target: Float[Tensor, ...],
    ) -> tuple[
        Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
        Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    ]:
        """Canonicalize valid audio to ``(B,C,T)`` and reject unsafe inputs.

        :param rendered: Predicted mono or channelized audio.
        :param target: Target mono or channelized audio.
        :returns: Matching nonempty channelized tensors.
        :raises ValueError: Geometry, dtype, or finite-value validation fails.
        """
        if rendered.ndim not in (2, 3) or target.ndim not in (2, 3):
            raise ValueError(
                "rendered and target must have matching nonempty (B,T) or (B,C,T) shapes"
            )
        rendered = rendered.unsqueeze(1) if rendered.ndim == 2 else rendered
        target = target.unsqueeze(1) if target.ndim == 2 else target
        if rendered.shape != target.shape or any(size == 0 for size in rendered.shape):
            raise ValueError(
                "rendered and target must have matching nonempty (B,T) or (B,C,T) shapes"
            )
        if rendered.dtype not in (torch.float32, torch.float64) or target.dtype != rendered.dtype:
            raise ValueError("rendered and target must share float32 or float64 dtype")
        if not torch.isfinite(rendered).all() or not torch.isfinite(target).all():
            raise ValueError("rendered and target must contain only finite values")
        return rendered, target

    @jaxtyped(typechecker=beartype)
    def _pair_mldr(
        self,
        rendered: Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
        target: Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Average MLDR over all within-signal unordered sum/difference channel pairs.

        :param rendered: Validated predicted audio.
        :param target: Validated detached target audio.
        :returns: Per-row pair distance, zero when pair scoring is disabled or audio is mono.
        """
        batch, channels, _ = rendered.shape
        if self.pair_mldr_weight == 0.0 or channels == 1:
            return rendered.new_zeros(batch)
        pairs = torch.triu_indices(channels, channels, offset=1, device=rendered.device)
        scale = 2.0**-0.5
        rendered_pairs, target_pairs = [
            torch.stack(
                (
                    (audio[:, pairs[0]] + audio[:, pairs[1]]) * scale,
                    (audio[:, pairs[0]] - audio[:, pairs[1]]) * scale,
                ),
                dim=2,
            ).flatten(1, 2)
            for audio in (rendered, target)
        ]
        return _mldr_corresponding_channels(
            rendered_pairs, target_pairs, sample_rate=self.sample_rate
        )


@jaxtyped(typechecker=beartype)
def _canonical_mono_audio(
    audio: Float[Tensor, _BATCH_AUDIO_SHAPE] | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
) -> Float[Tensor, _BATCH_AUDIO_SHAPE]:
    """Remove a singleton channel axis for mono-only embedding distances.

    :param audio: Flat or explicitly mono waveform batch.
    :returns: Flat waveform batch.
    :raises ValueError: Audio contains more than one channel.
    """
    if audio.ndim == 2:
        return audio
    if audio.shape[1] != 1:
        raise ValueError(f"embedding distances require mono audio, got {audio.shape[1]} channels")
    return audio[:, 0]


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
        rendered: Float[Tensor, _BATCH_AUDIO_SHAPE] | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
        target: Float[Tensor, _BATCH_AUDIO_SHAPE] | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Embed both mono waveforms and return one minus their cosine similarity.

        Cosine rather than raw distance because embedding norm carries no fixed meaning, and
        ``flatten(1)`` collapses token axes so sequence encoders also reduce per sample.

        :param rendered: Rendered estimate shaped ``(batch, samples)``.
        :param target: Observed audio, same shape.
        :returns: Per-sample distance in ``[0, 2]`` shaped ``(batch,)``.
        """
        embedded: list[Float[Tensor, _BATCH_ANY_SHAPE]] = [
            self.encoder(_canonical_mono_audio(signal)).flatten(start_dim=1)
            for signal in (rendered, target)
        ]
        return 1.0 - functional.cosine_similarity(embedded[0], embedded[1].detach(), dim=-1)


class LatentMseDistance(_FrozenEncoderDistance):
    """Magnitude-normalized squared error in a frozen encoder's latent space, per sample."""

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        rendered: Float[Tensor, _BATCH_AUDIO_SHAPE] | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
        target: Float[Tensor, _BATCH_AUDIO_SHAPE] | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Return the target-variance-normalized latent error of each mono row.

        Normalizing by the target's own detached variance — Stable Audio 3's ``sample``-mode
        loss normalization — keeps high-magnitude latents from swamping quiet rows.

        :param rendered: Rendered estimate shaped ``(batch, samples)``.
        :param target: Observed audio, same shape.
        :returns: Per-sample distance shaped ``(batch,)``.
        """
        rendered_latents = self.encoder(_canonical_mono_audio(rendered)).flatten(start_dim=1)
        target_latents = self.encoder(_canonical_mono_audio(target)).flatten(start_dim=1).detach()
        magnitude = target_latents.var(dim=-1, keepdim=True, unbiased=False) + _LOSS_NORM_EPS
        return ((rendered_latents - target_latents) ** 2 / magnitude).mean(dim=-1)
