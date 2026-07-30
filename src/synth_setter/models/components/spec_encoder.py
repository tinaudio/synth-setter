"""Waveform-to-spectrogram front end and the encoder pairing it with a backbone.

Online-render synths have no stored mel column, so their conditioning encoder computes
features from the waveform. Keeping the front end separate from the backbone lets the
same spectrogram-in backbones serve both the stored-mel and the online path.

Example::

    encoder = SpecEncoder(
        frontend=LogMelFrontend(176_400, sample_rate=44_100),
        backbone=MelCNN(hidden_dim=16, out_dim=512),
    )
"""

import math
from typing import Final, Literal

import torch
import torch.nn as nn
import torchaudio
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from synth_setter.data.vst.shapes import MEL_N_MELS, mel_hop_length, mel_n_fft

# Permissive so wrong-rank batches reach this module's own shape error rather than
# beartype's, which cannot report the expected sample count.
_BATCH_AUDIO_SHAPE: Final = "batch ... samples"
_BATCH_GRID_SHAPE: Final = "batch 1 mels frames"
_BATCH_ANY_SHAPE: Final = "batch ..."


class LogMelFrontend(nn.Module):
    """Convert fixed-length waveforms into the log-mel grid the dataset writers store."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        in_dim: int,
        *,
        sample_rate: int,
        center: bool = True,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_fft: int | None = None,
        hop_length: int | None = None,
        n_mels: int = MEL_N_MELS,
        pad_mode: Literal["constant", "reflect"] = "constant",
        power: float = 2.0,
        mel_norm: Literal["slaney"] | None = "slaney",
        mel_scale: Literal["htk", "slaney"] = "slaney",
        window: Literal["hamming", "hann"] = "hamming",
        amin: float = 1e-10,
        top_db: float | None = 80.0,
    ) -> None:
        """Build the mel transform and the decibel scaling applied to its output.

        :param in_dim: Expected waveform length in samples.
        :param sample_rate: Waveform sample rate in Hz.
        :param center: Whether to pad waveforms so frames are centered on timestamps.
        :param f_min: Lowest frequency included in the mel filter bank, in Hz.
        :param f_max: Highest included frequency, in Hz; ``None`` selects Nyquist.
        :param n_fft: Fourier transform size; defaults to 25 ms of audio.
        :param hop_length: Frame stride; defaults to 100 frames per second.
        :param n_mels: Number of mel-frequency bins.
        :param pad_mode: Waveform padding mode used when ``center`` is enabled.
        :param power: Exponent applied to the magnitude spectrogram.
        :param mel_norm: Area normalization applied to mel filter-bank weights.
        :param mel_scale: Mel-frequency conversion formula.
        :param window: Window function applied before each Fourier transform.
        :param amin: Lower power bound used before converting to decibels.
        :param top_db: Dynamic range limit in decibels; ``None`` disables clipping.
        :raises ValueError: If any numeric bound is unsupported.
        """
        super().__init__()
        if not math.isfinite(amin) or amin <= 0:
            raise ValueError(f"amin must be positive and finite, got {amin}")
        nyquist = sample_rate / 2
        if not math.isfinite(f_min) or not 0 <= f_min < nyquist:
            raise ValueError(f"f_min must be finite and below Nyquist, got {f_min}")
        if f_max is not None and (not math.isfinite(f_max) or not f_min < f_max <= nyquist):
            raise ValueError(
                f"f_max must be finite, above f_min, and no greater than Nyquist, got {f_max}"
            )
        for name, value in (("hop_length", hop_length), ("n_fft", n_fft), ("n_mels", n_mels)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not math.isfinite(power) or power <= 0:
            raise ValueError(f"power must be positive and finite, got {power}")
        if top_db is not None and (not math.isfinite(top_db) or top_db < 0):
            raise ValueError(f"top_db must be non-negative and finite, got {top_db}")
        window_fn = {"hamming": torch.hamming_window, "hann": torch.hann_window}[window]

        self.in_dim = in_dim
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            center=center,
            f_min=f_min,
            f_max=f_max,
            n_fft=n_fft if n_fft is not None else mel_n_fft(sample_rate),
            hop_length=hop_length if hop_length is not None else mel_hop_length(sample_rate),
            n_mels=n_mels,
            pad_mode=pad_mode,
            window_fn=window_fn,
            power=power,
            norm=mel_norm,
            mel_scale=mel_scale,
        )
        self.amin = amin
        self.db_multiplier = 20.0 / power
        self.top_db = top_db

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, _BATCH_AUDIO_SHAPE]) -> Float[Tensor, _BATCH_GRID_SHAPE]:
        """Return per-waveform log-mel power relative to each waveform's peak.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Decibel-scaled mel grids shaped ``(batch, 1, mels, frames)``.
        :raises ValueError: If the waveform shape differs from the configured input length.
        """
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected waveform shape (batch, {self.in_dim}), got {tuple(x.shape)}"
            )
        log_mel = self.db_multiplier * torch.log10(torch.clamp(self.mel(x), min=self.amin))
        log_mel = log_mel - log_mel.amax(dim=(-2, -1), keepdim=True)
        if self.top_db is not None:
            log_mel = torch.clamp(log_mel, min=-self.top_db)
        return log_mel.unsqueeze(1)


class SpecEncoder(nn.Module):
    """Encode waveforms by running a feature front end into a spectrogram-in backbone."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, frontend: nn.Module, backbone: nn.Module) -> None:
        """Pair a front end with the backbone that consumes its feature grid.

        :param frontend: Waveform-in module emitting ``(batch, channels, mels, frames)``.
        :param backbone: Spectrogram-in module producing the conditioning tensor.
        """
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, _BATCH_AUDIO_SHAPE]) -> Float[Tensor, _BATCH_ANY_SHAPE]:
        """Encode a mono waveform batch into the backbone's conditioning tensor.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Whatever the backbone emits for the front end's feature grid.
        """
        return self.backbone(self.frontend(x))
