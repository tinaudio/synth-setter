"""Convolutional and residual building blocks used by spectrum encoders.

Example::

    encoder = LogMelEncoder(176_400, 16, 256, sample_rate=44_100)
    embeddings = encoder(torch.zeros(2, 176_400))
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torchaudio

from synth_setter.data.vst.shapes import MEL_N_MELS, mel_hop_length, mel_n_fft


class ResidualMLPBlock(nn.Module):
    """Two-layer MLP with a learned linear shortcut and LayerNorm front."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        if out_dim is None and hidden_dim is not None:
            out_dim = hidden_dim
        elif out_dim is None:
            out_dim = in_dim

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.residual = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        )

    def forward(self, x):
        return self.residual(x) + self.net(x)


class LayerNormConv1dFriendly(nn.LayerNorm):
    """LayerNorm that normalizes over the channel axis of a ``(B, C, T)`` tensor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(-1, -2)).transpose(-1, -2)


class ResidualBlock(nn.Module):
    """1-D conv residual block with a 1x1-conv shortcut for channel-count changes."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        kernel_size: int = 7,
        norm: Literal["bn", "ln"] = "bn",
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_dim

        if out_dim is None and hidden_dim is not None:
            out_dim = hidden_dim
        elif out_dim is None:
            out_dim = in_dim

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            (nn.BatchNorm1d(hidden_dim) if norm == "bn" else LayerNormConv1dFriendly(hidden_dim)),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=out_dim,
                kernel_size=1,
                padding=0,
            ),
            nn.GELU(),
            nn.BatchNorm1d(out_dim) if norm == "bn" else nn.Identity(),
        )

        self.residual = (
            nn.Identity()
            if in_dim == out_dim
            else nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=1,
                padding=0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class ConvDownsampler(nn.Module):
    """1-D conv that halves (or stride-divides) the temporal axis and re-projects channels."""

    def __init__(self, in_dim: int, out_dim: int, stride: int, norm: Literal["bn", "ln"] = "bn"):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim) if norm == "bn" else LayerNormConv1dFriendly(in_dim),
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=stride * 2,
                stride=stride,
                padding=stride // 2,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualEncoder(nn.Module):
    """Stack of ResidualBlock + ConvDownsampler layers followed by an MLP head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_blocks: int = 4,
        kernel_size: int = 7,
        norm: Literal["bn", "ln"] = "bn",
    ):
        super().__init__()
        conv_layers = []
        dim = hidden_dim
        for i in range(num_blocks):
            conv_layers.extend(
                [
                    ResidualBlock(
                        1 if i == 0 else dim,
                        dim,
                        kernel_size=kernel_size,
                        norm=norm,
                    ),
                    ResidualBlock(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        norm=norm,
                    ),
                    ConvDownsampler(
                        dim,
                        dim * 2,
                        stride=3,
                        norm=norm,
                    ),
                ]
            )
            dim *= 2
        self.conv_net = nn.Sequential(*conv_layers)
        self.net = nn.Sequential(
            nn.LazyLinear(in_dim // 2),
            ResidualMLPBlock(in_dim // 2, in_dim // 2, out_dim),
        )

        self.register_buffer("_d", torch.empty(()))

        self._pass_junk_batch(in_dim)

    @property
    def device(self):
        return self._d.device

    def _pass_junk_batch(self, in_dim: int) -> torch.Tensor:
        in_batch = torch.randn(1, 1, in_dim // 2 + 1, device=self.device)
        y = self.conv_net(in_batch)
        self.net(y.flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(x, norm="forward")
        X = torch.abs(X)
        X = X[:, None, :]
        Z = self.conv_net(X)
        Z = Z.view(Z.shape[0], -1)
        return self.net(Z)


class LogMelEncoder(nn.Module):
    """Encode fixed-length waveforms with log-mel features and a pooled 2-D CNN.

    :param in_dim: Expected waveform length in samples.
    :param hidden_dim: Channel count in the first convolutional block.
    :param out_dim: Width of the returned embedding.
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
    :param num_blocks: Number of convolution and pooling blocks.
    :param kernel_size: Height and width of each convolutional kernel.
    :param norm: Normalization applied after each convolution.
    :raises ValueError: If ``norm`` or ``window`` is unsupported.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
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
        num_blocks: int = 4,
        kernel_size: int = 3,
        norm: Literal["bn", "ln"] = "bn",
    ) -> None:
        super().__init__()
        if not math.isfinite(amin) or amin <= 0:
            raise ValueError(f"amin must be positive and finite, got {amin}")
        if not math.isfinite(power) or power <= 0:
            raise ValueError(f"power must be positive and finite, got {power}")
        if top_db is not None and (not math.isfinite(top_db) or top_db < 0):
            raise ValueError(f"top_db must be non-negative and finite, got {top_db}")
        self.in_dim = in_dim
        try:
            window_fn = {"hamming": torch.hamming_window, "hann": torch.hann_window}[window]
        except KeyError as error:
            raise ValueError(f"Unsupported window: {window}") from error
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

        conv_layers: list[nn.Module] = []
        in_channels = 1
        for block_index in range(num_blocks):
            out_channels = hidden_dim * 2**block_index
            normalizer: nn.Module
            if norm == "bn":
                normalizer = nn.BatchNorm2d(out_channels)
            elif norm == "ln":
                normalizer = nn.GroupNorm(1, out_channels)
            else:
                raise ValueError(f"Unsupported norm: {norm}")
            conv_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.GELU(),
                    normalizer,
                    nn.MaxPool2d(2, ceil_mode=True),
                ]
            )
            in_channels = out_channels

        self.conv_net = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(in_channels, out_dim)

    def log_mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-waveform log-mel power relative to each waveform's peak.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Decibel-scaled mel spectrograms shaped ``(batch, mels, frames)``.
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
        return log_mel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a mono waveform batch into fixed-width embeddings.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Embeddings shaped ``(batch, out_dim)``.
        """
        mel = self.log_mel_spectrogram(x).unsqueeze(1)
        features = self.conv_net(mel)
        return self.projection(self.pool(features).flatten(1))
