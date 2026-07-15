"""Residual MLP backbones (plain, FiLM-conditioned, spectral, and CNN-fronted variants)."""

from typing import Literal

import torch
import torch.nn as nn

from synth_setter.models.components.cnn import LogMelEncoder, ResidualEncoder
from synth_setter.models.components.transformer import SinusoidalEncoding


class ResidualMLPBlock(nn.Module):
    """Two-layer MLP with LayerNorm front and a learned linear shortcut."""

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


class ResidualMLP(nn.Sequential):
    """Stack of ``ResidualMLPBlock`` layers for predicting parameter vectors from features."""

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 1024,
        out_dim: int = 16,
        num_blocks: int = 6,
    ):
        layers = [
            ResidualMLPBlock(
                in_dim if i == 0 else hidden_dim,
                out_dim if i == (num_blocks - 1) else hidden_dim,
            )
            for i in range(num_blocks)
        ]

        super().__init__(*layers)


class ConditionalResidualMLPBlock(nn.Module):
    """Residual MLP block with FiLM-style (gate, bias, scale) conditioning from a context."""

    def __init__(
        self,
        d_model: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.cond = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model * 3))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        res = x

        x = self.norm(x)
        a, g, b = self.cond(c).chunk(3, dim=-1)

        x = g * self.norm(x) + b
        x = self.net(x)
        return res + a * x


class ConditionalResidualMLP(nn.Module):
    """Conditional residual MLP for flow matching with timestep and audio context."""

    def __init__(
        self,
        n_params: int = 100,
        d_model: int = 1024,
        d_enc: int = 256,
        conditioning_dim: int = 512,
        num_layers: int = 6,
        time_encoding: Literal["sinusoidal", "scalar"] = "sinusoidal",
    ):
        super().__init__()

        self.cfg_dropout_token = nn.Parameter(torch.randn(1, 1, conditioning_dim))

        self.in_proj = nn.Linear(n_params, d_model)
        self.out_proj = nn.Linear(d_model, n_params)

        layers = [ConditionalResidualMLPBlock(d_model) for i in range(num_layers)]
        self.net = nn.Sequential(*layers)

        if time_encoding == "scalar":
            self.time_encoding = nn.Identity()
            d_enc = 1
        else:
            self.time_encoding = SinusoidalEncoding(d_enc)

        self.conditioning_ffn = nn.Sequential(
            nn.Linear(conditioning_dim + d_enc, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def penalty(self):
        return 0.0

    def apply_dropout(self, z: torch.tensor, rate: float = 0.1):
        if rate == 0.0:
            return z

        dropout_mask = torch.rand(z.shape[0], device=z.device) > rate
        if z.ndim == 2:
            dropout_mask = dropout_mask[..., None]
            dropout_token = self.cfg_dropout_token[0]
        elif z.ndim == 3:
            dropout_mask = dropout_mask[..., None, None]
            dropout_token = self.cfg_dropout_token
        else:
            raise ValueError("unexpected z shape")

        return torch.where(dropout_mask, z, dropout_token)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        if c is None:
            c = self.cfg_dropout_token[0].expand(x.shape[0], -1)

        t = self.time_encoding(t)

        if c.ndim == 3:
            t = t.unsqueeze(1).repeat(1, c.shape[1], 1)

        z = torch.cat([c, t], dim=-1)
        z = self.conditioning_ffn(z)

        x = self.in_proj(x)
        for i, layer in enumerate(self.net):
            if z.ndim == 2:
                z_ = z
            else:
                z_ = z[:, i]
            x = layer(x, z_)
        x = self.out_proj(x)

        return x


class SpectralResidualMLP(ResidualMLP):
    """ResidualMLP fed the magnitude rFFT of the input waveform."""

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 1024,
        out_dim: int = 16,
        num_blocks: int = 6,
    ):
        true_in_dim = in_dim // 2 + 1
        super().__init__(true_in_dim, hidden_dim, out_dim, num_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(x, norm="forward")
        X = torch.abs(X)
        return super().forward(X)


class CNNResidualMLP(nn.Module):
    """Predict parameters with a global-FFT CNN encoder and residual MLP trunk.

    :param in_dim: Expected waveform length in samples.
    :param channels: Channel count in the encoder's first convolutional block.
    :param encoder_blocks: Number of convolution and downsampling blocks.
    :param trunk_blocks: Number of residual MLP blocks.
    :param hidden_dim: Encoder output and MLP hidden width.
    :param out_dim: Number of predicted parameters.
    :param kernel_size: Encoder convolution kernel size.
    :param norm: Encoder normalization type.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        channels: int = 16,
        encoder_blocks: int = 4,
        trunk_blocks: int = 5,
        hidden_dim: int = 2048,
        out_dim: int = 16,
        kernel_size: int = 7,
        norm: Literal["bn", "ln"] = "bn",
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(
            in_dim, channels, hidden_dim, encoder_blocks, kernel_size, norm
        )
        self.trunk = ResidualMLP(hidden_dim, hidden_dim, out_dim, trunk_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict parameters from a batch of mono waveforms.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Parameter predictions shaped ``(batch, out_dim)``.
        """
        z = self.encoder(x)
        return self.trunk(z)


class LogMelCNNResidualMLP(nn.Module):
    """Predict normalized parameters with a log-mel CNN encoder and residual MLP trunk.

    :param in_dim: Expected waveform length in samples.
    :param channels: Channel count in the encoder's first convolutional block.
    :param encoder_blocks: Number of convolution and pooling blocks.
    :param trunk_blocks: Number of residual MLP blocks.
    :param hidden_dim: Encoder output and MLP hidden width.
    :param out_dim: Number of predicted parameters.
    :param kernel_size: Encoder convolution kernel size.
    :param norm: Encoder normalization type.
    :param sample_rate: Waveform sample rate in Hz.
    :param center: Whether STFT frames are centered on timestamps.
    :param f_min: Lowest mel-filter frequency in Hz.
    :param f_max: Highest mel-filter frequency in Hz; ``None`` selects Nyquist.
    :param n_fft: Fourier transform size.
    :param hop_length: Fourier frame stride.
    :param n_mels: Mel-filter count.
    :param pad_mode: Centering pad mode.
    :param power: Magnitude exponent.
    :param mel_norm: Mel-filter normalization.
    :param mel_scale: Mel-frequency conversion formula.
    :param window: Fourier window function.
    :param amin: Positive logarithm floor.
    :param top_db: Optional dynamic-range limit.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        channels: int = 16,
        encoder_blocks: int = 4,
        trunk_blocks: int = 5,
        hidden_dim: int = 2048,
        out_dim: int = 16,
        kernel_size: int = 7,
        norm: Literal["bn", "ln"] = "bn",
        *,
        sample_rate: int,
        center: bool = True,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_fft: int | None = None,
        hop_length: int | None = None,
        n_mels: int = 128,
        pad_mode: Literal["constant", "reflect"] = "constant",
        power: float = 2.0,
        mel_norm: Literal["slaney"] | None = "slaney",
        mel_scale: Literal["htk", "slaney"] = "slaney",
        window: Literal["hamming", "hann"] = "hamming",
        amin: float = 1e-10,
        top_db: float | None = 80.0,
    ) -> None:
        super().__init__()
        self.encoder = LogMelEncoder(
            in_dim=in_dim,
            hidden_dim=channels,
            out_dim=hidden_dim,
            sample_rate=sample_rate,
            center=center,
            f_min=f_min,
            f_max=f_max,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            pad_mode=pad_mode,
            power=power,
            mel_norm=mel_norm,
            mel_scale=mel_scale,
            window=window,
            amin=amin,
            top_db=top_db,
            num_blocks=encoder_blocks,
            kernel_size=kernel_size,
            norm=norm,
        )
        self.trunk = ResidualMLP(hidden_dim, hidden_dim, out_dim, trunk_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict parameters from a batch of mono waveforms.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Normalized parameter predictions shaped ``(batch, out_dim)``.
        """
        return torch.sigmoid(self.trunk(self.encoder(x)))
