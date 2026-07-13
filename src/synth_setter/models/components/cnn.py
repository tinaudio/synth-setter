"""Convolutional and residual building blocks used by the spectrum encoders."""

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
    :param num_blocks: Number of convolution and pooling blocks.
    :param kernel_size: Height and width of each convolutional kernel.
    :param norm: Normalization applied after each convolution.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        sample_rate: int,
        num_blocks: int = 4,
        kernel_size: int = 3,
        norm: Literal["bn", "ln"] = "bn",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=mel_n_fft(sample_rate),
            hop_length=mel_hop_length(sample_rate),
            n_mels=MEL_N_MELS,
            window_fn=torch.hamming_window,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
        )

        conv_layers: list[nn.Module] = []
        in_channels = 1
        for block_index in range(num_blocks):
            out_channels = hidden_dim * 2**block_index
            normalizer: nn.Module
            if norm == "bn":
                normalizer = nn.BatchNorm2d(out_channels)
            else:
                normalizer = nn.GroupNorm(1, out_channels)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a mono waveform batch into fixed-width embeddings.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Embeddings shaped ``(batch, out_dim)``.
        :raises ValueError: If the waveform shape differs from the configured input length.
        """
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected waveform shape (batch, {self.in_dim}), got {tuple(x.shape)}"
            )
        mel = torch.log1p(self.mel(x)).unsqueeze(1)
        features = self.conv_net(mel)
        return self.projection(self.pool(features).flatten(1))
