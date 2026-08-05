"""Convolutional and residual building blocks used by spectrum encoders.

Example::

    backbone = MelCNN(16, 256)
    embeddings = backbone(torch.zeros(2, 1, 128, 401))
"""

from typing import Final, Literal

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from synth_setter.models.components.spec_encoder import LogMelFrontend

_BATCH_GRID_SHAPE: Final = "batch channels mels frames"
_BATCH_EMBEDDING_SHAPE: Final = "batch embedding"
_BATCH_AUDIO_SHAPE: Final = "batch samples"


@jaxtyped(typechecker=beartype)
def _make_conv_stack(
    input_channels: int,
    hidden_dim: int,
    num_blocks: int,
    kernel_size: int,
    norm: Literal["bn", "ln"],
) -> tuple[nn.Sequential, int]:
    """Build the downsampling CNN and return its final channel count.

    :param input_channels: Channel count of the incoming spectrogram grid.
    :param hidden_dim: Channel count in the first block.
    :param num_blocks: Number of convolution and pooling blocks.
    :param kernel_size: Height and width of each convolutional kernel.
    :param norm: Normalization applied after each convolution.
    :returns: Sequential CNN and its final channel count.
    """
    layers: list[nn.Module] = []
    in_channels = input_channels
    for block_index in range(num_blocks):
        out_channels = hidden_dim * 2**block_index
        normalizer: nn.Module = (
            nn.BatchNorm2d(out_channels) if norm == "bn" else nn.GroupNorm(1, out_channels)
        )
        layers.extend(
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
    return nn.Sequential(*layers), in_channels


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


class MelCNN(nn.Module):
    """Reduce a spectrogram grid to one embedding per row with a pooled 2-D CNN.

    :param hidden_dim: Channel count in the first convolutional block.
    :param out_dim: Width of the returned embedding.
    :param input_channels: Channel count of the incoming spectrogram grid.
    :param num_blocks: Number of convolution and pooling blocks.
    :param kernel_size: Height and width of each convolutional kernel.
    :param norm: Normalization applied after each convolution.
    :raises ValueError: If ``kernel_size`` is non-positive.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        *,
        input_channels: int = 1,
        num_blocks: int = 4,
        kernel_size: int = 3,
        norm: Literal["bn", "ln"] = "bn",
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.conv_net, final_channels = _make_conv_stack(
            input_channels, hidden_dim, num_blocks, kernel_size, norm
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(final_channels, out_dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, _BATCH_GRID_SHAPE]
    ) -> Float[Tensor, _BATCH_EMBEDDING_SHAPE]:
        """Pool a spectrogram grid into fixed-width embeddings.

        :param x: Spectrogram grids shaped ``(batch, channels, mels, frames)``.
        :returns: Embeddings shaped ``(batch, out_dim)``.
        """
        return self.projection(self.pool(self.conv_net(x)).flatten(1))


class _FusedLogMelEncoder(nn.Module):
    """Restore encoders pickled before #2755 split the fused log-mel encoder.

    ``VSTFlowMatchingModule`` pickles the encoder instance into
    ``hyper_parameters``, so checkpoints written before the split still name the
    fused class and cannot load without it. Its state is the flattened union of
    :class:`~synth_setter.models.components.spec_encoder.LogMelFrontend` and
    :class:`MelCNN`, which is why the halves run against ``self`` here: keeping
    the flat layout is what lets a legacy ``state_dict`` load unchanged.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, _BATCH_AUDIO_SHAPE]
    ) -> Float[Tensor, _BATCH_EMBEDDING_SHAPE]:
        """Encode a mono waveform batch into fixed-width embeddings.

        :param x: Waveforms shaped ``(batch, samples)``.
        :returns: Embeddings shaped ``(batch, out_dim)``.
        """
        return MelCNN.forward(self, LogMelFrontend.forward(self, x))


@jaxtyped(typechecker=beartype)
def __getattr__(name: str) -> object:
    """Resolve the pre-#2755 encoder name so old checkpoints unpickle.

    :param name: Requested module attribute.
    :returns: Compatibility target for the renamed class.
    :raises AttributeError: If ``name`` is not the renamed class.
    """
    if name == "LogMelEncoder":
        return _FusedLogMelEncoder
    raise AttributeError(name)
