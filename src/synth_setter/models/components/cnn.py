from typing import Literal

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(-1, -2)).transpose(-1, -2)


class ResidualBlock(nn.Module):
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
