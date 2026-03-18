from typing import Literal

import torch
import torch.nn as nn

from src.models.components.cnn import ResidualEncoder
from src.models.components.transformer import SinusoidalEncoding


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


class ResidualMLP(nn.Sequential):
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
    ):
        super().__init__()

        self.encoder = ResidualEncoder(
            in_dim, channels, hidden_dim, encoder_blocks, kernel_size, norm
        )
        self.trunk = ResidualMLP(hidden_dim, hidden_dim, out_dim, trunk_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.trunk(z)
