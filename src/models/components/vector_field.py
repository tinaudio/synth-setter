import torch
import torch.nn as nn


class AdaptiveLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, conditioning_dim: int, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.shift = nn.Linear(conditioning_dim, dim)
        self.scale = nn.Linear(conditioning_dim, dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        shift = self.shift(z)
        scale = self.scale(z)
        x = super().forward(x)
        return x * scale + shift


class VectorFieldBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        conditioning_dim: int | None = None,
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

        self.conditioning = conditioning_dim is not None
        if self.conditioning:
            self.norm = AdaptiveLayerNorm(in_dim, conditioning_dim)
        else:
            self.norm = nn.LayerNorm(in_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.residual = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditioning:
            assert z is not None, "z must be provided for conditional vector field block."
            y = self.norm(x, z)
        else:
            y = self.norm(x)

        y = self.net(y)
        res = self.residual(x)

        return res + y


class VectorField(nn.Module):
    def __init__(self, field_dim: int, hidden_dim: int, conditioning_dim: int, num_blocks: int):
        super().__init__()

        self.field_dim = field_dim
        self.input = nn.Linear(field_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                VectorFieldBlock(
                    hidden_dim,
                    conditioning_dim + 1,
                    out_dim=None if i < num_blocks - 1 else field_dim,
                )
                for i in range(num_blocks)
            ]
        )
        # We use this to replace conditioning vectors for some proportion of training
        # inputs. This enables us to use classifier free guidance at inference.
        self.cfg_dropout_token = nn.Parameter(torch.randn(1, conditioning_dim))

    def apply_dropout(self, z: torch.tensor, rate: float = 0.1):
        if rate == 0.0:
            return z

        dropout_mask = torch.rand(z.shape[0], 1, device=z.device) > rate
        return z.where(dropout_mask, self.cfg_dropout_token)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ):
        if conditioning is None:
            conditioning = self.cfg_dropout_token.expand(x.shape[0], -1)

        z = torch.cat((conditioning, t), dim=-1)
        y = self.input(x)

        for block in self.blocks:
            y = block(y, z)

        return y
