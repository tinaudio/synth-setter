import math
from typing import Literal

import torch
import torch.nn as nn


def make_sin_pos_enc(max_len, d_enc):
    pe = torch.zeros(max_len, d_enc)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_enc, 2, dtype=torch.float32) * (-math.log(10000.0) / d_enc)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add an extra dimension for batch size compatibility.
    pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    return pe


class PosEnc(nn.Module):
    def __init__(
        self,
        d_enc: int,
        max_len: int,
        pos_enc_type: Literal["sin", "learned"] = "sin",
    ):
        super().__init__()

        if pos_enc_type == "sin":
            self.register_buffer("pe", make_sin_pos_enc(max_len, d_enc))
        elif pos_enc_type == "learned":
            self.pe = nn.Parameter(torch.randn(1, max_len, d_enc) / math.sqrt(d_enc))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :]
        return x


class EmbeddingPool(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        d_model: int,
        num_heads: int,
        pos_enc: Literal["sin", "learned"] = "sin",
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = PosEnc(embed_dim, 42, pos_enc)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.residual = nn.Linear(embed_dim, d_model, bias=False)

    def forward(self, embed: torch.Tensor):
        embed = embed.permute(0, 2, 1)

        embed = self.positional_encoding(embed)
        embed = self.ffn(embed) + self.residual(embed)
        query = self.query.repeat(embed.shape[0], 1, 1)
        embed, _ = self.attn(query=query, key=embed, value=embed)

        return embed.squeeze(1)
