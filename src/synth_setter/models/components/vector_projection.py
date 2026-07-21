"""Projection encoder for fixed-width conditioning vectors."""

import torch
from torch import nn


class VectorProjection(nn.Module):
    """Project one fixed-width embedding vector per batch row."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        """Initialize the fixed-width projection.

        :param input_dim: Expected input embedding width.
        :param d_model: Returned conditioning width.
        """
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, d_model)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project ``(batch, input_dim)`` embeddings to ``(batch, d_model)``.

        :param embedding: Fixed-width vector batch.
        :returns: Projected conditioning vectors.
        :raises ValueError: If the input is not rank two or has the configured width.
        """
        if embedding.ndim != 2 or embedding.shape[1] != self.input_dim:
            raise ValueError(
                f"expected embedding shape (batch, {self.input_dim}), got {tuple(embedding.shape)}"
            )
        return self.projection(embedding)
