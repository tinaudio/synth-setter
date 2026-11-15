"""Projection encoder for fixed-width conditioning vectors."""

import torch
from torch import nn


class VectorProjection(nn.Module):
    """Project one fixed-width embedding vector into shared or per-layer conditioning."""

    def __init__(self, input_dim: int, d_model: int, n_conditioning_outputs: int = 1) -> None:
        """Initialize the fixed-width projection.

        :param input_dim: Expected input embedding width.
        :param d_model: Returned conditioning width.
        :param n_conditioning_outputs: Conditioning slots to emit; one per consuming
            field layer, or ``1`` for a single shared conditioning vector.
        :raises ValueError: ``n_conditioning_outputs`` is not positive.
        """
        super().__init__()
        if n_conditioning_outputs < 1:
            raise ValueError(
                f"n_conditioning_outputs must be positive, got {n_conditioning_outputs}"
            )
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_conditioning_outputs = n_conditioning_outputs
        self.projection = nn.Linear(input_dim, n_conditioning_outputs * d_model)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project ``(batch, input_dim)`` embeddings to the configured conditioning rank.

        :param embedding: Fixed-width vector batch.
        :returns: ``(batch, d_model)`` for a single slot, otherwise
            ``(batch, n_conditioning_outputs, d_model)``.
        :raises ValueError: If the input is not rank two or has the configured width.
        """
        if embedding.ndim != 2 or embedding.shape[1] != self.input_dim:
            raise ValueError(
                f"expected embedding shape (batch, {self.input_dim}), got {tuple(embedding.shape)}"
            )
        projected = self.projection(embedding)
        n_conditioning_outputs = getattr(self, "n_conditioning_outputs", 1)
        if n_conditioning_outputs == 1:
            return projected
        return projected.unflatten(-1, (n_conditioning_outputs, self.d_model))
