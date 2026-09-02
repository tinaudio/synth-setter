"""Projection encoder for fixed-width conditioning vectors."""

import torch
from beartype import beartype
from jaxtyping import jaxtyped
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

    @jaxtyped(typechecker=beartype)
    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore projection metadata absent from pre-slot pickles.

        :param state: Serialized ``torch.nn.Module`` state.
        """
        super().__setstate__(state)
        if not hasattr(self, "n_conditioning_outputs"):
            self.n_conditioning_outputs = 1
        if not hasattr(self, "d_model"):
            self.d_model = self.projection.out_features // self.n_conditioning_outputs

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
        if self.n_conditioning_outputs == 1:
            return projected
        return projected.unflatten(-1, (self.n_conditioning_outputs, self.d_model))
