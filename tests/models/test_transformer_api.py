"""Public API contracts for the approximately equivariant transformer."""

import pytest
import torch

from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
    MutualAttentionProjection,
)


def test_learned_projection_default_freezes_all_parameters() -> None:
    """Freeze assignment and feed-forward weights as one projection unit."""
    projection = LearntProjection(
        d_model=8,
        d_token=8,
        num_params=4,
        num_tokens=2,
    )

    ApproxEquivTransformer(
        projection=projection,
        d_model=8,
        conditioning_dim=4,
        num_heads=1,
        d_ff=8,
        num_tokens=2,
    )

    assert all(not parameter.requires_grad for parameter in projection.parameters())


def test_mutual_attention_projection_default_freezes_all_parameters() -> None:
    """Freeze attention queries and mapping layers as one projection unit."""
    projection = MutualAttentionProjection(d_model=8, num_params=4, num_tokens=2)

    ApproxEquivTransformer(
        projection=projection,
        d_model=8,
        conditioning_dim=4,
        num_heads=1,
        d_ff=8,
        num_tokens=2,
    )

    assert all(not parameter.requires_grad for parameter in projection.parameters())


def test_learn_projection_true_keeps_all_parameters_trainable() -> None:
    """Keep every mapping stage trainable under the production opt-in."""
    projection = LearntProjection(
        d_model=8,
        d_token=8,
        num_params=4,
        num_tokens=2,
    )

    transformer = ApproxEquivTransformer(
        projection=projection,
        d_model=8,
        conditioning_dim=4,
        num_heads=1,
        d_ff=8,
        num_tokens=2,
        learn_projection=True,
        zero_init=False,
    )

    prediction = transformer(
        torch.randn(2, 4),
        torch.rand(2, 1),
        torch.randn(2, 4),
    )
    prediction.square().mean().backward()

    gradients = [parameter.grad for parameter in projection.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients if gradient is not None)


def test_frozen_projection_transformer_overfits_fixed_batch() -> None:
    """Retain useful training capacity when projection parameters stay fixed."""
    torch.manual_seed(0)
    projection = LearntProjection(
        d_model=8,
        d_token=8,
        num_params=4,
        num_tokens=2,
    )
    transformer = ApproxEquivTransformer(
        projection=projection,
        num_layers=1,
        d_model=8,
        conditioning_dim=4,
        num_heads=1,
        d_ff=16,
        num_tokens=2,
        learn_projection=False,
        zero_init=False,
    )
    inputs = torch.randn(2, 4)
    times = torch.rand(2, 1)
    conditioning = torch.randn(2, 4)
    targets = torch.randn(2, 4)
    optimizer = torch.optim.Adam(
        (parameter for parameter in transformer.parameters() if parameter.requires_grad),
        lr=0.01,
    )

    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(
            transformer(inputs, times, conditioning), targets
        )

    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(transformer(inputs, times, conditioning), targets)
        loss.backward()
        optimizer.step()

    assert loss < initial_loss * 0.8


def test_zero_transformer_layers_raises_value_error() -> None:
    """Reject architectures that cannot train when their projection is frozen."""
    projection = LearntProjection(
        d_model=8,
        d_token=8,
        num_params=4,
        num_tokens=2,
    )

    with pytest.raises(ValueError, match="num_layers must be at least 1"):
        ApproxEquivTransformer(projection=projection, num_layers=0)
