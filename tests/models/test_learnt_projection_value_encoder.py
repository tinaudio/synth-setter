"""Behaviour of the optional value encoder injected into ``LearntProjection``."""

import pytest
import torch
from torch import nn

from synth_setter.models.components.fourier_number import FourierNumberEmbedder
from synth_setter.models.components.transformer import LearntProjection

BASELINE_STATE_DICT_KEYS = {
    "_assignment",
    "_in_projection",
    "_out_projection",
    "initial_ffn.0.weight",
    "initial_ffn.0.bias",
    "initial_ffn.2.weight",
    "initial_ffn.2.bias",
}


def build_projection(
    value_encoder: nn.Module | None = None, initial_ffn: bool = True
) -> LearntProjection:
    """Build a small projection sharing the production token/param contract.

    :param value_encoder: Optional module encoding parameter values.
    :param initial_ffn: Whether the pre-assignment FFN head is built.
    :returns: Configured projection.
    """
    return LearntProjection(
        d_model=8,
        d_token=8,
        num_params=6,
        num_tokens=3,
        initial_ffn=initial_ffn,
        final_ffn=False,
        value_encoder=value_encoder,
    )


def test_learnt_projection_without_value_encoder_keeps_baseline_state_dict_keys() -> None:
    """The default path adds no tensors, so checkpoint layout is unchanged."""
    projection = build_projection()

    assert set(projection.state_dict()) == BASELINE_STATE_DICT_KEYS


def test_learnt_projection_without_value_encoder_loads_baseline_checkpoint_strictly() -> None:
    """A checkpoint written before the hook existed still loads strictly."""
    checkpoint = build_projection().state_dict()

    build_projection().load_state_dict(checkpoint, strict=True)


def test_learnt_projection_without_value_encoder_scales_linearly_with_values() -> None:
    """Without an encoder the value path stays the original linear ray."""
    projection = build_projection(initial_ffn=False)
    params = torch.rand(2, 6)

    torch.testing.assert_close(
        projection.param_to_token(2 * params), 2 * projection.param_to_token(params)
    )


def test_learnt_projection_with_value_encoder_breaks_linearity_in_values() -> None:
    """An injected encoder actually reshapes the value path, not just the wiring."""
    projection = build_projection(FourierNumberEmbedder(features=8, dim=16), initial_ffn=False)
    params = torch.rand(2, 6)

    doubled = projection.param_to_token(2 * params)

    assert not torch.allclose(doubled, 2 * projection.param_to_token(params))


def test_learnt_projection_with_value_encoder_returns_token_shape() -> None:
    """Encoding leaves the (B, K, D) token contract intact."""
    projection = build_projection(FourierNumberEmbedder(features=8, dim=16))

    assert projection.param_to_token(torch.rand(2, 6)).shape == (2, 3, 8)


def test_learnt_projection_with_value_encoder_decodes_back_to_parameter_width() -> None:
    """The unchanged decoder still returns one value per parameter."""
    projection = build_projection(FourierNumberEmbedder(features=8, dim=16))

    tokens = projection.param_to_token(torch.rand(2, 6))

    assert projection.token_to_param(tokens).shape == (2, 6)


def test_learnt_projection_backward_reaches_value_encoder_and_assignment() -> None:
    """Gradients flow to both the encoder and the assignment matrix."""
    projection = build_projection(FourierNumberEmbedder(features=8, dim=16))

    projection.token_to_param(projection.param_to_token(torch.rand(2, 6))).sum().backward()

    encoder = projection.value_encoder
    assert isinstance(encoder, FourierNumberEmbedder)
    assert encoder.projection.weight.grad is not None
    assert projection.assignment.grad is not None


def test_learnt_projection_value_encoder_wrong_width_raises_value_error() -> None:
    """A mis-sized encoder fails with a named contract, not a deep tensor error."""
    projection = build_projection(FourierNumberEmbedder(features=5, dim=16))

    with pytest.raises(ValueError, match="d_token"):
        projection.param_to_token(torch.rand(2, 6))
