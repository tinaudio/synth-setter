"""Conditioning-dropout contracts shared by flow vector-field backbones."""

import math
from collections.abc import Callable
from typing import get_type_hints

import pytest
import torch

from synth_setter.models.components.residual_mlp import ConditionalResidualMLP
from synth_setter.models.components.transformer import ApproxEquivTransformer, LearntProjection
from synth_setter.models.components.vector_field import VectorField


def _conditional_residual_mlp() -> ConditionalResidualMLP:
    return ConditionalResidualMLP(
        n_params=2,
        d_model=4,
        conditioning_dim=3,
        num_layers=1,
        time_encoding="scalar",
    )


def _approx_equiv_transformer() -> ApproxEquivTransformer:
    return ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=4,
            d_token=4,
            num_params=2,
            num_tokens=2,
        ),
        num_layers=1,
        d_model=4,
        conditioning_dim=3,
        num_heads=1,
        d_ff=4,
        num_tokens=2,
        learn_projection=True,
        pe_type="none",  # pyright: ignore[reportArgumentType]
    )


@pytest.mark.parametrize(
    "model_type",
    [ConditionalResidualMLP, ApproxEquivTransformer, VectorField],
)
def test_apply_dropout_signature_uses_jaxtyping_tensor_input_and_return(
    model_type: type[ConditionalResidualMLP] | type[ApproxEquivTransformer] | type[VectorField],
) -> None:
    """Every caller-facing dropout method exposes a shaped Tensor contract.

    :param model_type: Backbone class whose public annotations are inspected.
    """
    annotations = get_type_hints(model_type.apply_dropout)

    assert annotations["z"].array_type is torch.Tensor
    assert annotations["z"].dim_str == "batch ..."
    assert annotations["return"].array_type is torch.Tensor
    assert annotations["return"].dim_str == "batch ..."


@pytest.mark.parametrize(
    "model_factory",
    [_conditional_residual_mlp, _approx_equiv_transformer],
    ids=["conditional-residual-mlp", "approx-equiv-transformer"],
)
@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 3)], ids=["rank-2", "rank-3"])
def test_apply_dropout_with_caller_mask_keeps_and_drops_requested_rows(
    model_factory: Callable[[], ConditionalResidualMLP | ApproxEquivTransformer],
    shape: tuple[int, ...],
) -> None:
    """Caller masks select conditioned and CFG-token rows at both supported ranks.

    :param model_factory: Backbone constructor under test.
    :param shape: Rank-2 or rank-3 conditioning shape.
    """
    model = model_factory()
    z = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)

    dropped = model.apply_dropout(z, rate=0.5, keep_mask=torch.tensor([True, False]))

    assert torch.equal(dropped[0], z[0])
    expected_token = model.cfg_dropout_token.reshape(-1, shape[-1])[0]
    assert torch.equal(dropped[1], expected_token.expand_as(dropped[1]))
