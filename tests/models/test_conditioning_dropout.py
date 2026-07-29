"""Conditioning-dropout contracts shared by flow vector-field backbones."""

import math
from collections.abc import Callable
from typing import get_args, get_type_hints

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


def _vector_field() -> VectorField:
    return VectorField(field_dim=2, hidden_dim=4, conditioning_dim=3, num_blocks=1)


_MODEL_FACTORIES = (_conditional_residual_mlp, _approx_equiv_transformer, _vector_field)


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
    conditioning_annotation, keep_annotation = get_args(annotations["return"])

    assert annotations["z"].array_type is torch.Tensor
    assert annotations["z"].dim_str == "batch ..."
    assert conditioning_annotation.array_type is torch.Tensor
    assert conditioning_annotation.dim_str == "batch ..."
    assert keep_annotation.array_type is torch.Tensor
    assert keep_annotation.dim_str == "batch"


@pytest.mark.parametrize(
    "model_factory",
    _MODEL_FACTORIES,
    ids=["conditional-residual-mlp", "approx-equiv-transformer", "vector-field"],
)
@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 3)], ids=["rank-2", "rank-3"])
def test_apply_dropout_returned_mask_marks_exactly_the_conditioned_rows(
    model_factory: Callable[[], ConditionalResidualMLP | ApproxEquivTransformer | VectorField],
    shape: tuple[int, ...],
) -> None:
    """The returned keep mask is True on the rows that kept their conditioning.

    A co-located loss reads this mask, so a row's mask entry and its output row must never
    disagree.

    :param model_factory: Backbone constructor under test.
    :param shape: Rank-2 or rank-3 conditioning shape.
    """
    if model_factory is _vector_field and len(shape) == 3:
        pytest.skip("VectorField conditions on flat vectors only")
    torch.manual_seed(0)
    model = model_factory()
    z = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)

    dropped, keep = model.apply_dropout(z, rate=0.5)

    expected_token = model.cfg_dropout_token.reshape(-1, shape[-1])[0]
    assert keep.shape == (shape[0],)
    for row, kept in enumerate(keep.tolist()):
        expected = z[row] if kept else expected_token.expand_as(dropped[row])
        assert torch.equal(dropped[row], expected)


@pytest.mark.parametrize(
    "model_factory",
    _MODEL_FACTORIES,
    ids=["conditional-residual-mlp", "approx-equiv-transformer", "vector-field"],
)
def test_apply_dropout_at_zero_rate_keeps_every_row_and_reports_an_all_true_mask(
    model_factory: Callable[[], ConditionalResidualMLP | ApproxEquivTransformer | VectorField],
) -> None:
    """Disabled dropout keeps conditioning intact and says so in the mask.

    :param model_factory: Backbone constructor under test.
    """
    model = model_factory()
    z = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    dropped, keep = model.apply_dropout(z, rate=0.0)

    assert torch.equal(dropped, z)
    assert torch.equal(keep, torch.ones(2, dtype=torch.bool))
