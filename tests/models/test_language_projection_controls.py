"""Matched ablations distinguish field identity from language semantics."""

from typing import Literal

import pytest
import torch

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.language_projection import LanguageParameterProjection
from synth_setter.models.components.transformer import GroupedParameterProjection


@pytest.mark.parametrize("source", ["random", "learned"])
def test_control_vectors_require_no_language_artifact(
    source: Literal["random", "learned"],
) -> None:
    """Identity controls can train without a text model or metadata file.

    :param source: Non-language treatment.
    """
    projection = LanguageParameterProjection(16, "surge_4", "surge_4", embedding_source=source)
    tokens = projection.param_to_token(torch.ones(2, param_specs["surge_4"].encoded_width))
    assert torch.isfinite(tokens).all()
    assert projection.language_embeddings.requires_grad == (source == "learned")


def test_learned_control_optimizer_updates_field_vectors() -> None:
    """Trainable identity vectors receive updates after the residual branch opens."""
    projection = LanguageParameterProjection(16, "surge_4", "surge_4", embedding_source="learned")
    before = projection.language_embeddings.detach().clone()
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.01)
    x = torch.randn(2, param_specs["surge_4"].encoded_width)
    for _ in range(2):
        optimizer.zero_grad()
        projection.param_to_token(x).square().mean().backward()
        optimizer.step()
    assert not torch.equal(projection.language_embeddings, before)


def test_language_construction_preserves_common_backbone_rng() -> None:
    """Added fusion layers do not change downstream baseline weight initialization."""
    torch.manual_seed(12)
    GroupedParameterProjection(16, "surge_4")
    baseline_next = torch.randn(5)
    torch.manual_seed(12)
    LanguageParameterProjection(16, "surge_4", "surge_4")
    torch.testing.assert_close(torch.randn(5), baseline_next)
