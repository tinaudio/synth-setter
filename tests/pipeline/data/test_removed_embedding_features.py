"""Removed embedding features stay unavailable without affecting parameter language."""

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.language_projection import LanguageParameterProjection
from synth_setter.pipeline.data.param_language import describe_fields, save_param_language
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig


@pytest.mark.parametrize("embedding", ["param_shift", "t5gemma"])
def test_removed_embedding_selection_is_rejected(embedding: str) -> None:
    """The add-embeddings boundary rejects removed registry names.

    :param embedding: Removed registry key.
    """
    with pytest.raises(ValidationError, match="must each be one of"):
        AddEmbeddingsConfig(lance_uri="unused.lance", embeddings=(embedding,))


def test_parameter_language_artifact_still_initializes_projection(tmp_path: Path) -> None:
    """Parameter-language artifacts remain usable after removing row embeddings.

    :param tmp_path: Isolated artifact directory.
    """
    field_count = len(describe_fields("surge_4", "surge_4"))
    vectors = np.full((field_count, 128), 1 / np.sqrt(128), dtype=np.float32)
    artifact = tmp_path / "param_language.npz"
    save_param_language(artifact, vectors, "surge_4", "surge_4")
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )

    projection.initialize_embeddings()
    tokens = projection.param_to_token(
        torch.zeros(2, param_specs["surge_4"].encoded_width)
    )

    assert tokens.shape == (2, field_count, 16)
