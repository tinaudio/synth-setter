"""Removed embedding features stay unavailable without affecting parameter language."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.errors import MissingConfigException
from pydantic import ValidationError

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.language_projection import LanguageParameterProjection
from synth_setter.pipeline.data.add_embeddings import main
from synth_setter.pipeline.data.param_language import describe_fields, save_param_language
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig


@pytest.mark.parametrize("embedding", ["param_shift", "t5gemma"])
def test_removed_embedding_selection_is_rejected(embedding: str) -> None:
    """The add-embeddings boundary rejects removed registry names.

    :param embedding: Removed registry key.
    """
    with pytest.raises(ValidationError, match="must each be one of"):
        AddEmbeddingsConfig(lance_uri="unused.lance", embeddings=(embedding,))


@pytest.mark.parametrize("embedding", ["param_shift", "t5gemma"])
def test_add_embeddings_cli_with_removed_selection_exits_one(
    embedding: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The production CLI rejects removed registry names.

    :param embedding: Removed registry key.
    :param monkeypatch: Fixture replacing command-line arguments.
    :param tmp_path: Isolated Hydra output directory.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            "logger=[]",
            "lance_uri=unused.lance",
            f"embeddings=[{embedding}]",
            f"hydra.run.dir={tmp_path / embedding}",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


@pytest.mark.parametrize("config_name", ["train", "eval"])
def test_removed_t5gemma_conditioning_profile_does_not_compose(config_name: str) -> None:
    """Training and evaluation cannot select the removed conditioning profile.

    :param config_name: Production root config under test.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        with pytest.raises(MissingConfigException, match="t5gemma"):
            compose(config_name=config_name, overrides=["conditioning=t5gemma"])


def test_parameter_language_artifact_still_initializes_projection(tmp_path: Path) -> None:
    """Parameter-language artifacts initialize independently of embedding registry entries.

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

    torch.testing.assert_close(projection.language_embeddings, torch.from_numpy(vectors))
    assert tokens.shape == (2, field_count, 16)
    assert torch.isfinite(tokens).all()
