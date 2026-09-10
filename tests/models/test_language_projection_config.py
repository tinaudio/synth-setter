"""Opt-in language projection composes in both model consumers."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from synth_setter.models.components.language_projection import LanguageParameterProjection


@pytest.mark.parametrize("consumer", ["flow", "slap"])
def test_language_config_instantiates_projection_for_consumer(
    consumer: str, tmp_path: Path
) -> None:
    """Shipped Hydra selectors construct the shared projection without loading text weights.

    :param consumer: Training consumer whose projection is instantiated.
    :param tmp_path: Dataset location; no artifact is opened during construction.
    """
    overrides = (
        ["experiment=surge/flow_simple", "model/projection=language"]
        if consumer == "flow"
        else ["experiment=surge/slap_language"]
    )
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train", overrides=[*overrides, f"datamodule.dataset_root={tmp_path}"]
        )
    config = (
        cfg.model.vector_field.projection
        if consumer == "flow"
        else cfg.model.param_encoder.encoder.projection
    )
    projection = instantiate(config)
    assert isinstance(projection, LanguageParameterProjection)
    assert projection.embedding_dim == 128
