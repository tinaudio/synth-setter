"""Hydra contracts for removed synthetic-synth training paths."""

import pytest
from hydra import compose, initialize_config_module
from hydra.errors import MissingConfigException


@pytest.mark.parametrize("datamodule", ["fm", "kosc", "ksin", "ksin_ood"])
def test_removed_datamodule_cannot_be_composed(datamodule: str) -> None:
    """Legacy synthetic datamodule names are no longer accepted.

    :param datamodule: Removed Hydra datamodule group member.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        with pytest.raises(MissingConfigException):
            compose(
                config_name="train.yaml",
                overrides=[f"datamodule={datamodule}", "model=ffn"],
            )


@pytest.mark.parametrize("experiment", ["fm/base", "kosc/base", "ksin/base", "ksin_ood/base"])
def test_removed_experiment_cannot_be_composed(experiment: str) -> None:
    """Legacy synthetic experiment names are no longer accepted.

    :param experiment: Removed Hydra experiment group member.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        with pytest.raises(MissingConfigException):
            compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
