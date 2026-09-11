"""Hydra composition rejects unsupported synthetic training paths."""

import pytest
from hydra import compose, initialize_config_module
from hydra.errors import MissingConfigException


@pytest.mark.parametrize("datamodule", ["fm", "kosc", "ksin", "ksin_ood"])
def test_removed_datamodule_cannot_be_composed(datamodule: str) -> None:
    """Reject unsupported synthetic datamodule names.

    :param datamodule: Unsupported Hydra datamodule group member.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        with pytest.raises(MissingConfigException):
            compose(
                config_name="train.yaml",
                overrides=[f"datamodule={datamodule}", "model=ffn"],
            )


def test_removed_non_householder_pyfdn_synth_cannot_be_composed() -> None:
    """Reject the pyFDN identity whose decoded feedback can lose orthogonality."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        with pytest.raises(MissingConfigException):
            compose(
                config_name="train.yaml",
                overrides=[
                    "datamodule=pyfdn",
                    "synth=pyfdn_n8_mono",
                    "model=vst_flow",
                ],
            )


@pytest.mark.parametrize("experiment", ["fm/base", "kosc/base", "ksin/base", "ksin_ood/base"])
def test_removed_experiment_cannot_be_composed(experiment: str) -> None:
    """Reject unsupported synthetic experiment names.

    :param experiment: Unsupported Hydra experiment group member.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        with pytest.raises(MissingConfigException):
            compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
