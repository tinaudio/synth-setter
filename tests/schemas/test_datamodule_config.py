"""Behavioural tests for the ``DataModuleConfig`` pydantic model.

Hydra ``???`` mandatory-override sentinels survive ``to_container(resolve=False)``
as the literal string ``"???"``; the schema sees them as ordinary extras
and accepts them, deferring the failure to ``hydra.utils.instantiate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.resources import configs_dir
from synth_setter.schemas.datamodule_config import DataModuleConfig
from synth_setter.schemas.paths_config import PathsConfig
from tests.schemas.conftest import compose_subtree

_DATAMODULE_CONFIG_DIR = Path(str(configs_dir() / "datamodule"))


def _all_datamodule_config_names() -> list[str]:
    """Return the YAML stem of every direct datamodule config under ``configs/datamodule/``.

    :return: Sorted list of YAML stems found in ``configs/datamodule/``.
    """
    names = sorted(p.stem for p in _DATAMODULE_CONFIG_DIR.glob("*.yaml"))
    assert names, (
        f"no datamodule YAMLs found under {_DATAMODULE_CONFIG_DIR} — has the layout changed?"
    )
    return names


class TestDataModuleConfigAcceptsEveryConfig:
    """Every shipped datamodule YAML must validate against ``DataModuleConfig``."""

    @pytest.mark.parametrize("datamodule_name", _all_datamodule_config_names())
    def test_datamodule_yaml_validates(self, datamodule_name: str) -> None:
        """The composed ``datamodule`` subtree validates as ``DataModuleConfig``.

        :param datamodule_name: Parametrized YAML stem under ``configs/datamodule/``.
        """
        datamodule_subtree = compose_subtree("datamodule", datamodule_name)
        parsed = DataModuleConfig.model_validate(datamodule_subtree)
        assert parsed.target_

    def test_target_field_typed(self) -> None:
        """``_target_`` lands on ``target_`` with the expected datamodule path."""
        datamodule_subtree = compose_subtree("datamodule", "ksin")
        parsed = DataModuleConfig.model_validate(datamodule_subtree)
        assert parsed.target_.endswith("KSinDataModule")


class TestPathsConfigResolvedInterpolation:
    """A real resolved ``PROJECT_ROOT`` must round-trip through ``NonBlankStr``."""

    def test_paths_resolved_with_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``${oc.env:PROJECT_ROOT}`` resolves to a real path and validates as non-blank.

        :param monkeypatch: Pytest fixture used to set ``PROJECT_ROOT``.
        """
        monkeypatch.setenv("PROJECT_ROOT", "/tmp/x")  # noqa: S108
        # output_dir / work_dir overridden because ``${hydra:runtime.*}`` is
        # only populated at fit time, not at compose time.
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=[
                    "datamodule=ksin",
                    "model=ffn",
                    "trainer=cpu",
                    "paths.output_dir=/tmp/x/out",  # noqa: S108
                    "paths.work_dir=/tmp/x",  # noqa: S108
                ],
            )
            HydraConfig.instance().set_config(cfg)
            resolved_paths = cast(
                "dict[str, Any]", OmegaConf.to_container(cfg.paths, resolve=True)
            )
        parsed = PathsConfig.model_validate(resolved_paths)
        assert parsed.root_dir == "/tmp/x"  # noqa: S108
        assert parsed.data_dir.startswith("/tmp/x")  # noqa: S108


class TestDataModuleConfigRejectsBadInputs:
    """Validators must catch obvious mistakes on the typed fields."""

    def test_missing_target_rejected(self) -> None:
        """Hydra needs ``_target_`` to instantiate the datamodule — required."""
        with pytest.raises(ValidationError):
            DataModuleConfig.model_validate({"batch_size": 32})

    def test_blank_target_rejected(self) -> None:
        """A blank ``_target_`` would crash ``hydra.utils.instantiate`` mid-run."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            DataModuleConfig.model_validate({"_target_": "  ", "batch_size": 32})
