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

from synth_setter.data.vst.param_spec_registry import param_specs
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


class TestSurgeDatamoduleOverlays:
    """The thin surge instances overlay ``vst`` and target ``VSTDataModule``."""

    def test_surge_mini_overlays_param_spec_name_surge_4(self) -> None:
        """``surge_mini`` overrides ``param_spec_name`` to ``surge_4`` over the ``vst`` base."""
        subtree = compose_subtree("datamodule", "surge_mini")
        assert subtree["param_spec_name"] == "surge_4"
        assert subtree["_target_"].endswith("VSTDataModule")

    def test_surge_simple_overlays_param_spec_name_surge_simple(self) -> None:
        """``surge_simple`` overrides ``param_spec_name`` over the ``vst`` base."""
        subtree = compose_subtree("datamodule", "surge_simple")
        assert subtree["param_spec_name"] == "surge_simple"
        assert subtree["_target_"].endswith("VSTDataModule")

    def test_surge_declares_param_spec_name_surge_xt(self) -> None:
        """``surge`` sets ``param_spec_name: surge_xt`` over the abstract ``vst`` base."""
        subtree = compose_subtree("datamodule", "surge")
        assert subtree["param_spec_name"] == "surge_xt"
        assert subtree["_target_"].endswith("VSTDataModule")

    def test_surge_debug_sets_repeat_first_batch_and_declares_spec(self) -> None:
        """``surge_debug`` flips ``repeat_first_batch`` and declares ``surge_xt`` explicitly."""
        subtree = compose_subtree("datamodule", "surge_debug")
        assert subtree["repeat_first_batch"] is True
        assert subtree["param_spec_name"] == "surge_xt"
        assert subtree["_target_"].endswith("VSTDataModule")


class TestParamSpecNameIsExplicit:
    """``param_spec_name`` must be chosen deliberately, never inherited from a base default."""

    def test_vst_base_leaves_param_spec_name_mandatory(self) -> None:
        """The abstract ``vst`` base carries the ``???`` sentinel, not a synth default."""
        subtree = compose_subtree("datamodule", "vst")
        assert subtree["param_spec_name"] == "???"

    # The abstract ``vst`` base is excluded: its mandatory sentinel is pinned above.
    @pytest.mark.parametrize(
        "datamodule_name",
        [name for name in _all_datamodule_config_names() if name != "vst"],
    )
    def test_vst_family_config_resolves_param_spec_name_to_registered_spec(
        self, datamodule_name: str
    ) -> None:
        """Every concrete VST-family config composes to a registered ``param_spec_name``.

        Guards new configs that inherit ``vst`` without declaring a spec: they
        compose to the literal ``"???"`` and fail here instead of at run time.

        :param datamodule_name: Parametrized YAML stem under ``configs/datamodule/``.
        """
        subtree = compose_subtree("datamodule", datamodule_name)
        if "param_spec_name" not in subtree:
            pytest.skip("not a VST-family datamodule")
        assert subtree["param_spec_name"] in param_specs, (
            f"datamodule/{datamodule_name}.yaml must declare an explicit, registered "
            f"param_spec_name; composed to {subtree['param_spec_name']!r}"
        )


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
