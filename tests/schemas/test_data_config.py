"""Behavioural tests for the ``DataConfig`` pydantic model.

Hydra ``???`` mandatory-override sentinels survive ``to_container(resolve=False)``
as the literal string ``"???"``; the schema sees them as ordinary extras
and accepts them, deferring the failure to ``hydra.utils.instantiate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.schemas.data_config import DataConfig
from synth_setter.schemas.paths_config import PathsConfig
from tests.schemas.conftest import compose_subtree

_DATA_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "data"


def _all_data_config_names() -> list[str]:  # noqa: DOC201,DOC203
    """Return the YAML stem of every direct data config under ``configs/data/``."""
    names = sorted(p.stem for p in _DATA_CONFIG_DIR.glob("*.yaml"))
    assert names, f"no data YAMLs found under {_DATA_CONFIG_DIR} — has the layout changed?"
    return names


class TestDataConfigAcceptsEveryConfig:
    """Every shipped data YAML must validate against ``DataConfig``."""

    @pytest.mark.parametrize("data_name", _all_data_config_names())
    def test_data_yaml_validates(self, data_name: str) -> None:  # noqa: DOC101,DOC103
        """The composed ``data`` subtree validates as ``DataConfig``."""
        data_subtree = compose_subtree("data", data_name)
        parsed = DataConfig.model_validate(data_subtree)
        assert parsed.target_

    def test_target_field_typed(self) -> None:
        """``_target_`` lands on ``target_`` with the expected datamodule path."""
        data_subtree = compose_subtree("data", "ksin")
        parsed = DataConfig.model_validate(data_subtree)
        assert parsed.target_.endswith("KSinDataModule")


class TestPathsConfigResolvedInterpolation:
    """A real resolved ``PROJECT_ROOT`` must round-trip through ``NonBlankStr``."""

    def test_paths_resolved_with_env_var(  # noqa: DOC101,DOC103
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``${oc.env:PROJECT_ROOT}`` resolves to a real path and validates as non-blank."""
        monkeypatch.setenv("PROJECT_ROOT", "/tmp/x")  # noqa: S108
        # output_dir / work_dir overridden because ``${hydra:runtime.*}`` is
        # only populated at fit time, not at compose time.
        with initialize(version_base="1.3", config_path="../../configs"):
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=[
                    "data=ksin",
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


class TestDataConfigRejectsBadInputs:
    """Validators must catch obvious mistakes on the typed fields."""

    def test_missing_target_rejected(self) -> None:
        """Hydra needs ``_target_`` to instantiate the datamodule — required."""
        with pytest.raises(ValidationError):
            DataConfig.model_validate({"batch_size": 32})

    def test_blank_target_rejected(self) -> None:
        """A blank ``_target_`` would crash ``hydra.utils.instantiate`` mid-run."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            DataConfig.model_validate({"_target_": "  ", "batch_size": 32})
