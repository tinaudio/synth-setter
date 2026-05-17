"""Behavioural tests for the ``DataConfig`` pydantic model.

Every YAML under ``configs/data/`` must validate against ``DataConfig`` —
that's the contract the published docs assert. Datamodule-specific keys
live under ``extra="allow"`` so a new datamodule can ship without
re-touching the schema; the common shape (``_target_``) stays typed.

Some shipped YAMLs declare Hydra ``???`` mandatory-override sentinels for
fields like ``dataset_root`` / ``stats_file`` / ``predict_file``. OmegaConf
preserves those as the literal string ``"???"`` under
``to_container(resolve=False)``; they only fail at attribute access /
``hydra.utils.instantiate`` time, which is the intended behaviour, so the
schema sees them as ordinary extras and accepts them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.schemas.data_config import DataConfig

_DATA_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "data"


def _all_data_config_names() -> list[str]:  # noqa: DOC201,DOC203
    """Return the YAML stem of every direct data config under ``configs/data/``."""
    names = sorted(p.stem for p in _DATA_CONFIG_DIR.glob("*.yaml"))
    assert names, f"no data YAMLs found under {_DATA_CONFIG_DIR} — has the layout changed?"
    return names


def _compose_data_cfg(data_name: str) -> dict[str, Any]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Compose a full train config with ``data=<data_name>`` selected."""
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"data={data_name}", "model=ffn", "trainer=cpu"],
        )
    data_subtree = OmegaConf.to_container(cfg.data, resolve=False)
    assert isinstance(data_subtree, dict)
    return cast("dict[str, Any]", data_subtree)


class TestDataConfigAcceptsEveryConfig:
    """Every shipped data YAML must validate against ``DataConfig``."""

    @pytest.mark.parametrize("data_name", _all_data_config_names())
    def test_data_yaml_validates(self, data_name: str) -> None:  # noqa: DOC101,DOC103
        """The composed ``data`` subtree validates as ``DataConfig``."""
        data_subtree = _compose_data_cfg(data_name)
        DataConfig.model_validate(data_subtree)

    def test_target_field_typed(self) -> None:
        """``_target_`` lands on ``target_`` with the expected datamodule path."""
        data_subtree = _compose_data_cfg("ksin")
        parsed = DataConfig.model_validate(data_subtree)
        assert parsed.target_.endswith("KSinDataModule")


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
