"""Behavioural tests for the ``LoggerConfig`` / ``LoggerInstance`` models.

Each YAML under ``configs/logger/`` that selects one or more loggers must
validate against ``LoggerConfig`` (a RootModel wrapping
``dict[str, LoggerInstance]``). Logger-class-specific kwargs vary per
backend and live under ``extra="allow"`` on ``LoggerInstance``; only
``_target_`` is typed at this layer.

The leaf logger YAMLs (``wandb``, ``tensorboard``, ``mlflow``, ``csv``,
``aim``, ``comet``, ``neptune``) are also legal group selections — they
each declare a single named logger at the top level. ``many_loggers.yaml``
is the composition that pulls in several at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.schemas.logger_config import LoggerConfig, LoggerInstance

_LOGGER_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "logger"


def _all_logger_config_names() -> list[str]:  # noqa: DOC201,DOC203
    """Return the YAML stem of every direct logger config under ``configs/logger/``."""
    names = sorted(p.stem for p in _LOGGER_CONFIG_DIR.glob("*.yaml"))
    assert names, f"no logger YAMLs found under {_LOGGER_CONFIG_DIR} — has the layout changed?"
    return names


def _compose_logger_cfg(logger_name: str) -> dict[str, Any]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Compose a full train config with ``logger=<logger_name>`` selected."""
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"logger={logger_name}",
                "data=ksin",
                "model=ffn",
                "trainer=cpu",
            ],
        )
    logger_subtree = OmegaConf.to_container(cfg.logger, resolve=False)
    assert isinstance(logger_subtree, dict)
    return cast("dict[str, Any]", logger_subtree)


class TestLoggerConfigAcceptsEveryComposition:
    """Every shipped logger group must validate against ``LoggerConfig``."""

    @pytest.mark.parametrize("logger_name", _all_logger_config_names())
    def test_logger_yaml_validates(self, logger_name: str) -> None:  # noqa: DOC101,DOC103
        """The composed ``logger`` subtree validates as ``LoggerConfig``."""
        logger_subtree = _compose_logger_cfg(logger_name)
        LoggerConfig.model_validate(logger_subtree)

    def test_many_loggers_yields_multiple_entries(self) -> None:
        """``many_loggers.yaml`` composes more than one logger entry."""
        logger_subtree = _compose_logger_cfg("many_loggers")
        parsed = LoggerConfig.model_validate(logger_subtree)
        assert len(parsed.root) > 1


_VALID_LOGGER = {
    "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
    "save_dir": "/tmp/csv",  # noqa: S108
}


class TestLoggerConfigRejectsBadInputs:
    """Validators must catch obvious mistakes on the typed fields."""

    def test_missing_target_in_instance_rejected(self) -> None:
        """Each logger instance must carry ``_target_``; reject if absent."""
        with pytest.raises(ValidationError):
            LoggerConfig.model_validate({"csv": {"save_dir": "/tmp/csv"}})  # noqa: S108

    def test_blank_target_rejected(self) -> None:
        """A blank ``_target_`` would crash ``hydra.utils.instantiate`` mid-fit."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            LoggerConfig.model_validate({"lg": {"_target_": "  "}})

    def test_blank_logger_name_rejected(self) -> None:
        """RootModel key is ``NonBlankStr`` — empty logger names are rejected."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            LoggerConfig.model_validate({"   ": _VALID_LOGGER})


class TestLoggerInstanceDirect:
    """Direct ``LoggerInstance`` validation works for individual entries."""

    def test_valid_instance_parses(self) -> None:
        """A minimal valid logger dict parses cleanly."""
        parsed = LoggerInstance.model_validate(_VALID_LOGGER)
        assert parsed.target_ == "lightning.pytorch.loggers.csv_logs.CSVLogger"
