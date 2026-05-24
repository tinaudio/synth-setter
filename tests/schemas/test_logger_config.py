"""Behavioural tests for ``LoggerConfig`` / ``LoggerInstance``.

Every ``configs/logger/`` YAML must validate; logger kwargs ride
``extra="allow"`` on ``LoggerInstance``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.resources import configs_dir
from synth_setter.schemas.logger_config import LoggerConfig, LoggerInstance
from tests.schemas.conftest import compose_subtree

_LOGGER_CONFIG_DIR = configs_dir() / "logger"


def _all_logger_config_names() -> list[str]:
    """Return the YAML stem of every direct logger config under ``configs/logger/``.

    :return: Sorted list of YAML stems found in ``configs/logger/``.
    """
    names = sorted(
        p.name.removesuffix(".yaml")
        for p in _LOGGER_CONFIG_DIR.iterdir()
        if p.is_file() and p.name.endswith(".yaml")
    )
    assert names, f"no logger YAMLs found under {_LOGGER_CONFIG_DIR} — has the layout changed?"
    return names


class TestLoggerConfigAcceptsEveryComposition:
    """Every shipped logger group must validate against ``LoggerConfig``."""

    @pytest.mark.parametrize("logger_name", _all_logger_config_names())
    def test_logger_yaml_validates(self, logger_name: str) -> None:
        """The composed ``logger`` subtree validates as ``LoggerConfig``.

        :param logger_name: Parametrized YAML stem under ``configs/logger/``.
        """
        logger_subtree = compose_subtree("logger", logger_name)
        parsed = LoggerConfig.model_validate(logger_subtree)
        assert parsed.root

    def test_many_loggers_yields_multiple_entries(self) -> None:
        """``many_loggers.yaml`` composes more than one logger entry."""
        logger_subtree = compose_subtree("logger", "many_loggers")
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
