"""Pydantic schemas for the YAMLs under ``configs/logger/``.

Mirrors the callbacks-config layout: composed ``cfg.logger`` is a flat
``name → instance`` dict iterated by
``synth_setter.utils.instantiate_loggers``, which skips any entry that is
``None`` (a disabled logger) or carries no ``_target_`` (a partial override).
The schema constrains ``_target_`` only when present, matching that contract.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field, RootModel

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["LoggerConfig", "LoggerInstance"]


class LoggerInstance(StrictAllowExtraModel):
    """One entry of ``cfg.logger``; logger kwargs pass through via ``extra="allow"``.

    .. attribute :: target_

        Fully-qualified logger class path, or ``None`` for a partial override.
    """

    target_: NonBlankStr | None = Field(
        default=None,
        alias="_target_",
        description=(
            "Fully-qualified logger class path (e.g. "
            "``lightning.pytorch.loggers.wandb.WandbLogger``); "
            "``hydra.utils.instantiate`` builds the entry only when set. ``None`` "
            "(a partial override) is skipped at instantiation time."
        ),
    )


class LoggerConfig(RootModel[dict[NonBlankStr, LoggerInstance | None]]):
    """Shape of ``cfg.logger`` — a ``name → LoggerInstance | None`` mapping.

    ``instantiate_loggers`` short-circuits on a falsy ``cfg.logger`` (empty
    or ``None``), so that whole-group case bypasses this schema; a ``None``
    *value* here is a single logger an experiment disabled via ``<name>: null``.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.
    """

    model_config = ConfigDict(strict=True)
