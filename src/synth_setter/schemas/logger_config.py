"""Pydantic schemas for the YAMLs under ``configs/logger/``.

Mirrors the callbacks-config layout: composed ``cfg.logger`` is a flat
``name → instance`` dict iterated by
``synth_setter.utils.instantiate_loggers``.
"""

from __future__ import annotations

from pydantic import Field, RootModel

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["LoggerConfig", "LoggerInstance"]


class LoggerInstance(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """One entry of ``cfg.logger``; logger kwargs pass through via ``extra="allow"``."""

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified logger class path (e.g. "
            "``lightning.pytorch.loggers.wandb.WandbLogger``). Each entry of "
            "``cfg.logger`` is passed to ``hydra.utils.instantiate``."
        ),
    )


class LoggerConfig(RootModel[dict[NonBlankStr, LoggerInstance]]):  # noqa: DOC601,DOC603
    """Shape of ``cfg.logger`` — a ``name → LoggerInstance`` mapping.

    ``instantiate_loggers`` short-circuits on a falsy ``cfg.logger`` (empty
    or ``None``), so that case bypasses this schema.
    """
