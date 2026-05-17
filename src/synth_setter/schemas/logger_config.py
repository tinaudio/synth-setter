"""Pydantic schemas for Lightning logger configs under ``configs/logger/``.

Mirrors the callbacks-config layout: each YAML under ``configs/logger/``
either defines a single named logger (``wandb.yaml`` →
``{"wandb": {"_target_": ...}}``) or composes several via ``defaults:``
(``many_loggers.yaml`` pulls in csv + tensorboard + wandb). Composed
``cfg.logger`` is a flat mapping name → logger-instance dict, iterated
by ``synth_setter.utils.instantiate_loggers``.

The two schemas here mirror that shape:

* :class:`LoggerInstance` is what each value in the dict must satisfy.
* :class:`LoggerConfig` is the top-level :class:`~pydantic.RootModel`
  that validates the whole dict at once.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)

from synth_setter.schemas._types import NonBlankStr

__all__ = ["LoggerConfig", "LoggerInstance"]


class LoggerInstance(BaseModel):  # noqa: DOC601,DOC603
    """One entry of the ``cfg.logger`` dict.

    Only ``_target_`` is typed at this layer; logger-class kwargs
    (``api_key``, ``project``, ``save_dir``, ``tags``, ...) vary per logger
    and pass through via ``extra="allow"``. The constraints on those kwargs
    live on the upstream logger class signatures.
    """

    model_config = ConfigDict(strict=True, extra="allow", populate_by_name=True)

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified logger class path (e.g. "
            "``lightning.pytorch.loggers.wandb.WandbLogger``). Each entry of "
            "``cfg.logger`` is passed to ``hydra.utils.instantiate``."
        ),
    )


class LoggerConfig(RootModel[dict[NonBlankStr, LoggerInstance]]):  # noqa: DOC601,DOC603
    """Top-level shape of ``cfg.logger`` — a mapping of name → instance.

    The keys are logger names (``csv``, ``tensorboard``, ``wandb``, ...).
    Each value validates against :class:`LoggerInstance`.
    ``instantiate_loggers`` short-circuits when the composed config is falsy
    (no logger configured), so an empty or ``None`` ``cfg.logger`` is
    handled outside this schema.
    """
