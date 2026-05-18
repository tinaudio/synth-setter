"""Pydantic schemas for the YAMLs under ``configs/callbacks/``.

After Hydra composes, ``cfg.callbacks`` is a flat ``name → instance`` dict
that ``synth_setter.utils.instantiate_callbacks`` iterates over (skipping
entries that lack ``_target_``).
"""

from __future__ import annotations

from pydantic import Field, RootModel

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["CallbackInstance", "CallbacksConfig"]


class CallbackInstance(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """One entry of ``cfg.callbacks``; callback kwargs pass through via ``extra="allow"``."""

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified callback class path. Each entry of "
            "``cfg.callbacks`` is passed to ``hydra.utils.instantiate``."
        ),
    )


class CallbacksConfig(RootModel[dict[NonBlankStr, CallbackInstance]]):  # noqa: DOC601,DOC603
    """Shape of ``cfg.callbacks`` — a ``name → CallbackInstance`` mapping.

    ``configs/callbacks/none.yaml`` resolves to ``None``, not ``{}``, and is
    handled by ``instantiate_callbacks`` short-circuiting on a falsy config.
    """
