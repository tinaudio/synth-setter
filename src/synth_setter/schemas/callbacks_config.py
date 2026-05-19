"""Pydantic schemas for the YAMLs under ``configs/callbacks/``.

After Hydra composes, ``cfg.callbacks`` is a flat ``name → instance`` dict.
The schema requires every value to carry ``_target_``; the runtime helper
``synth_setter.utils.instantiate_callbacks`` is more lenient and skips
non-``_target_`` entries, but that path is reserved for raw DictConfigs that
bypass schema validation.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field, RootModel

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["CallbackInstance", "CallbacksConfig"]


class CallbackInstance(StrictAllowExtraModel):
    """One entry of ``cfg.callbacks``; callback kwargs pass through via ``extra="allow"``.

    .. attribute :: target_

        Fully-qualified callback class path.
    """

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified callback class path. Each entry of "
            "``cfg.callbacks`` is passed to ``hydra.utils.instantiate``."
        ),
    )


class CallbacksConfig(RootModel[dict[NonBlankStr, CallbackInstance]]):
    """Shape of ``cfg.callbacks`` — a ``name → CallbackInstance`` mapping.

    ``configs/callbacks/none.yaml`` resolves to ``None``, not ``{}``, and is
    handled by ``instantiate_callbacks`` short-circuiting on a falsy config.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.
    """

    model_config = ConfigDict(strict=True)
