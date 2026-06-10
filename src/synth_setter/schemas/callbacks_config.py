"""Pydantic schemas for the YAMLs under ``configs/callbacks/``.

After Hydra composes, ``cfg.callbacks`` is a flat ``name → instance`` dict.
An entry is ``None`` when an experiment disables a base callback
(``early_stopping: null``), and may carry no ``_target_`` when an experiment
partially overrides one (``model_checkpoint: {monitor: val/lsd}`` merged onto a
callbacks group that never defined a target). ``instantiate_callbacks`` skips
both, so the schema only constrains ``_target_`` *when present*: a typo'd
non-string target is still caught, while a disabled or partial entry passes.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field, RootModel

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["CallbackInstance", "CallbacksConfig"]


class CallbackInstance(StrictAllowExtraModel):
    """One entry of ``cfg.callbacks``; callback kwargs pass through via ``extra="allow"``.

    .. attribute :: target_

        Fully-qualified callback class path, or ``None`` for a partial override.
    """

    target_: NonBlankStr | None = Field(
        default=None,
        alias="_target_",
        description=(
            "Fully-qualified callback class path; ``hydra.utils.instantiate`` "
            "builds the entry only when set. ``None`` (a partial override that "
            "never landed on a base target) is skipped at instantiation time."
        ),
    )


class CallbacksConfig(RootModel[dict[NonBlankStr, CallbackInstance | None]]):
    """Shape of ``cfg.callbacks`` — a ``name → CallbackInstance | None`` mapping.

    ``configs/callbacks/none.yaml`` resolves the whole group to ``None`` (handled
    by the caller short-circuiting on a falsy config); a ``None`` *value* here is
    a single callback an experiment disabled via ``<name>: null``.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.
    """

    model_config = ConfigDict(strict=True)
