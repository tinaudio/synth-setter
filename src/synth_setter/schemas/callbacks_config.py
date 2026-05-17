"""Pydantic schemas for Lightning callback configs under ``configs/callbacks/``.

Each YAML under ``configs/callbacks/`` either defines a single named callback
(``model_checkpoint.yaml`` → ``{"model_checkpoint": {"_target_": ...}}``) or
composes several via ``defaults:`` plus per-callback overrides
(``default.yaml``). Once Hydra has finished composing, ``cfg.callbacks`` is a
flat mapping from callback-name → callback-instance dict, which
``synth_setter.utils.instantiate_callbacks`` then iterates over —
``hydra.utils.instantiate`` is called only on values that contain
``_target_``, so unrelated keys are silently skipped.

The two schemas here mirror that shape:

* :class:`CallbackInstance` is what each value in the dict must satisfy.
* :class:`CallbacksConfig` is the top-level :class:`~pydantic.RootModel`
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

__all__ = ["CallbackInstance", "CallbacksConfig"]


class CallbackInstance(BaseModel):  # noqa: DOC601,DOC603
    """One entry of the ``cfg.callbacks`` dict.

    Only ``_target_`` is typed at this layer; callback-class kwargs
    (``monitor``, ``dirpath``, ``patience``, ``every_n_train_steps``, ...)
    vary per callback and pass through via ``extra="allow"``. The constraints
    on those kwargs live on the upstream Lightning / project-local callback
    class signatures rather than being duplicated here.
    """

    model_config = ConfigDict(strict=True, extra="allow", populate_by_name=True)

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified callback class path. Each entry of "
            "``cfg.callbacks`` is passed to ``hydra.utils.instantiate``."
        ),
    )


class CallbacksConfig(RootModel[dict[NonBlankStr, CallbackInstance]]):  # noqa: DOC601,DOC603
    """Top-level shape of ``cfg.callbacks`` — a mapping of name → instance.

    The keys are callback names (``model_checkpoint``, ``lr_monitor``,
    ``plot_proj_ii``, ...). Each value validates against
    :class:`CallbackInstance`. ``configs/callbacks/none.yaml`` is an empty
    document and Hydra resolves it to ``None`` rather than ``{}``; that
    pathway bypasses this schema entirely and is handled by
    ``synth_setter.utils.instantiate_callbacks`` (it short-circuits on a
    falsy config).
    """
