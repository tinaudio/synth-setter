"""Pydantic schemas for per-model Hydra configs under ``configs/model/``.

Every YAML under ``configs/model/`` declares: the LightningModule target, a
partial torch optimizer, an optional partial LR scheduler, and a ``compile``
flag. Variant-specific fields vary per model module and are accepted via
``extra="allow"`` — adding a new model YAML does not require touching this
schema.

``_target_`` is exposed on the Python model as ``target_`` (trailing
underscore) because leading-underscore field names are not addressable on a
pydantic ``BaseModel``. The alias ``_target_`` is preserved on input/output.
"""

from __future__ import annotations

from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    StrictBool,
)

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["ModelConfig", "OptimizerConfig", "SchedulerConfig"]


class OptimizerConfig(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """Partial torch optimizer config injected at fit-time with model parameters.

    ``_partial_: true`` means ``hydra.utils.instantiate`` returns a
    ``functools.partial`` instead of a fully-constructed optimizer; the
    LightningModule binds the model parameters at the call site. Per-field
    descriptions live on the ``Field`` definitions below.
    """

    target_: NonBlankStr = Field(
        alias="_target_",
        description="Fully-qualified torch optimizer class path (e.g. ``torch.optim.Adam``).",
    )
    partial_: StrictBool = Field(
        default=True,
        alias="_partial_",
        description="Whether Hydra returns a partial; ``True`` is the convention here.",
    )
    lr: PositiveFloat = Field(description="Learning rate (must be positive).")
    weight_decay: NonNegativeFloat = Field(
        default=0.0,
        description="L2 weight-decay coefficient (must be non-negative).",
    )


class SchedulerConfig(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """Partial torch LR scheduler config injected at fit-time with the optimizer.

    Several configs ship ``scheduler: null``; in those cases the YAML maps to
    ``ModelConfig.scheduler = None`` rather than this model. Per-field
    descriptions live on the ``Field`` definitions below.
    """

    target_: NonBlankStr = Field(
        alias="_target_",
        description="Fully-qualified LR-scheduler class path.",
    )
    partial_: StrictBool = Field(
        default=True,
        alias="_partial_",
        description="Whether Hydra returns a partial; conventionally ``True``.",
    )


class ModelConfig(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """Per-model Hydra config (one of the YAMLs under ``configs/model/``).

    The typed fields are the common surface area; variant-specific keys are
    accepted via ``extra="allow"`` so a new model module ships without a
    schema edit. Per-field descriptions live on the ``Field`` definitions
    below.
    """

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified ``LightningModule`` class path. Resolved by "
            "``hydra.utils.instantiate(cfg.model)`` in ``cli/train.py``."
        ),
    )
    optimizer: OptimizerConfig = Field(
        description="Partial torch optimizer config (see ``OptimizerConfig``)."
    )
    scheduler: SchedulerConfig | None = Field(
        default=None,
        description=(
            "Partial LR scheduler config, or ``null`` to run without a "
            "scheduler. See ``SchedulerConfig``."
        ),
    )
    # Field name mirrors the YAML key; shadows the ``compile()`` builtin in cfg.model.compile.
    compile: StrictBool = Field(
        default=True,
        description="Whether to wrap the module in ``torch.compile`` at setup time.",
    )
