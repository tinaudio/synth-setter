"""Pydantic schemas for the YAMLs under ``configs/model/``.

``_target_`` is exposed as ``target_`` (trailing underscore) because
leading-underscore field names are not addressable on a pydantic
``BaseModel``; the ``_target_`` alias is preserved on input/output.
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
    """Partial torch optimizer; the LightningModule binds model params at call time."""

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
    """Partial torch LR scheduler; ``scheduler: null`` maps to ``None`` instead."""

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
    """One of the YAMLs under ``configs/model/``; variant kwargs via ``extra="allow"``."""

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
    compile: StrictBool = Field(
        default=True,
        description="Whether to wrap the module in ``torch.compile`` at setup time.",
    )
