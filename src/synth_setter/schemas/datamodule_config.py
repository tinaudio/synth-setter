"""Pydantic schema for the YAMLs under ``configs/datamodule/``.

Datamodule kwargs vary per module (``VSTDataModule`` is path-driven,
``KSinDataModule`` is synthetic) and pass through via ``extra="allow"``.
"""

from __future__ import annotations

from pydantic import Field

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["DataModuleConfig"]


class DataModuleConfig(StrictAllowExtraModel):
    """One of the YAMLs under ``configs/datamodule/``; only ``_target_`` is typed.

    .. attribute :: target_

        Fully-qualified ``LightningDataModule`` class path.
    """

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified ``LightningDataModule`` class path. Resolved by "
            "``hydra.utils.instantiate(cfg.datamodule)`` in ``cli/train.py``."
        ),
    )
