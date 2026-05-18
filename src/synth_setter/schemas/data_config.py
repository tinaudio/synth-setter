"""Pydantic schema for the YAMLs under ``configs/data/``.

Datamodule kwargs vary per module (``SurgeDataModule`` is path-driven,
``KSinDataModule`` is synthetic) and pass through via ``extra="allow"``.
"""

from __future__ import annotations

from pydantic import Field

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["DataConfig"]


class DataConfig(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """One of the YAMLs under ``configs/data/``; only ``_target_`` is typed."""

    target_: NonBlankStr = Field(
        alias="_target_",
        description=(
            "Fully-qualified ``LightningDataModule`` class path. Resolved by "
            "``hydra.utils.instantiate(cfg.data)`` in ``cli/train.py``."
        ),
    )
