"""Shared annotated types and base class for the training-config schemas.

Leaf module to avoid a circular import via ``schemas/__init__.py``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

__all__ = ["NonBlankStr", "StrictAllowExtraModel"]


# Strips before the length check, so ``"train"`` passes and ``"   "`` doesn't.
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictAllowExtraModel(BaseModel):  # noqa: DOC601,DOC603
    """Trust-boundary base: ``strict=True`` + ``extra="allow"`` + alias-by-name.

    Diverges from ``pipeline.schemas``' ``extra="forbid"`` because training
    configs are composed from many Hydra subtrees whose keys vary per variant.
    """

    model_config = ConfigDict(strict=True, extra="allow", populate_by_name=True)
