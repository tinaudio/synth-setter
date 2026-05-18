"""Shared annotated types and base classes used across the training-config schemas.

Lives in a leaf module to avoid a circular import between
``schemas/__init__.py`` and the ``train_config`` / ``model_config`` modules
that consume these types.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

__all__ = ["NonBlankStr", "StrictAllowExtraModel"]


# Shared annotated type for fields whose blank value would silently break a
# downstream path derivation (run-id, output dir) or crash a Hydra
# instantiation. Stripping leading/trailing whitespace before the length
# check accepts ``"train"`` but rejects ``"   "``.
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictAllowExtraModel(BaseModel):  # noqa: DOC601,DOC603
    """Project-standard trust-boundary base for Hydra training-config schemas.

    Combines ``strict=True`` (refuse type-coercion at validation time) with
    ``extra="allow"`` (let Hydra-owned subtree keys pass through unchanged)
    and ``populate_by_name=True`` (so ``_target_``-style aliases work both
    by alias and by attribute name).

    The ``extra="allow"`` choice intentionally diverges from
    ``synth_setter.pipeline.schemas``'s ``extra="forbid"`` posture: pipeline
    schemas describe data on disk in R2 and benefit from strict closed
    shapes, while training configs are composed from many Hydra subtrees
    whose keys vary per variant and would be brittle under ``forbid``.
    """

    model_config = ConfigDict(strict=True, extra="allow", populate_by_name=True)
