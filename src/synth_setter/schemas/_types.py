"""Shared annotated types used across the training-config schemas.

Lives in a leaf module to avoid a circular import between
``schemas/__init__.py`` and the ``train_config`` / ``model_config`` modules
that consume these types.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

__all__ = ["NonBlankStr"]


# Shared annotated type for fields whose blank value would silently break a
# downstream path derivation (run-id, output dir) or crash a Hydra
# instantiation. Stripping leading/trailing whitespace before the length
# check accepts ``"train"`` but rejects ``"   "``.
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
