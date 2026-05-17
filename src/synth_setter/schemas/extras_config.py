"""Pydantic schema for the run-time extras under ``configs/extras/``.

Toggles consumed by ``synth_setter.utils.extras()`` (the helper invoked
from ``cli/train.py`` at startup) and by ``torch.set_float32_matmul_precision``.
The shipped composition is ``configs/extras/default.yaml``; the schema names
each toggle so a YAML rename surfaces here at validation time rather than
silently dropping the override.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
)

__all__ = ["ExtrasConfig"]


class ExtrasConfig(BaseModel):  # noqa: DOC601,DOC603
    """Run-time toggles read at the top of ``cli/train.py``.

    Per-field descriptions live on the ``Field`` definitions below.
    """

    model_config = ConfigDict(strict=True, extra="allow")

    ignore_warnings: StrictBool = Field(
        default=False,
        description=(
            "If ``True``, ``synth_setter.utils.extras`` silences Python "
            "``warnings`` (``warnings.filterwarnings('ignore')``)."
        ),
    )
    enforce_tags: StrictBool = Field(
        default=True,
        description=(
            "If ``True``, ``extras`` prompts on startup when no ``tags`` are "
            "set so every run ships with at least one tag for the logger."
        ),
    )
    print_config: StrictBool = Field(
        default=True,
        description=(
            "Pretty-print the composed Hydra config tree at startup using "
            "Rich, helpful for catching surprise overrides at launch."
        ),
    )
    float32_matmul_precision: Literal["highest", "high", "medium"] = Field(
        default="high",
        description=(
            "Passed to ``torch.set_float32_matmul_precision`` — controls "
            "TF32 on Ampere+ GPUs. ``high`` (the shipped default) trades a "
            "small numerical error for substantial throughput."
        ),
    )
