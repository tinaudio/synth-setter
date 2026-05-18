"""Pydantic schema for the startup toggles under ``configs/extras/``.

Read by ``synth_setter.utils.extras()`` and by
``torch.set_float32_matmul_precision`` at the top of ``cli/train.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool

from synth_setter.schemas._types import StrictAllowExtraModel

__all__ = ["ExtrasConfig"]


# Restated in the Literal annotation below because Literal[*tuple] unpacking
# needs Python 3.11+ and the project pins ``python_requires = ">=3.10"``.
_FLOAT32_MATMUL_PRECISIONS: tuple[str, ...] = ("highest", "high", "medium")


class ExtrasConfig(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """Startup toggles consumed by ``synth_setter.utils.extras(cfg)``."""

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
