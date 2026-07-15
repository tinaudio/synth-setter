"""Conditioning types shared across data and model layers.

Use ``ConditioningMode`` to annotate routing boundaries::

    mode: ConditioningMode = "mel"
"""

from typing import Literal

type ConditioningMode = Literal["mel", "m2l"]
