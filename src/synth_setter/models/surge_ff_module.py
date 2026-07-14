"""Deprecation shim re-exporting :mod:`synth_setter.models.vst_ff_module`.

Archived W&B run configs and external job scripts resolve ``_target_`` paths
under this old module name; the symbols now live in ``vst_ff_module``. See #1664.
"""

from __future__ import annotations

from synth_setter.models.vst_ff_module import (
    SurgeFeedForwardModule,
    VSTFeedForwardModule,
)

__all__ = ["SurgeFeedForwardModule", "VSTFeedForwardModule"]
