"""Deprecation shim re-exporting :mod:`synth_setter.models.vst_flow_matching_module`.

Archived W&B run configs and external job scripts resolve ``_target_`` paths
under this old module name; the symbols now live in ``vst_flow_matching_module``.
See #1664.
"""

from __future__ import annotations

from synth_setter.models.vst_flow_matching_module import (
    SurgeFlowMatchingModule,
    VSTFlowMatchingModule,
    call_with_cfg,
    rk4_with_cfg,
)

__all__ = [
    "SurgeFlowMatchingModule",
    "VSTFlowMatchingModule",
    "call_with_cfg",
    "rk4_with_cfg",
]
