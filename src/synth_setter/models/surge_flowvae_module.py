"""Deprecation shim re-exporting :mod:`synth_setter.models.vst_flowvae_module`.

Archived W&B run configs and external job scripts resolve ``_target_`` paths
under this old module name; the symbols now live in ``vst_flowvae_module``.
Importing either module pulls the optional ``nflows`` dependency. See #1664.
"""

from __future__ import annotations

from synth_setter.models.vst_flowvae_module import (
    SurgeFlowVAEModule,
    VSTFlowVAEModule,
)

__all__ = ["SurgeFlowVAEModule", "VSTFlowVAEModule"]
