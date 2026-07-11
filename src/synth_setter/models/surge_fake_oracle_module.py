"""Deprecation shim re-exporting :mod:`synth_setter.models.vst_fake_oracle_module`.

Archived W&B run configs and external job scripts resolve ``_target_`` paths
under this old module name; the symbols now live in ``vst_fake_oracle_module``.
See #1664.
"""

from __future__ import annotations

from synth_setter.models.vst_fake_oracle_module import (
    FakeOracleNet,
    SurgeFakeOracleModule,
    VSTFakeOracleModule,
)

__all__ = ["FakeOracleNet", "SurgeFakeOracleModule", "VSTFakeOracleModule"]
