"""Leaf-module home for the wds tar shard's ``metadata.json`` sidecar model.

Kept free of project imports (only ``pydantic``) so consumers on either side
of the ``src/`` ↔ ``src/pipeline/`` boundary — notably ``src/data/vst`` — can
import it without picking up the transitive ``pedalboard`` dependency that
would otherwise form a launcher-side import cycle.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ShardMetadata(BaseModel):
    """Sidecar JSON written into wds tar shards (member ``metadata.json``).

    Mirrors the ``audio`` HDF5 dataset attrs that the wds layout doesn't have
    a natural home for. Validated on read by ``validate_shard`` so a malformed
    sidecar fails loudly instead of silently shipping a half-described shard.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    velocity: int
    signal_duration_seconds: float
    sample_rate: int
    channels: int
    min_loudness: float
