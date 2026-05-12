"""Leaf-module home for the wds tar shard's ``metadata.json`` sidecar model.

Kept free of project imports (only ``pydantic``) so consumers on either side
of the ``src/`` ↔ ``src/pipeline/`` boundary — notably ``src/data/vst`` — can
import it without picking up the transitive ``pedalboard`` dependency that
would otherwise form a launcher-side import cycle.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class ShardMetadata(BaseModel):
    """Sidecar JSON written into wds tar shards (member ``metadata.json``).

    Mirrors the ``audio`` HDF5 dataset attrs that the wds layout doesn't have
    a natural home for. The wds writer (PR-13) and the wds branch of
    ``validate_shard`` (also PR-13) will consume this model directly so a
    malformed sidecar fails loudly at write or read time instead of silently
    shipping a half-described shard.

    JSON read off R2 is a trust boundary — the value ranges below match
    ``RenderConfig._ranges_must_be_sane`` so a corrupted or hand-edited
    sidecar fails validation rather than being accepted with nonsensical
    values that would only surface later as a training-time crash.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    velocity: int
    signal_duration_seconds: float
    sample_rate: int
    channels: int
    min_loudness: float

    @model_validator(mode="after")
    def _ranges_must_be_sane(self) -> ShardMetadata:
        """Reject out-of-range values — mirrors ``RenderConfig._ranges_must_be_sane``."""
        if not (0 <= self.velocity <= 127):
            raise ValueError("velocity must be in [0, 127]")
        if self.signal_duration_seconds <= 0:
            raise ValueError("signal_duration_seconds must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        return self
