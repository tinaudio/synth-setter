"""Sidecar metadata model for wds tar shards.

Owns ``ShardMetadata`` — the leaf-module Pydantic model written as
``metadata.json`` inside each wds tar. Pinned in its own module so consumers
on either side of the ``src/`` ↔ ``pipeline/`` boundary can import it
without picking up transitive ML dependencies (``pedalboard`` via
``src.data.vst``) that would form an import cycle with the renderer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class ShardMetadata(BaseModel):
    """Sidecar JSON written into wds tar shards (member ``metadata.json``).

    Mirrors the ``audio`` HDF5 dataset attrs the wds layout doesn't have a
    natural home for. Validated on read by ``validate_shard`` so a malformed
    sidecar fails loudly instead of silently shipping a half-described shard.

    Lives in a leaf module (no project imports) so consumers on either side
    of the ``src/`` ↔ ``pipeline/`` boundary can import it without picking up
    transitive dependencies that would form an import cycle.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    velocity: int
    signal_duration_seconds: float
    sample_rate: float
    channels: int
    min_loudness: float

    @model_validator(mode="after")
    def _ranges_must_be_sane(self) -> ShardMetadata:
        """Reject metadata.json sidecars whose values are strict-typed but semantically broken.

        ``DatasetSpec`` / ``RenderConfig`` validate these on the upstream side; mirroring the same
        checks here closes the trust boundary on read so an externally-produced metadata.json can't
        ship with negative sample_rate or out-of-range velocity.
        """
        if not (0 <= self.velocity <= 127):
            raise ValueError(f"velocity must be in [0, 127], got {self.velocity}")
        if self.signal_duration_seconds <= 0:
            raise ValueError(
                f"signal_duration_seconds must be positive, got {self.signal_duration_seconds}"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels < 1:
            raise ValueError(f"channels must be >= 1, got {self.channels}")
        return self
