from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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
