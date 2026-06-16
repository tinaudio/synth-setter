"""Leaf-module home for the wds tar shard's ``metadata.json`` sidecar model.

Kept free of project imports (only ``pydantic``) so consumers on either side
of the ``synth_setter/`` boundary — notably ``synth_setter.data.vst`` — can
import it without picking up the transitive ``pedalboard`` dependency that
would otherwise form a launcher-side import cycle.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_ATTEMPTS_PER_SAMPLE = 100


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

    .. attribute :: model_config

        Strict, frozen, extra-forbid Pydantic configuration.

    .. attribute :: velocity
        :type: int

        MIDI velocity used for every render.

    .. attribute :: signal_duration_seconds
        :type: float

        Duration of each rendered audio sample, in seconds.

    .. attribute :: sample_rate
        :type: int

        Audio sample rate in Hz.

    .. attribute :: channels
        :type: int

        Audio channel count.

    .. attribute :: min_loudness
        :type: float

        Per-sample loudness floor used during rendering.

    .. attribute :: base_seed
        :type: int

        Per-shard master seed the row RNGs are derived from (#884).

    .. attribute :: attempts_per_sample
        :type: int

        Per-row loudness-gate retry budget used during rendering.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    velocity: int = Field(description="MIDI velocity used for every render (0-127).")
    signal_duration_seconds: float = Field(
        description="Duration of each rendered audio sample, in seconds."
    )
    sample_rate: int = Field(description="Audio sample rate in Hz.")
    channels: int = Field(description="Audio channel count.")
    min_loudness: float = Field(description="Per-sample loudness floor used during rendering.")
    base_seed: int = Field(
        default=0,
        description=(
            "Per-shard master seed the row RNGs are derived from (#884). Defaults to 0 so "
            "sidecars written before this field existed still validate; new shards write the "
            "real seed."
        ),
    )
    attempts_per_sample: int = Field(
        default=DEFAULT_ATTEMPTS_PER_SAMPLE,
        description=(
            "Per-row loudness-gate retry budget used during rendering. Defaults to "
            f"{DEFAULT_ATTEMPTS_PER_SAMPLE} so sidecars written before this field existed "
            "still validate."
        ),
    )

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
        if self.attempts_per_sample < 1:
            raise ValueError("attempts_per_sample must be >= 1")
        return self
