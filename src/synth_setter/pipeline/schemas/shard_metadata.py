"""Leaf-module home for the wds tar shard's ``metadata.json`` sidecar model.

Kept free of project imports (only ``pydantic``) so consumers on either side
of the ``synth_setter/`` boundary — notably ``synth_setter.data.vst`` — can
import it without picking up the transitive ``pedalboard`` dependency that
would otherwise form a launcher-side import cycle.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    velocity: int = Field(description="MIDI velocity used for every render (0-127).")
    signal_duration_seconds: float = Field(
        description="Duration of each rendered audio sample, in seconds."
    )
    sample_rate: int = Field(description="Audio sample rate in Hz.")
    channels: int = Field(description="Audio channel count.")
    min_loudness: float = Field(description="Per-sample loudness floor used during rendering.")

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


class BlobFieldSpec(BaseModel):
    """Per-row shape and dtype of a column stored as opaque ``large_binary`` bytes.

    Embedded in the Lance schema metadata so a BLOB column — which, unlike a
    fixed-shape tensor, carries no shape or dtype in its Arrow type — stays
    self-describing: the reader and validator recover the row geometry without
    the render config. Read off a shard file, so a trust boundary.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: shape

        Per-row inner shape (no leading row axis).

    .. attribute :: dtype

        NumPy dtype name of the stored bytes, e.g. ``float16``.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    shape: list[int]
    dtype: str

    @field_validator("shape")
    @classmethod
    def _shape_dims_must_be_positive(cls, value: list[int]) -> list[int]:
        """Reject empty or non-positive dims so ``reshape`` can't infer a wrong geometry.

        A ``-1`` would be numpy's "infer" sentinel and an empty shape collapses the
        row; both must fail at this trust boundary rather than silently mis-decode.

        :param value: Candidate per-row inner shape.
        :returns: The validated shape unchanged.
        :raises ValueError: If the shape is empty or has a non-positive dimension.
        """
        if not value or any(dim <= 0 for dim in value):
            raise ValueError(f"shape dims must be positive and non-empty, got {value}")
        return value

    @field_validator("dtype")
    @classmethod
    def _dtype_must_name_a_numpy_dtype(cls, value: str) -> str:
        """Reject a ``dtype`` string numpy can't resolve, so decode fails at parse not read.

        :param value: Candidate numpy dtype name.
        :returns: The validated dtype name unchanged.
        :raises ValueError: If ``np.dtype(value)`` does not resolve.
        """
        import numpy as np

        try:
            np.dtype(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numpy dtype {value!r}") from exc
        return value
