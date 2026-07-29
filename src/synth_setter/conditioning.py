"""Conditioning contracts shared across data and model layers."""

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

ConditioningMode = Literal["mel", "m2l"]
LEGACY_M2L_INPUT_SHAPE = (128, 42)

# Sketch-control storage contract (#2612), hosted here (``data.vst.shapes``
# re-exports it) so model modules import it without the VST runtime package.
SKETCH_CTRL_FIELD: str = "sketch_ctrl"
# PESTO mir-1k_g7 activation width: 128 semitones x 3 bins.
SKETCH_PITCH_BINS: int = 384
# Rows: the two scalar tracks (loudness, centroid), then the pitch block.
NUM_SKETCH_TRACK_ROWS: int = 2
NUM_SKETCH_CONTROLS: int = NUM_SKETCH_TRACK_ROWS + SKETCH_PITCH_BINS
SKETCH_LOUDNESS_ROW: int = 0
SKETCH_CENTROID_ROW: int = 1
SKETCH_PITCH_SLICE: slice = slice(NUM_SKETCH_TRACK_ROWS, NUM_SKETCH_CONTROLS)


class EmbeddingConditioningSpec(BaseModel):
    """Select one fixed-shape Lance embedding column for conditioning.

    .. attribute :: model_config

        Strict immutable Pydantic model configuration.

    .. attribute :: column

        Stored Lance column name.

    .. attribute :: input_shape

        Fixed per-row tensor shape expected from the column.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    column: str = Field(min_length=1)
    input_shape: tuple[PositiveInt, ...] = Field(min_length=1)


Conditioning = ConditioningMode | EmbeddingConditioningSpec | Mapping[str, object]


def select_conditioning(
    batch: Mapping[str, torch.Tensor], embedding: EmbeddingConditioningSpec | None
) -> torch.Tensor:
    """Select the legacy mel or canonical cached-conditioning tensor.

    :param batch: Model batch containing the configured conditioning tensor.
    :param embedding: Resolved cached-embedding spec, or ``None`` for legacy mel.
    :returns: The tensor selected for model conditioning.
    """
    if embedding is None:
        return batch["mel"]
    return batch["conditioning"]


def resolve_embedding_conditioning(
    conditioning: Conditioning,
) -> EmbeddingConditioningSpec | None:
    """Resolve generic embedding configuration while leaving mel on its legacy path.

    :param conditioning: Legacy literal, parsed spec, or Hydra mapping.
    :returns: Fixed-shape embedding spec, or ``None`` for legacy mel.
    :raises TypeError: If ``conditioning`` is neither a supported literal nor mapping.
    :raises ValueError: If an unsupported string literal is provided.
    """
    if isinstance(conditioning, str):
        if conditioning == "mel":
            return None
        if conditioning == "m2l":
            return EmbeddingConditioningSpec(
                column="music2latent", input_shape=LEGACY_M2L_INPUT_SHAPE
            )
        raise ValueError(f"unknown conditioning mode {conditioning!r}")
    if isinstance(conditioning, EmbeddingConditioningSpec):
        return conditioning
    if not isinstance(conditioning, Mapping):
        raise TypeError(f"conditioning must be 'mel', 'm2l', or a mapping, got {conditioning!r}")

    values = dict(conditioning)
    input_shape = values.get("input_shape")
    if isinstance(input_shape, Sequence) and not isinstance(input_shape, (str, tuple)):
        values["input_shape"] = tuple(input_shape)
    return EmbeddingConditioningSpec.model_validate(values)


class SketchControlSpec(BaseModel):
    """Select the stored sketch-control column and its token budget.

    The channel layout is fixed by this module's ``SKETCH_*`` constants and is
    not configurable here.

    .. attribute :: model_config

        Strict immutable Pydantic model configuration.

    .. attribute :: column

        Stored Lance column name.

    .. attribute :: num_frames

        Mel-grid frames per stored control row.

    .. attribute :: num_ctrl_tokens

        Control tokens the time axis is resampled to.

    .. attribute :: pitch_zero_threshold

        Pitch activations below this zero-bin at batch preparation (#2614).
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    column: str = Field(default=SKETCH_CTRL_FIELD, min_length=1)
    num_frames: PositiveInt
    num_ctrl_tokens: PositiveInt = 32
    # Bounded to the documented [0, 1] activation range: a negative threshold
    # silently disables binning and one above 1 zeroes the whole pitch block.
    pitch_zero_threshold: float = Field(default=0.1, ge=0.0, le=1.0)


SketchControls = SketchControlSpec | Mapping[str, object] | None


def resolve_sketch_controls(sketch: SketchControls) -> SketchControlSpec | None:
    """Resolve optional sketch-control configuration from Hydra or code.

    :param sketch: ``None``, a parsed spec, or a Hydra mapping.
    :returns: Parsed spec, or ``None`` when sketch conditioning is off.
    :raises TypeError: If ``sketch`` is neither ``None``, a spec, nor a mapping.
    """
    if sketch is None or isinstance(sketch, SketchControlSpec):
        return sketch
    if not isinstance(sketch, Mapping):
        raise TypeError(f"sketch must be None, a spec, or a mapping, got {sketch!r}")
    return SketchControlSpec.model_validate(dict(sketch))
