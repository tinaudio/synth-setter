"""Conditioning contracts shared across data and model layers."""

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

ConditioningMode = Literal["mel", "m2l"]
LEGACY_M2L_INPUT_SHAPE = (128, 42)

# Sketch-control row layout (#2612), hosted here rather than in
# ``data.vst.shapes`` (which re-exports them) so model modules can import it
# without initializing the VST runtime package.
# PESTO mir-1k_g7 activation width: 128 semitones x 3 bins.
SKETCH_PITCH_BINS = 384
# Rows: loudness, centroid, then the pitch-activation block.
NUM_SKETCH_CONTROLS = 2 + SKETCH_PITCH_BINS
SKETCH_LOUDNESS_ROW = 0
SKETCH_CENTROID_ROW = 1
SKETCH_PITCH_SLICE = slice(2, 2 + SKETCH_PITCH_BINS)


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


class SketchControlSpec(BaseModel):
    """Select the stored sketch-control column and its token budget.

    The channel layout (loudness, centroid, pitch rows) is fixed by
    ``synth_setter.data.vst.shapes`` and is not configurable here.

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

    column: str = "sketch_ctrl"
    num_frames: PositiveInt
    num_ctrl_tokens: PositiveInt = 32
    pitch_zero_threshold: float = 0.1


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
        return batch["mel_spec"]
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
