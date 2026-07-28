"""Conditioning contracts shared across data and model layers."""

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

ConditioningMode = Literal["mel", "m2l"]
# Mirrors shapes.SKETCH_CONTROL_FIELD / NUM_SKETCH_CONTROLS; importing the
# data.vst package here would break the model modules' thin import contract.
SKETCH_CONTROL_FIELD = "sketch_ctrl"
NUM_SKETCH_CONTROLS = 3
LEGACY_M2L_INPUT_SHAPE = (128, 42)


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
    """Select the stored time-varying sketch control column for conditioning.

    .. attribute :: model_config

        Strict immutable Pydantic model configuration.

    .. attribute :: column

        Stored Lance column name.

    .. attribute :: num_controls

        Control tracks per row (loudness, spectral centroid, pitch).

    .. attribute :: num_frames

        Mel-grid frames per control track.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    column: str = SKETCH_CONTROL_FIELD
    num_controls: PositiveInt = NUM_SKETCH_CONTROLS
    num_frames: PositiveInt


SketchControls = SketchControlSpec | Mapping[str, object] | None


def resolve_sketch_controls(sketch: SketchControls) -> SketchControlSpec | None:
    """Resolve optional sketch-control configuration from Hydra or code.

    :param sketch: ``None``, a parsed spec, or a Hydra mapping.
    :returns: Parsed spec, or ``None`` when sketch conditioning is off.
    :raises TypeError: If ``sketch`` is neither ``None``, a spec, nor a mapping.
    """
    if sketch is None:
        return None
    if isinstance(sketch, SketchControlSpec):
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
