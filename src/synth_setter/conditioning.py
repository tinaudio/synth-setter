"""Conditioning contracts shared across data and model layers."""

from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

ConditioningMode = Literal["mel", "m2l", "audio"]
LEGACY_M2L_INPUT_SHAPE = (128, 42)
# Modes read straight from the model-batch entry of the same name. "audio" serves
# online-render synths, which have no stored mel because their audio only exists
# at training time.
RAW_CONDITIONING_MODES: frozenset[str] = frozenset({"mel", "audio"})
# Every cached embedding is collated here, whatever its stored column.
EMBEDDING_BATCH_KEY = "conditioning"


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


def conditioning_batch_key(conditioning: Conditioning) -> str:
    """Return the model-batch key holding the configured conditioning tensor.

    :param conditioning: Configured mode literal, parsed spec, or Hydra mapping.
    :returns: Batch key; a raw mode names its own entry, embeddings share one.
    """
    if resolve_embedding_conditioning(conditioning) is not None:
        return EMBEDDING_BATCH_KEY
    # Only RAW_CONDITIONING_MODES resolve to no embedding, and each names its key.
    return cast(str, conditioning)


def resolve_embedding_conditioning(
    conditioning: Conditioning,
) -> EmbeddingConditioningSpec | None:
    """Resolve generic embedding configuration while leaving raw modes unrouted.

    :param conditioning: Legacy literal, parsed spec, or Hydra mapping.
    :returns: Fixed-shape embedding spec, or ``None`` for a raw observation.
    :raises TypeError: If ``conditioning`` is neither a supported literal nor mapping.
    :raises ValueError: If an unsupported string literal is provided.
    """
    if isinstance(conditioning, str):
        if conditioning in RAW_CONDITIONING_MODES:
            return None
        if conditioning == "m2l":
            return EmbeddingConditioningSpec(
                column="music2latent", input_shape=LEGACY_M2L_INPUT_SHAPE
            )
        raise ValueError(f"unknown conditioning mode {conditioning!r}")
    if isinstance(conditioning, EmbeddingConditioningSpec):
        return conditioning
    if not isinstance(conditioning, Mapping):
        raise TypeError(
            f"conditioning must be 'mel', 'm2l', 'audio', or a mapping, got {conditioning!r}"
        )

    values = dict(conditioning)
    input_shape = values.get("input_shape")
    if isinstance(input_shape, Sequence) and not isinstance(input_shape, (str, tuple)):
        values["input_shape"] = tuple(input_shape)
    return EmbeddingConditioningSpec.model_validate(values)
