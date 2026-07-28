"""Conditioning contracts shared across data and model layers."""

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

ConditioningMode = Literal["mel", "m2l", "audio"]
LEGACY_M2L_INPUT_SHAPE = (128, 42)
# Batch key each non-embedding mode observes. "audio" serves online-render synths, which
# have no stored mel column because their audio only exists at training time.
_RAW_CONDITIONING_KEYS: dict[str, str] = {"mel": "mel_spec", "audio": "audio"}


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


def raw_conditioning_key(conditioning: Conditioning) -> str:
    """Return the batch key a non-embedding conditioning mode observes.

    :param conditioning: Configured conditioning mode or embedding spec.
    :returns: Batch key; ``"mel_spec"`` for embeddings, which ignore it.
    """
    if isinstance(conditioning, str):
        return _RAW_CONDITIONING_KEYS.get(conditioning, "mel_spec")
    return "mel_spec"


def select_conditioning(
    batch: Mapping[str, torch.Tensor],
    embedding: EmbeddingConditioningSpec | None,
    raw_key: str = "mel_spec",
) -> torch.Tensor:
    """Select the raw-observation or canonical cached-conditioning tensor.

    :param batch: Model batch containing the configured conditioning tensor.
    :param embedding: Resolved cached-embedding spec, or ``None`` for a raw observation.
    :param raw_key: Batch key for the raw observation; see :func:`raw_conditioning_key`.
    :returns: The tensor selected for model conditioning.
    """
    if embedding is None:
        return batch[raw_key]
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
        if conditioning in _RAW_CONDITIONING_KEYS:
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
