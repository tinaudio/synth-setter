"""Strict config for registry-selected Lance embedding augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    DEFAULT_INDEX_METRIC,
    DEFAULT_LANCE_BATCH_SIZE,
    DEFAULT_NUM_SUB_VECTORS,
    EMBEDDING_REGISTRY,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = ["AddEmbeddingsConfig"]


class AddEmbeddingsConfig(BaseModel):
    """Validate one registry-driven embedding augmentation run.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: lance_uri

        Finalized Lance dataset to augment.

    .. attribute :: embeddings

        Ordered registry keys to write.

    .. attribute :: checkpoints

        Per-registry-key checkpoint overrides.

    .. attribute :: device

        Torch device, or ``None`` for automatic selection.

    .. attribute :: batch_size

        Rows per Lance UDF call.

    .. attribute :: build_index

        Whether selected specs may build their declared indexes.

    .. attribute :: num_partitions

        IVF partition override, or ``None`` for a row-derived count.

    .. attribute :: num_sub_vectors

        PQ sub-vector override.

    .. attribute :: metric

        Vector distance-metric override.

    .. attribute :: resume_cache

        Lance UDF checkpoint cache removed after commit.

    .. attribute :: debug

        Whether to log every batch and enable native Lance debug output.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(description="Finalized Lance dataset to augment.")
    embeddings: tuple[str, ...] = Field(
        default=("clap", "m2l"), description="Ordered embedding registry keys to write."
    )
    checkpoints: dict[str, str] = Field(
        default_factory=dict, description="Checkpoint overrides keyed by registry name."
    )
    device: str | None = Field(default=None, description="Torch device; null auto-selects.")
    batch_size: int = Field(
        default=DEFAULT_LANCE_BATCH_SIZE, ge=1, description="Rows per Lance UDF call."
    )
    build_index: bool = Field(
        default=True, description="Build indexes declared by selected embedding specs."
    )
    num_partitions: int | None = Field(
        default=None, ge=1, description="IVF partition override; null derives it from rows."
    )
    num_sub_vectors: int = Field(
        default=DEFAULT_NUM_SUB_VECTORS, ge=1, description="PQ sub-vector override."
    )
    metric: str = Field(default=DEFAULT_INDEX_METRIC, description="Vector metric override.")
    resume_cache: Path | None = Field(
        default=None, description="Lance UDF checkpoint cache removed after commit."
    )
    debug: bool = Field(default=False, description="Enable per-batch and native debug logs.")

    @field_validator("embeddings", mode="before")
    @classmethod
    def _coerce_and_check_embeddings(cls, value: object) -> object:
        """Coerce Hydra lists and reject unknown or duplicate registry keys.

        :param value: Raw embedding selection.
        :returns: Tuple for list or tuple input, otherwise the original value.
        :raises ValueError: A key is unknown or repeated.
        """
        if not isinstance(value, (list, tuple)):
            return value
        embeddings = tuple(value)
        if not embeddings:
            raise ValueError("embeddings must select at least one registry key")
        unknown = [name for name in embeddings if name not in EMBEDDING_REGISTRY]
        if unknown:
            raise ValueError(
                f"embeddings {unknown} must each be one of {sorted(EMBEDDING_REGISTRY)}"
            )
        if len(set(embeddings)) != len(embeddings):
            raise ValueError(f"embeddings {list(embeddings)} has duplicate entries")
        return embeddings

    @field_validator("checkpoints")
    @classmethod
    def _check_checkpoint_keys(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject checkpoint overrides that cannot resolve to a registry entry.

        :param value: Validated checkpoint mapping.
        :returns: Mapping unchanged when every key is known.
        :raises ValueError: A checkpoint key is absent from the registry.
        """
        unknown = sorted(set(value) - set(EMBEDDING_REGISTRY))
        if unknown:
            raise ValueError(
                f"checkpoints keys {unknown} must each be one of {sorted(EMBEDDING_REGISTRY)}"
            )
        return value

    @field_validator("resume_cache", mode="before")
    @classmethod
    def _coerce_resume_cache(cls, value: object) -> object:
        """Coerce a Hydra string override to ``Path`` under strict validation.

        :param value: Raw resume-cache value.
        :returns: Path for a string input, otherwise the original value.
        """
        return Path(value) if isinstance(value, str) else value

    @field_validator("metric")
    @classmethod
    def _metric_is_supported(cls, value: str) -> str:
        """Restrict metrics to Lance IVF_PQ values.

        :param value: Configured metric.
        :returns: Supported metric unchanged.
        :raises ValueError: Lance does not support the metric for IVF_PQ.
        """
        allowed = {"cosine", "l2", "dot"}
        if value not in allowed:
            raise ValueError(f"metric {value!r} must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def _num_sub_vectors_divides_selected_clap_dim(self) -> Self:
        """Reject incompatible PQ splits only when CLAP is selected.

        :returns: Validated config unchanged.
        :raises ValueError: The count cannot evenly split selected CLAP vectors.
        """
        if "clap" in self.embeddings and CLAP_EMBEDDING_DIM % self.num_sub_vectors != 0:
            raise ValueError(
                f"num_sub_vectors ({self.num_sub_vectors}) must divide the clap dim "
                f"({CLAP_EMBEDDING_DIM})"
            )
        return self

    @classmethod
    def from_hydra_cfg(cls, cfg: DictConfig) -> AddEmbeddingsConfig:
        """Validate only this model's fields from a composed Hydra config.

        :param cfg: Hydra mapping containing endpoint and composition keys.
        :returns: Strict validated endpoint config.
        :raises TypeError: The composed config is not mapping-shaped.
        """
        from omegaconf import OmegaConf

        spec_keys = [key for key in cfg if isinstance(key, str) and key in cls.model_fields]
        try:
            masked = OmegaConf.masked_copy(cfg, spec_keys)
        except ValueError as exc:
            raise TypeError(f"composed config is not a mapping: {type(cfg).__name__}") from exc
        raw = OmegaConf.to_container(masked, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"composed config is not a mapping: {type(raw).__name__}")
        return cls(**{key: value for key, value in raw.items() if isinstance(key, str)})
