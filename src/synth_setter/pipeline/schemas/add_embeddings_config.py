"""Strict config for the ``add_embeddings`` m2l+clap augmenter endpoint.

The Hydra ``add_embeddings.yaml`` composes a dict; the entrypoint builds this
model from it via :meth:`AddEmbeddingsConfig.from_hydra_cfg` (mirroring
``DatasetSpec.from_hydra_cfg``) so the CLI is a thin Hydra→pydantic shell. SAME
knobs are intentionally absent — SAME dispatch is a stacked follow-up (#2319).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth_setter.pipeline.data.add_embeddings import (
    DEFAULT_CLAP_CHECKPOINT,
    DEFAULT_INDEX_METRIC,
    DEFAULT_LANCE_BATCH_SIZE,
    DEFAULT_NUM_SUB_VECTORS,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = ["AddEmbeddingsConfig"]


class AddEmbeddingsConfig(BaseModel):
    """Validated knobs for one m2l+clap embedding-augmentation run.

    Strict trust boundary for the Hydra-composed dict: ``strict`` rejects loose
    coercions, ``extra="forbid"`` rejects stray keys, ``frozen`` makes the run
    config immutable post-construction.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below.

    .. attribute :: lance_uri

        Dataset directory to augment (local path, ``r2://``, or ``s3://``).

    .. attribute :: clap_checkpoint

        HuggingFace CLAP model id whose audio tower sets the ``clap`` width.

    .. attribute :: device

        Torch device for both encoders; ``None`` auto-selects cuda, MPS, then cpu.

    .. attribute :: batch_size

        Rows per UDF call (ignored for legacy v1 Lance datasets).

    .. attribute :: build_index

        Build an IVF_PQ index on the ``clap`` column after the column lands.

    .. attribute :: num_partitions

        IVF partition count; ``None`` uses ``round(sqrt(rows))``.

    .. attribute :: num_sub_vectors

        PQ sub-vector count; must divide the ``clap`` dim.

    .. attribute :: metric

        Vector-index distance metric.

    .. attribute :: resume_cache

        Per-batch UDF-output cache; a rerun with the same file resumes an
        interrupted run. Deleted after a successful commit.

    .. attribute :: debug

        Log every batch with stage timings and enable native Lance debug telemetry.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(description="Dataset to augment (local path, ``r2://``, or ``s3://``).")
    clap_checkpoint: str = Field(
        default=DEFAULT_CLAP_CHECKPOINT, description="HuggingFace CLAP model id."
    )
    device: str | None = Field(
        default=None, description="Torch device for both encoders; ``None`` auto-selects."
    )
    batch_size: int = Field(
        default=DEFAULT_LANCE_BATCH_SIZE, ge=1, description="Rows per UDF call."
    )
    build_index: bool = Field(
        default=True, description="Build an IVF_PQ index on the clap column."
    )
    num_partitions: int | None = Field(
        default=None, description="IVF partition count; ``None`` uses ``round(sqrt(rows))``."
    )
    num_sub_vectors: int = Field(
        default=DEFAULT_NUM_SUB_VECTORS,
        description="PQ sub-vector count; must divide the clap dim.",
    )
    metric: str = Field(default=DEFAULT_INDEX_METRIC, description="Vector-index distance metric.")
    resume_cache: Path | None = Field(
        default=None, description="Per-batch UDF-output cache enabling resume; deleted on success."
    )
    debug: bool = Field(
        default=False, description="Log every batch and enable native Lance debug telemetry."
    )

    @field_validator("resume_cache", mode="before")
    @classmethod
    def _coerce_resume_cache(cls, value: object) -> object:
        """Accept a Hydra string override for the ``Path`` field under ``strict``.

        A Hydra override (``resume_cache=/tmp/x``) reaches the model as ``str``,
        which strict mode would reject; coerce it to ``Path`` before validation.

        :param value: Raw ``resume_cache`` value from the composed cfg.
        :returns: ``Path`` for a string input, else ``value`` unchanged.
        """
        return Path(value) if isinstance(value, str) else value

    @classmethod
    def from_hydra_cfg(cls, cfg: DictConfig) -> AddEmbeddingsConfig:
        """Build from a Hydra-composed cfg, masking to model fields before resolving.

        Mirrors ``DatasetSpec.from_hydra_cfg``: masking to ``cls.model_fields``
        *before* ``resolve=True`` keeps non-spec groups (``paths``, ``hydra``)
        from being evaluated, so the config resolves under a plain ``compose()``.

        :param cfg: Composed cfg; only keys matching ``cls.model_fields`` survive.
        :returns: Validated config built from the masked, resolved mapping.
        :raises TypeError: ``cfg`` is not mapping-shaped, or the masked cfg did
            not resolve to a mapping.
        """
        from omegaconf import OmegaConf

        spec_keys = [k for k in cfg if isinstance(k, str) and k in cls.model_fields]
        try:
            masked = OmegaConf.masked_copy(cfg, spec_keys)
        except ValueError as exc:
            raise TypeError(f"composed config is not a mapping: {type(cfg).__name__}") from exc
        raw = OmegaConf.to_container(masked, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"composed config is not a mapping: {type(raw).__name__}")
        return cls(**{k: v for k, v in raw.items() if isinstance(k, str)})
