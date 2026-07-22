"""Strict config for the ``add_embeddings`` m2l+clap / SAME augmenter endpoint.

The Hydra ``add_embeddings.yaml`` composes a dict; the entrypoint builds this
model from it via :meth:`AddEmbeddingsConfig.from_hydra_cfg` (mirroring
``DatasetSpec.from_hydra_cfg``) so the CLI is a thin Hydra→pydantic shell. A
non-empty ``same_variants`` runs the SAME path; otherwise the m2l+clap path runs.

Example::

    cfg = compose(config_name="add_embeddings", overrides=["lance_uri=train.lance"])
    config = AddEmbeddingsConfig.from_hydra_cfg(cfg)
    add_embeddings(config)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    DEFAULT_CLAP_CHECKPOINT,
    DEFAULT_INDEX_METRIC,
    DEFAULT_LANCE_BATCH_SIZE,
    DEFAULT_NUM_SUB_VECTORS,
    DEFAULT_SAME_L_CHECKPOINT,
    DEFAULT_SAME_S_CHECKPOINT,
)

# The SAME variant tokens the endpoint accepts, each mapping to a Lance column.
SAME_VARIANT_CHOICES: frozenset[str] = frozenset({"s", "l"})

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

    .. attribute :: same_variants

        SAME variants to write (``"s"``/``"l"``); a non-empty tuple runs the SAME
        path instead of m2l+clap. Order sets the column-write order.

    .. attribute :: same_s_checkpoint

        SAME-S checkpoint (local dir, ``r2://`` mirror, or HuggingFace repo id).

    .. attribute :: same_l_checkpoint

        SAME-L checkpoint (local dir, ``r2://`` mirror, or HuggingFace repo id).
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
        default=None, ge=1, description="IVF partition count; ``None`` uses ``round(sqrt(rows))``."
    )
    num_sub_vectors: int = Field(
        default=DEFAULT_NUM_SUB_VECTORS,
        ge=1,
        description="PQ sub-vector count; must divide the clap dim.",
    )
    metric: str = Field(
        default=DEFAULT_INDEX_METRIC, description="IVF_PQ distance metric Lance accepts."
    )
    resume_cache: Path | None = Field(
        default=None, description="Per-batch UDF-output cache enabling resume; deleted on success."
    )
    debug: bool = Field(
        default=False, description="Log every batch and enable native Lance debug telemetry."
    )
    same_variants: tuple[str, ...] = Field(
        default=(),
        description='SAME variants ("s"/"l"); non-empty runs the SAME path instead of m2l+clap.',
    )
    same_s_checkpoint: str = Field(
        default=DEFAULT_SAME_S_CHECKPOINT,
        description="SAME-S checkpoint (local dir, ``r2://`` mirror, or HF repo id).",
    )
    same_l_checkpoint: str = Field(
        default=DEFAULT_SAME_L_CHECKPOINT,
        description="SAME-L checkpoint (local dir, ``r2://`` mirror, or HF repo id).",
    )

    @field_validator("same_variants", mode="before")
    @classmethod
    def _coerce_and_check_same_variants(cls, value: object) -> object:
        """Coerce a Hydra list to a tuple and reject unknown or duplicate variants.

        A Hydra ``same_variants=[s,l]`` override reaches the model as a ``list``,
        which strict mode rejects for a ``tuple`` field; coerce it and validate
        membership before the strict schema runs so a bad token fails at config
        time, not after the multi-GB SAME weights download.

        :param value: Raw ``same_variants`` value from the composed cfg.
        :returns: A ``tuple`` of variant tokens for a list/tuple input, else ``value``.
        :raises ValueError: A token is not ``"s"``/``"l"`` or appears more than once.
        """
        if not isinstance(value, (list, tuple)):
            return value
        variants = tuple(value)
        unknown = [v for v in variants if v not in SAME_VARIANT_CHOICES]
        if unknown:
            raise ValueError(
                f"same_variants {unknown} must each be one of {sorted(SAME_VARIANT_CHOICES)}"
            )
        if len(set(variants)) != len(variants):
            raise ValueError(f"same_variants {list(variants)} has duplicate entries")
        return variants

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

    @field_validator("metric")
    @classmethod
    def _metric_is_supported(cls, value: str) -> str:
        """Reject a distance metric Lance's IVF_PQ build would not accept.

        :param value: Configured distance metric.
        :returns: ``value`` unchanged when supported.
        :raises ValueError: If ``value`` is not one of ``cosine``/``l2``/``dot``.
        """
        allowed = {"cosine", "l2", "dot"}
        if value not in allowed:
            raise ValueError(f"metric {value!r} must be one of {sorted(allowed)}")
        return value

    @field_validator("num_sub_vectors")
    @classmethod
    def _num_sub_vectors_divides_clap_dim(cls, value: int) -> int:
        """Reject a PQ sub-vector count that cannot evenly split the clap vector.

        Lance's IVF_PQ build requires ``clap_dim % num_sub_vectors == 0``; check
        it at config time so a bad value fails before the render+encode, not after.

        :param value: Configured PQ sub-vector count.
        :returns: ``value`` unchanged when it divides the clap dimensionality.
        :raises ValueError: If ``value`` does not divide ``CLAP_EMBEDDING_DIM``.
        """
        if CLAP_EMBEDDING_DIM % value != 0:
            raise ValueError(
                f"num_sub_vectors ({value}) must divide the clap dim ({CLAP_EMBEDDING_DIM})"
            )
        return value

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
