"""Strict config for registry-selected Lance embedding augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.data.vst.param_text import (
    DEFAULT_PARAM_TEXT_NORMALIZER,
    PARAM_TEXT_NORMALIZERS,
)
from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from synth_setter.pipeline.data.add_embeddings import (
    DEFAULT_INDEX_METRIC,
    DEFAULT_LANCE_BATCH_SIZE,
    EMBEDDING_REGISTRY,
    SKETCH_ENCODE_MAX_BATCH,
)
from synth_setter.pipeline.schemas.spec import RenderConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = ["AddEmbeddingsConfig"]


class AddEmbeddingsConfig(BaseModel):
    """Validate one registry-driven embedding augmentation run.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: lance_uri

        One finalized Lance split to augment.

    .. attribute :: embeddings

        Ordered registry keys to write.

    .. attribute :: checkpoints

        Per-registry-key checkpoint overrides; unsupported for checkpoint-free policies.

    .. attribute :: device

        Torch device, or ``None`` for automatic selection.

    .. attribute :: batch_size

        Rows per Lance UDF call.

    .. attribute :: num_workers

        Worker processes for CPU-bound registry encoders; ``1`` keeps them in-process.

    .. attribute :: sketch_encode_chunk

        Rows per sketch extractor invocation.

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

    .. attribute :: param_spec_name

        Param spec describing ``param_array``; required by param-sourced embeddings.

    .. attribute :: param_text_normalizer

        Strategy rendering param rows as conditioning text.

    .. attribute :: render

        Composed render and synth identity, or ``None`` when nothing re-renders.

    .. attribute :: param_shift_seed

        Master seed for ``param_shift``'s per-row replacement draws.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    lance_uri: str = Field(description="One finalized Lance dataset to augment.")
    embeddings: tuple[str, ...] = Field(
        default=("clap", "m2l"), description="Ordered embedding registry keys to write."
    )
    checkpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Checkpoint overrides keyed by registry name; checkpoint-free entries reject them.",
    )
    device: str | None = Field(default=None, description="Torch device; null auto-selects.")
    batch_size: int = Field(
        default=DEFAULT_LANCE_BATCH_SIZE, ge=1, description="Rows per Lance UDF call."
    )
    num_workers: int = Field(
        default=1,
        ge=1,
        description="Worker processes for CPU-bound encoders; torch/GPU encoders ignore it.",
    )
    sketch_encode_chunk: int = Field(
        default=SKETCH_ENCODE_MAX_BATCH,
        ge=1,
        description="Rows per sketch extractor invocation; sizes memory and GPU utilization.",
    )
    build_index: bool = Field(
        default=True, description="Build indexes declared by selected embedding specs."
    )
    num_partitions: int | None = Field(
        default=None, ge=1, description="IVF partition override; null derives it from rows."
    )
    num_sub_vectors: int | None = Field(
        default=None, ge=1, description="PQ sub-vector override; null uses each spec's default."
    )
    metric: str = Field(default=DEFAULT_INDEX_METRIC, description="Vector metric override.")
    resume_cache: Path | None = Field(
        default=None, description="Lance UDF checkpoint cache removed after commit."
    )
    debug: bool = Field(default=False, description="Enable per-batch and native debug logs.")
    param_spec_name: str | None = Field(
        default=None, description="Param spec describing param_array; null unless text-sourced."
    )
    param_text_normalizer: str = Field(
        default=DEFAULT_PARAM_TEXT_NORMALIZER,
        description="Strategy rendering param rows as conditioning text.",
    )
    render: RenderConfig | None = Field(
        default=None,
        description="Composed render/synth selection; required by re-rendering embeddings.",
    )
    param_shift_seed: int = Field(
        default=0, description="Master seed for param_shift's per-row replacement draws."
    )

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
        if "m2l" in value:
            raise ValueError("music2latent does not support checkpoint overrides")
        if "pyfdn_sketch" in value:
            raise ValueError("pyfdn_sketch is checkpoint-free and rejects checkpoint overrides")
        return value

    @field_validator("resume_cache", mode="before")
    @classmethod
    def _coerce_resume_cache(cls, value: object) -> object:
        """Coerce Hydra string overrides to ``Path`` under strict validation.

        :param value: Raw path value.
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

    @field_validator("param_spec_name")
    @classmethod
    def _param_spec_name_is_registered(cls, value: str | None) -> str | None:
        """Reject a param spec that no registry entry can resolve.

        :param value: Configured param spec name.
        :returns: The name unchanged when it is registered or unset.
        :raises ValueError: The name is absent from the param-spec registry.
        """
        if value is not None and value not in param_specs:
            raise ValueError(f"param_spec_name {value!r} must be one of {sorted(param_specs)}")
        return value

    @field_validator("param_text_normalizer")
    @classmethod
    def _param_text_normalizer_is_registered(cls, value: str) -> str:
        """Reject an unregistered text normalizer.

        :param value: Configured normalizer name.
        :returns: The name unchanged when it is registered.
        :raises ValueError: The name is absent from the normalizer registry.
        """
        if value not in PARAM_TEXT_NORMALIZERS:
            raise ValueError(
                f"param_text_normalizer {value!r} must be one of {sorted(PARAM_TEXT_NORMALIZERS)}"
            )
        return value

    @model_validator(mode="after")
    def _param_sourced_embeddings_need_a_param_spec(self) -> Self:
        """Require a param spec whenever a selected embedding reads param rows.

        Re-rendering embeddings are exempt: their spec comes from the render config's synth
        identity, so a second, independently-set name could only disagree with it.

        :returns: Validated config unchanged.
        :raises ValueError: A param-sourced embedding is selected without a param spec.
        """
        param_sourced = sorted(
            name
            for name, spec in ((name, EMBEDDING_REGISTRY[name]) for name in self.embeddings)
            if PARAM_ARRAY_FIELD in spec.input_fields and not spec.rerenders
        )
        if param_sourced and self.param_spec_name is None:
            raise ValueError(f"embeddings {param_sourced} require param_spec_name")
        return self

    @model_validator(mode="after")
    def _rerendering_embeddings_need_a_render_config(self) -> Self:
        """Require a composed render config whenever a selected embedding re-renders audio.

        :returns: Validated config unchanged.
        :raises ValueError: A re-rendering embedding is selected without a render config.
        """
        rerendering = sorted(
            name for name in self.embeddings if EMBEDDING_REGISTRY[name].rerenders
        )
        if rerendering and self.render is None:
            raise ValueError(
                f"embeddings {rerendering} re-render every row and require a composed render "
                "config; pass `render=<group> synth=<group>`"
            )
        return self

    @model_validator(mode="after")
    def _num_sub_vectors_divides_selected_fixed_vector_dims(self) -> Self:
        """Reject incompatible PQ splits for selected fixed-width vectors.

        :returns: Validated config unchanged.
        :raises ValueError: The count cannot evenly split a selected vector.
        """
        if self.num_sub_vectors is None:
            return self
        for name in self.embeddings:
            index = EMBEDDING_REGISTRY[name].index
            dim = None if index is None else index.vector_dim
            if dim is not None and dim % self.num_sub_vectors != 0:
                raise ValueError(
                    f"num_sub_vectors ({self.num_sub_vectors}) must divide the {name} dim ({dim})"
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

        # ``render`` composes from two root groups (#2565), so it is masked out here and
        # rebuilt by ``RenderConfig.from_cfg_nodes`` rather than validated field-wise.
        spec_keys = [
            key
            for key in cfg
            if isinstance(key, str) and key in cls.model_fields and key != "render"
        ]
        try:
            masked = OmegaConf.masked_copy(cfg, spec_keys)
        except ValueError as exc:
            raise TypeError(f"composed config is not a mapping: {type(cfg).__name__}") from exc
        raw = OmegaConf.to_container(masked, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"composed config is not a mapping: {type(raw).__name__}")
        values = {key: value for key, value in raw.items() if isinstance(key, str)}
        if cfg.get("render") is not None:
            values["render"] = RenderConfig.from_cfg_nodes(cfg.render, cfg.get("synth"))
        return cls(**values)
