"""Strict configuration boundary for SLAP retrieval export.

Compose Hydra settings with ``ExportSlapConfig.from_hydra_cfg(cfg)`` before export.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.file_uri import file_uri_to_path, is_file_uri

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _canonical_root(uri: str) -> str:
    """Canonicalize supported roots for alias rejection.

    :param uri: Local path, file URI, or R2-compatible URI.
    :returns: Canonical comparison string.
    :raises ValueError: The URI scheme is unsupported.
    """
    if is_file_uri(uri):
        return f"file://{file_uri_to_path(uri).expanduser().resolve()}"
    if r2_io.is_r2_uri(uri):
        return r2_io.to_s3_uri(uri).rstrip("/")
    if uri.startswith("s3://"):
        return uri.rstrip("/")
    if "://" in uri:
        raise ValueError(f"unsupported dataset URI scheme: {uri!r}")
    return f"file://{Path(uri).expanduser().resolve()}"


class ExportSlapConfig(BaseModel):
    """Validate one split-oriented SLAP retrieval export request.

    .. attribute :: model_config
    .. attribute :: source_root_uri
    .. attribute :: output_root_uri
    .. attribute :: splits
    .. attribute :: source_versions
    .. attribute :: ckpt_path
    .. attribute :: model
    .. attribute :: use_saved_mean_and_variance
    .. attribute :: mel_stats_path
    .. attribute :: device
    .. attribute :: batch_size
    .. attribute :: build_index
    .. attribute :: metric
    .. attribute :: num_partitions
    .. attribute :: num_sub_vectors
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    source_root_uri: str
    output_root_uri: str
    splits: tuple[Literal["train", "val", "test"], ...] = ("train", "val", "test")
    source_versions: dict[str, int]
    ckpt_path: Path
    model: dict[str, object]
    use_saved_mean_and_variance: bool = True
    mel_stats_path: Path | None = None
    device: str | None = None
    batch_size: int = Field(default=256, ge=1)
    build_index: bool = False
    metric: Literal["cosine", "l2", "dot"] = "cosine"
    num_partitions: int | None = Field(default=None, ge=1)
    num_sub_vectors: int = Field(default=16, ge=1)

    @field_validator("splits", mode="before")
    @classmethod
    def _coerce_splits(cls, value: object) -> object:
        """Coerce Hydra lists to the immutable split representation.

        :param value: Raw split selection.
        :returns: Tuple for a list input, otherwise the original value.
        """
        return tuple(value) if isinstance(value, list) else value

    @field_validator("ckpt_path", "mel_stats_path", mode="before")
    @classmethod
    def _coerce_local_path(cls, value: object) -> object:
        """Coerce Hydra local path strings under strict parsing.

        :param value: Raw checkpoint or statistics path.
        :returns: Path for a string input, otherwise the original value.
        """
        return Path(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_request(self) -> Self:
        """Reject ambiguous split pins and source/destination aliasing.

        :returns: Validated request.
        :raises ValueError: Splits are empty, repeated, unpinned, or roots alias.
        """
        if not self.splits:
            raise ValueError("splits must select at least one split")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("splits must not contain duplicates")
        if set(self.source_versions) != set(self.splits):
            raise ValueError("source_versions keys must exactly match splits")
        if any(version < 1 for version in self.source_versions.values()):
            raise ValueError("source_versions must be positive Lance versions")
        if _canonical_root(self.source_root_uri) == _canonical_root(self.output_root_uri):
            raise ValueError("source_root_uri and output_root_uri must differ")
        if not self.ckpt_path.is_file():
            raise ValueError(f"ckpt_path is not a file: {self.ckpt_path}")
        if self.mel_stats_path is not None and not self.mel_stats_path.is_file():
            raise ValueError(f"mel_stats_path is not a file: {self.mel_stats_path}")
        if "_target_" not in self.model:
            raise ValueError("model must contain _target_")
        return self

    @classmethod
    def from_hydra_cfg(cls, cfg: DictConfig) -> ExportSlapConfig:
        """Resolve and validate exporter fields from a Hydra mapping.

        :param cfg: Composed Hydra configuration.
        :returns: Strict exporter configuration.
        :raises TypeError: The composed configuration is not a mapping.
        """
        from omegaconf import OmegaConf

        keys = list(cls.model_fields)
        masked = OmegaConf.masked_copy(cfg, keys)
        raw = OmegaConf.to_container(masked, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError("composed export config must resolve to a mapping")
        return cls.model_validate(raw)
