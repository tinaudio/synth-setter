"""Versioned model-construction metadata for trusted Lightning checkpoints.

A checkpoint and its bundled Hydra ``_target_`` graph are executable input. Load
only artifacts from trusted sources; validation checks the bundle contract, not
whether imported targets or pickle payloads are safe.

Typical usage::

    model = load_model_checkpoint(Path("/trusted/model.ckpt"))
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, cast

import hydra
import torch
from beartype import beartype
from jaxtyping import jaxtyped
from lightning import Callback, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

MODEL_BUNDLE_KEY = "synth_setter_model_bundle"
MODEL_BUNDLE_SCHEMA_VERSION = 1

type ModelConfigValue = (
    str | int | float | bool | None | list[ModelConfigValue] | dict[str, ModelConfigValue]
)


class ModelCheckpointBundle(BaseModel, strict=True, extra="forbid"):
    """Validated construction metadata embedded in a model checkpoint.

    .. attribute :: schema_version

        Supported bundle format version.

    .. attribute :: model

        Resolved model-only Hydra construction graph.
    """

    schema_version: Literal[1]
    model: dict[str, ModelConfigValue]


@dataclass(frozen=True)
class LoadedModelCheckpoint:
    """Validated trusted checkpoint payload retained for one reconstruction.

    .. attribute :: checkpoint_path

        Source path used in validation errors.

    .. attribute :: payload

        Deserialized Lightning checkpoint mapping.

    .. attribute :: model_config

        Validated model construction config.
    """

    checkpoint_path: Path
    payload: dict[str, object]
    model_config: DictConfig


@jaxtyped(typechecker=beartype)
def canonical_model_config(model_config: DictConfig) -> DictConfig:
    """Return a resolved model construction config containing only plain values.

    :param model_config: Model-only Hydra construction config.
    :returns: Canonical config used for both construction and persistence.
    :raises TypeError: If the config contains enums or is not a string-keyed mapping.
    """
    resolved = OmegaConf.to_container(
        model_config,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(resolved, dict) or not all(isinstance(key, str) for key in resolved):
        raise TypeError("model config must resolve to a string-keyed mapping")
    pending: list[object] = [resolved]
    while pending:
        value = pending.pop()
        if isinstance(value, Enum):
            raise TypeError("model config enum values are unsupported; use their string names")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    validated = ModelCheckpointBundle(
        schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
        model=cast(dict[str, ModelConfigValue], resolved),
    )
    return OmegaConf.create(validated.model)


class ModelCheckpointBundleCallback(Callback):
    """Embed the resolved model-only Hydra config in every Lightning checkpoint."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, model_config: DictConfig) -> None:
        """Retain a model config for resolution when each checkpoint is saved.

        :param model_config: Model-only Hydra construction config.
        """
        self._model_config = canonical_model_config(model_config)

    @jaxtyped(typechecker=beartype)
    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, object],
    ) -> None:
        """Add versioned, resolved model construction metadata.

        :param trainer: Trainer writing the checkpoint.
        :param pl_module: Model whose state is being saved.
        :param checkpoint: Mutable Lightning checkpoint payload.
        """
        del trainer, pl_module
        checkpoint[MODEL_BUNDLE_KEY] = ModelCheckpointBundle(
            schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
            model=cast(
                dict[str, ModelConfigValue],
                OmegaConf.to_container(self._model_config),
            ),
        ).model_dump(mode="python")


@jaxtyped(typechecker=beartype)
def _validated_bundle(
    checkpoint: Mapping[str, object], checkpoint_path: Path
) -> ModelCheckpointBundle:
    payload = checkpoint.get(MODEL_BUNDLE_KEY)
    if payload is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no {MODEL_BUNDLE_KEY!r} model config. "
            "Legacy checkpoints are smoke-only and cannot reconstruct a model; provide the "
            "original model config and use Lightning directly."
        )
    try:
        return ModelCheckpointBundle.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has an invalid model bundle: {exc}"
        ) from exc


@jaxtyped(typechecker=beartype)
def _load_trusted_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} must contain a mapping")
    return checkpoint


@jaxtyped(typechecker=beartype)
def load_trusted_model_checkpoint(  # noqa: DOC502 — bundle validation errors propagate
    checkpoint_path: Path,
) -> LoadedModelCheckpoint:
    """Deserialize and validate one trusted checkpoint for model reconstruction.

    :param checkpoint_path: Local checkpoint from a trusted source.
    :returns: Validated payload and canonical model construction config.
    :raises ValueError: If bundle metadata has an invalid contract.
    """
    payload = _load_trusted_checkpoint(checkpoint_path)
    bundle = _validated_bundle(payload, checkpoint_path)
    return LoadedModelCheckpoint(checkpoint_path, payload, OmegaConf.create(bundle.model))


@jaxtyped(typechecker=beartype)
def load_model_config(  # noqa: DOC502 — bundle validation errors propagate
    checkpoint: Path | LoadedModelCheckpoint,
) -> DictConfig:
    """Return validated model construction config from a trusted checkpoint.

    The checkpoint pickle is executable input. Callers must establish artifact trust before
    invoking this function.

    :param checkpoint: Local trusted checkpoint or an already validated payload.
    :returns: Resolved model-only Hydra construction config.
    :raises ValueError: If bundle metadata has an invalid contract.
    """
    loaded = (
        checkpoint
        if isinstance(checkpoint, LoadedModelCheckpoint)
        else load_trusted_model_checkpoint(checkpoint)
    )
    return loaded.model_config


@jaxtyped(typechecker=beartype)
def assert_model_config_compatible(
    current_model_config: DictConfig,
    bundled_model_config: DictConfig,
) -> None:
    """Reject a current model config that differs from bundled construction metadata.

    :param current_model_config: Model config selected by the current runtime.
    :param bundled_model_config: Authoritative config loaded from the checkpoint.
    :raises ValueError: If the resolved construction configs differ.
    """
    current = OmegaConf.to_container(canonical_model_config(current_model_config))
    bundled = OmegaConf.to_container(canonical_model_config(bundled_model_config))
    if current != bundled:
        raise ValueError(
            "Current cfg.model is incompatible with the checkpoint's bundled model config. "
            "Use the original model/experiment selection and keep current data, trainer, and "
            "logging overrides separate."
        )


@jaxtyped(typechecker=beartype)
def load_model_checkpoint(  # noqa: DOC503 — RuntimeError propagates from strict state loading
    checkpoint: Path | LoadedModelCheckpoint,
    *,
    expected_model_config: DictConfig | None = None,
) -> LightningModule:
    """Reconstruct and strictly load a model from a trusted bundled checkpoint.

    The checkpoint pickle and bundled Hydra targets can execute code. Callers must establish
    artifact trust before invoking this loader.

    :param checkpoint: Local trusted checkpoint or an already validated payload.
    :param expected_model_config: Optional current config required to match the bundle.
    :returns: Hydra-reconstructed model with the complete strict state restored.
    :raises ValueError: If bundle metadata or checkpoint state has an invalid contract.
    :raises RuntimeError: If strict state loading finds missing or unexpected tensors.
    """
    loaded = (
        checkpoint
        if isinstance(checkpoint, LoadedModelCheckpoint)
        else load_trusted_model_checkpoint(checkpoint)
    )
    if expected_model_config is not None:
        assert_model_config_compatible(expected_model_config, loaded.model_config)
    model = hydra.utils.instantiate(loaded.model_config)
    if not isinstance(model, LightningModule):
        raise ValueError(
            f"Checkpoint {loaded.checkpoint_path} model target constructed "
            f"{type(model).__name__}, not a LightningModule"
        )
    model.on_load_checkpoint(loaded.payload)
    state_dict = loaded.payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Checkpoint {loaded.checkpoint_path} has no mapping state_dict")
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict), strict=True)
    return model
