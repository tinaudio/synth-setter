"""Versioned model-construction metadata for trusted Lightning checkpoints.

A checkpoint and its bundled Hydra ``_target_`` graph are executable input. Load
only artifacts from trusted sources; validation checks the bundle contract, not
whether imported targets or pickle payloads are safe.

Typical usage::

    model = load_model_checkpoint(Path("/trusted/model.ckpt"))
"""

from __future__ import annotations

from collections.abc import Mapping
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


class ModelCheckpointBundleCallback(Callback):
    """Embed the resolved model-only Hydra config in every Lightning checkpoint."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, model_config: DictConfig) -> None:
        """Retain a model config for resolution when each checkpoint is saved.

        :param model_config: Model-only Hydra construction config.
        """
        self._model_config = model_config

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
        :raises TypeError: If the model config does not resolve to a string-keyed mapping.
        """
        del trainer, pl_module
        resolved = OmegaConf.to_container(
            self._model_config,
            resolve=True,
            throw_on_missing=True,
            enum_to_str=True,
        )
        if not isinstance(resolved, dict) or not all(isinstance(key, str) for key in resolved):
            raise TypeError("model config must resolve to a string-keyed mapping")
        checkpoint[MODEL_BUNDLE_KEY] = ModelCheckpointBundle(
            schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
            model=cast(dict[str, ModelConfigValue], resolved),
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
def load_model_config(checkpoint_path: Path) -> DictConfig:  # noqa: DOC502 — ValueError propagates from bundle validation
    """Return validated model construction config from a trusted checkpoint.

    The checkpoint pickle is executable input. Callers must establish artifact trust before
    invoking this function.

    :param checkpoint_path: Local checkpoint from a trusted source.
    :returns: Resolved model-only Hydra construction config.
    :raises ValueError: If bundle metadata has an invalid contract.
    """
    checkpoint = _load_trusted_checkpoint(checkpoint_path)
    bundle = _validated_bundle(checkpoint, checkpoint_path)
    return OmegaConf.create(bundle.model)


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
    current = OmegaConf.to_container(current_model_config, resolve=True, throw_on_missing=True)
    bundled = OmegaConf.to_container(bundled_model_config, resolve=True, throw_on_missing=True)
    if current != bundled:
        raise ValueError(
            "Current cfg.model is incompatible with the checkpoint's bundled model config. "
            "Use the original model/experiment selection and keep current data, trainer, and "
            "logging overrides separate."
        )


@jaxtyped(typechecker=beartype)
def load_model_checkpoint(  # noqa: DOC503 — RuntimeError propagates from strict state loading
    checkpoint_path: Path,
    *,
    expected_model_config: DictConfig | None = None,
) -> LightningModule:
    """Reconstruct and strictly load a model from a trusted bundled checkpoint.

    The checkpoint pickle and bundled Hydra targets can execute code. Callers must establish
    artifact trust before invoking this loader.

    :param checkpoint_path: Local checkpoint from a trusted source.
    :param expected_model_config: Optional current config required to match the bundle.
    :returns: Hydra-reconstructed model with the complete strict state restored.
    :raises ValueError: If bundle metadata or checkpoint state has an invalid contract.
    :raises RuntimeError: If strict state loading finds missing or unexpected tensors.
    """
    checkpoint = _load_trusted_checkpoint(checkpoint_path)
    bundle = _validated_bundle(checkpoint, checkpoint_path)
    bundled_model_config = OmegaConf.create(bundle.model)
    if expected_model_config is not None:
        assert_model_config_compatible(expected_model_config, bundled_model_config)
    model = hydra.utils.instantiate(bundled_model_config)
    if not isinstance(model, LightningModule):
        raise ValueError(
            f"Checkpoint {checkpoint_path} model target constructed {type(model).__name__}, "
            "not a LightningModule"
        )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Checkpoint {checkpoint_path} has no mapping state_dict")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict), strict=True)
    return model
