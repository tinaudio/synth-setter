"""Architecture fingerprint sidecar guarding canonical checkpoint overwrites (#2588)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

#: Suffix appended to a checkpoint URI to name its fingerprint sidecar object.
SIDECAR_SUFFIX = ".fingerprint.json"


class CheckpointFingerprint(BaseModel):
    """Architecture identity of one training run's checkpoint.

    Two runs sharing a config_id but differing in any of these fields would produce mutually
    unloadable checkpoints, so a canonical-slot overwrite across a fingerprint mismatch is refused
    (#2588).

    .. attribute :: model_config

        Strict frozen Pydantic configuration; ``protected_namespaces=()`` frees ``model_target``.

    .. attribute :: model_target

        ``model._target_`` — the LightningModule class.

    .. attribute :: encoder_target

        ``model.encoder._target_`` — the conditioning encoder class.

    .. attribute :: vector_field_target

        ``model.vector_field._target_`` — the backbone class.

    .. attribute :: conditioning

        ``model.conditioning`` — mode literal or embedding-spec mapping, as a plain container.

    .. attribute :: param_spec_name

        ``synth.param_spec_name`` — the parameter-space identity.
    """

    model_config = ConfigDict(strict=True, frozen=True, protected_namespaces=())

    model_target: str | None = None
    encoder_target: str | None = None
    vector_field_target: str | None = None
    conditioning: Any = None
    param_spec_name: str | None = None


def fingerprint_from_cfg(cfg: DictConfig) -> CheckpointFingerprint:
    """Extract the run's architecture fingerprint from a composed train cfg.

    Each field degrades to ``None`` when its node is absent, so minimal or
    legacy cfgs still fingerprint (and compare) without raising.

    :param cfg: Hydra-composed train cfg carrying ``model`` and ``synth`` nodes.
    :returns: The fingerprint of the architecture this cfg would train.
    """
    conditioning = OmegaConf.select(cfg, "model.conditioning", default=None)
    if OmegaConf.is_config(conditioning):
        conditioning = OmegaConf.to_container(conditioning, resolve=True)
    return CheckpointFingerprint(
        model_target=OmegaConf.select(cfg, "model._target_", default=None),
        encoder_target=OmegaConf.select(cfg, "model.encoder._target_", default=None),
        vector_field_target=OmegaConf.select(cfg, "model.vector_field._target_", default=None),
        conditioning=conditioning,
        param_spec_name=OmegaConf.select(cfg, "synth.param_spec_name", default=None),
    )


def fingerprint_sidecar_uri(ckpt_uri: str) -> str:
    """Return the sidecar object URI for a checkpoint URI.

    :param ckpt_uri: The canonical ``r2://.../model.ckpt`` URI.
    :returns: The adjacent ``model.ckpt.fingerprint.json`` URI.
    """
    return f"{ckpt_uri}{SIDECAR_SUFFIX}"
