"""Restoring a pretrained flow into a post-training module, and remembering which one it was."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import torch
from beartype import beartype
from jaxtyping import jaxtyped

from synth_setter.models.vst_flow_matching_module import (
    _LEGACY_PARAMETERIZATION,
    _PARAMETERIZATION_KEY,
)

logger = logging.getLogger(__name__)

_BASE_CHECKPOINT_SOURCE_ENV = "SYNTH_SETTER_BASE_CHECKPOINT_SOURCE"
_FROZEN_BACKBONE_PREFIX = "encoder.backbone."


@jaxtyped(typechecker=beartype)
def checkpoint_source_uri(checkpoint: str | Path) -> str:
    """Return a credential-free source URI for a materialized checkpoint.

    :param checkpoint: Materialized checkpoint path used when no source override is present.
    :returns: Source URI without user info, query parameters, or a fragment.
    """
    source = os.getenv(_BASE_CHECKPOINT_SOURCE_ENV)
    if source is None:
        return Path(checkpoint).expanduser().resolve(strict=True).as_uri()
    parsed = urlsplit(source)
    if not parsed.scheme:
        return Path(source).expanduser().absolute().as_uri()
    authority = parsed.netloc.rsplit("@", maxsplit=1)[-1]
    return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


@jaxtyped(typechecker=beartype)
def load_pretrained_flow(module: torch.nn.Module, checkpoint: str | Path) -> str:
    """Restore every pretrained weight, refusing a checkpoint that does not fit.

    Call before attaching any extra submodule, so the module's own shape is exactly the
    base run's: any missing or unexpected key means the wrong checkpoint, and a silent
    ``strict=False`` here would "finetune" a randomly initialised field.

    :param module: Freshly built module of the base run's shape.
    :param checkpoint: Path to a Lightning checkpoint of the base run.
    :returns: SHA-256 hex digest of the checkpoint file, the base's identity for provenance.
    :raises ValueError: The payload has no ``state_dict``, its keys do not match, or it
        trained a non-velocity parameterization.
    """
    with Path(checkpoint).open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    # The config records a mutable path, so without this two arms started from
    # different flows would still read as comparable runs.
    logger.info("base_checkpoint path=%s sha256=%s", checkpoint, digest)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint} holds no Lightning state_dict")
    # load_state_dict bypasses the base module's load hook, and endpoint weights have
    # velocity shapes, so without this check they would freeze as a velocity field.
    stamped = payload.get(_PARAMETERIZATION_KEY, _LEGACY_PARAMETERIZATION)
    if stamped != "velocity":
        raise ValueError(
            f"{checkpoint} trained parameterization={stamped!r}; post-training arms "
            "require a velocity base"
        )
    result = module.load_state_dict(state, strict=False)
    # A frozen pretrained backbone is stripped on save and re-resolved from its own
    # weights, so its absence is expected; nothing else may be.
    missing = [k for k in result.missing_keys if not k.startswith(_FROZEN_BACKBONE_PREFIX)]
    if missing or result.unexpected_keys:
        raise ValueError(
            f"{checkpoint} does not match this model: "
            f"{len(missing)} missing key(s) {missing[:5]}, "
            f"{len(result.unexpected_keys)} unexpected key(s) {result.unexpected_keys[:5]}"
        )
    return digest


class PretrainedBaseMixin:
    """Lightning hooks that tie a post-training module to the base checkpoint it refines.

    Mix in ahead of the LightningModule. The module sets ``base_checkpoint_sha256`` in its
    constructor (``None`` when no base was loaded), and the hooks then refuse a fresh fit
    without any weight source, record the base identity in every saved checkpoint, and reject
    a checkpoint refined from a different base.

    .. attribute :: base_checkpoint_sha256

       SHA-256 of the loaded base, or ``None`` until a saved checkpoint supplies it.

    .. attribute :: base_checkpoint_source

       Credential-free source URI of the loaded base, retained across resume.
    """

    base_checkpoint_sha256: str | None
    base_checkpoint_source: str | None = None

    @jaxtyped(typechecker=beartype)
    def on_fit_start(self) -> None:
        """Refuse a fresh fit that has no pretrained weights to refine.

        :raises ValueError: Neither ``base_checkpoint`` nor a resume checkpoint supplies them.
        """
        trainer: Any = self.trainer  # pyright: ignore[reportAttributeAccessIssue]
        if self.base_checkpoint_sha256 is None and not trainer.ckpt_path:
            raise ValueError(
                "base_checkpoint is required to start post-training; omit it only when "
                "ckpt_path restores a saved run"
            )

    @jaxtyped(typechecker=beartype)
    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Record which base this run refines, so a swapped base file cannot resume it.

        :param checkpoint: Mutable Lightning checkpoint payload.
        """
        super().on_save_checkpoint(checkpoint)  # pyright: ignore[reportAttributeAccessIssue]
        checkpoint["base_checkpoint_sha256"] = self.base_checkpoint_sha256
        checkpoint["base_checkpoint_source"] = self.base_checkpoint_source

    @jaxtyped(typechecker=beartype)
    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Adopt the saved base identity, refusing a checkpoint refined from another base.

        :param checkpoint: Mutable Lightning checkpoint payload.
        :raises ValueError: The configured base differs from the one the checkpoint records.
        """
        super().on_load_checkpoint(checkpoint)  # pyright: ignore[reportAttributeAccessIssue]
        saved = checkpoint.get("base_checkpoint_sha256")
        if not isinstance(saved, str):
            return
        if self.base_checkpoint_sha256 is None:
            self.base_checkpoint_sha256 = saved
        elif saved != self.base_checkpoint_sha256:
            raise ValueError(
                "base_checkpoint does not match the base this checkpoint was refined from "
                f"(configured sha256 {self.base_checkpoint_sha256[:12]}…, saved {saved[:12]}…)"
            )
        saved_source = checkpoint.get("base_checkpoint_source")
        if isinstance(saved_source, str):
            self.base_checkpoint_source = saved_source
