"""Restoring a pretrained flow into a post-training module, and remembering which one it was."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
from beartype import beartype
from jaxtyping import jaxtyped

logger = logging.getLogger(__name__)

_FROZEN_BACKBONE_PREFIX = "encoder.backbone."


@jaxtyped(typechecker=beartype)
def load_pretrained_flow(module: torch.nn.Module, checkpoint: str | Path) -> str:
    """Restore every pretrained weight, refusing a checkpoint that does not fit.

    Call before attaching any extra submodule, so the module's own shape is exactly the
    base run's: any missing or unexpected key means the wrong checkpoint, and a silent
    ``strict=False`` here would "finetune" a randomly initialised field.

    :param module: Freshly built module of the base run's shape.
    :param checkpoint: Path to a Lightning checkpoint of the base run.
    :returns: SHA-256 hex digest of the checkpoint file, the base's identity for provenance.
    :raises ValueError: The payload has no ``state_dict``, or its keys do not match.
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
    """

    base_checkpoint_sha256: str | None

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
