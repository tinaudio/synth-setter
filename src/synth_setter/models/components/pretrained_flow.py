"""Strict restoration of a pretrained flow's weights into a module of the same shape."""

import hashlib
import logging
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import jaxtyped

logger = logging.getLogger(__name__)

_FROZEN_BACKBONE_PREFIX = "encoder.backbone."


@jaxtyped(typechecker=beartype)
def load_pretrained_flow(module: torch.nn.Module, checkpoint: str | Path) -> None:
    """Restore every pretrained weight, refusing a checkpoint that does not fit.

    Call before attaching any extra submodule, so the module's own shape is exactly the
    base run's: any missing or unexpected key means the wrong checkpoint, and a silent
    ``strict=False`` here would "finetune" a randomly initialised field.

    :param module: Freshly built module of the base run's shape.
    :param checkpoint: Path to a Lightning checkpoint of the base run.
    :raises ValueError: The payload has no ``state_dict``, or its keys do not match.
    """
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
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
