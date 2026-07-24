"""Resolve shared local cache paths for model artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def synth_setter_cache_dir() -> Path:
    """Return the XDG-aware synth-setter cache root.

    :returns: ``synth-setter`` under ``XDG_CACHE_HOME`` or the conventional
        ``~/.cache`` fallback.
    """
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "synth-setter"


def embedding_model_dir(model_name: str) -> Path:
    """Return the canonical shared directory for an embedding model.

    :param model_name: Stable single-directory model name.
    :returns: Model directory under the shared embedding cache.
    """
    return synth_setter_cache_dir() / "models" / "embeddings" / model_name
