"""Resolve shared local cache paths for model artifacts and identify their contents."""

from __future__ import annotations

import hashlib
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


def checkpoint_tree_sha256(checkpoint_dir: Path) -> str:
    """Hash checkpoint file paths and contents in deterministic order.

    :param checkpoint_dir: Materialized checkpoint directory.
    :returns: SHA-256 identity for the complete checkpoint tree.
    :raises ValueError: The checkpoint directory contains no files.
    """
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and not path.relative_to(checkpoint_dir).parts[:2] == (".cache", "huggingface")
    )
    if not files:
        raise ValueError(f"checkpoint directory has no files: {checkpoint_dir}")
    for path in files:
        relative_path = path.relative_to(checkpoint_dir).as_posix().encode()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
