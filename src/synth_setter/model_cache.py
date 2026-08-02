"""Resolve shared local cache paths for model artifacts and identify their contents."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
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


def checkpoint_files_sha256(checkpoint_dir: Path, files: Iterable[Path]) -> str:
    """Hash selected checkpoint paths and contents in deterministic order.

    :param checkpoint_dir: Root used to frame snapshot-relative paths.
    :param files: Materialized files owned by one checkpoint identity.
    :returns: SHA-256 identity for the selected files.
    :raises ValueError: No files are selected or a selected path is invalid.
    """
    selected = sorted(files)
    if not selected:
        raise ValueError(f"checkpoint selection has no files: {checkpoint_dir}")
    digest = hashlib.sha256()
    for path in selected:
        try:
            relative_path = path.relative_to(checkpoint_dir).as_posix().encode()
        except ValueError as exc:
            raise ValueError(f"checkpoint file lies outside {checkpoint_dir}: {path}") from exc
        if not path.is_file():
            raise ValueError(f"checkpoint file is absent: {path}")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_tree_sha256(checkpoint_dir: Path) -> str:
    """Hash checkpoint file paths and contents in deterministic order.

    :param checkpoint_dir: Materialized checkpoint directory.
    :returns: SHA-256 identity for the complete checkpoint tree.
    """
    files = (
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and not path.relative_to(checkpoint_dir).parts[:2] == (".cache", "huggingface")
    )
    return checkpoint_files_sha256(checkpoint_dir, files)
