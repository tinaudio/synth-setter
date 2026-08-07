"""Resolve shared local cache paths for model artifacts and identify their contents."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from filelock import FileLock

from synth_setter.pipeline import r2_io


def synth_setter_cache_dir() -> Path:
    """Return the XDG-aware synth-setter cache root.

    :returns: ``synth-setter`` under ``XDG_CACHE_HOME`` or the conventional
        ``~/.cache`` fallback.
    """
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "synth-setter"


def _file_sha256(path: Path) -> str:
    """Hash one artifact without loading it into memory.

    :param path: File whose content identity is required.
    :returns: Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_r2_file(r2_uri: str, namespace: str, expected_sha256: str) -> Path:
    """Download and atomically cache one digest-pinned R2 object.

    :param r2_uri: Source object URI.
    :param namespace: Stable cache namespace without path separators.
    :param expected_sha256: Required lowercase SHA-256 content identity.
    :returns: Digest-verified local artifact path.
    :raises ValueError: An argument is malformed or downloaded bytes have the wrong digest.
    """
    if namespace in {"", ".", ".."} or Path(namespace).name != namespace:
        raise ValueError(f"cache namespace must be one path component, got {namespace!r}")
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if not r2_io.is_r2_uri(r2_uri):
        raise ValueError(f"not an r2:// URI: {r2_uri!r}")

    source_key = hashlib.sha256(r2_uri.encode()).hexdigest()[:16]
    destination = (
        synth_setter_cache_dir()
        / "models"
        / "artifacts"
        / namespace
        / source_key
        / Path(r2_uri).name
    )
    lock_path = destination.with_name(f".{destination.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        if destination.is_file() and _file_sha256(destination) == expected_sha256:
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.partial-", dir=destination.parent
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            r2_io.download_to_path(r2_uri, temporary_path)
            actual_sha256 = _file_sha256(temporary_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"downloaded artifact SHA-256 is {actual_sha256}, "
                    f"expected {expected_sha256}: {r2_uri}"
                )
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination


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
