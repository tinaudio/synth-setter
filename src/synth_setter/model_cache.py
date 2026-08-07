"""Resolve shared local cache paths for model artifacts and identify their contents."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from filelock import FileLock
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from synth_setter.pipeline import r2_io


def retry_external_io[**P, R](
    *, retry_exceptions: tuple[type[BaseException], ...]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Build the shared bounded retry policy for transient external I/O.

    :param retry_exceptions: Transient exception types safe to retry.
    :returns: Decorator applying three attempts with bounded exponential backoff.
    """

    def decorate(operation: Callable[P, R]) -> Callable[P, R]:
        return retry(
            reraise=True,
            retry=retry_if_exception_type(retry_exceptions),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
        )(operation)

    return decorate


def synth_setter_cache_dir() -> Path:
    """Return the XDG-aware synth-setter cache root.

    :returns: ``synth-setter`` under ``XDG_CACHE_HOME`` or the conventional
        ``~/.cache`` fallback.
    """
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "synth-setter"


def file_sha256(path: Path) -> str:
    """Hash one artifact without loading it into memory.

    :param path: File whose content identity is required.
    :returns: Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cache_request(r2_uri: str, namespace: str, expected_sha256: str) -> None:
    """Reject malformed cache identities before filesystem access.

    :param r2_uri: Source object URI.
    :param namespace: Stable cache namespace without path separators.
    :param expected_sha256: Required lowercase SHA-256 content identity.
    :raises ValueError: An argument is malformed.
    """
    if namespace in {"", ".", ".."} or Path(namespace).name != namespace:
        raise ValueError(f"cache namespace must be one path component, got {namespace!r}")
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if not r2_io.is_r2_uri(r2_uri):
        raise ValueError(f"not an r2:// URI: {r2_uri!r}")


def _cache_destination(r2_uri: str, namespace: str, expected_sha256: str) -> Path:
    """Derive a collision-resistant local path for one R2 object.

    :param r2_uri: Validated source object URI.
    :param namespace: Validated cache namespace.
    :param expected_sha256: Required content identity.
    :returns: Destination retaining the source basename.
    """
    source_key = hashlib.sha256(r2_uri.encode()).hexdigest()[:16]
    return (
        synth_setter_cache_dir()
        / "models"
        / "artifacts"
        / namespace
        / source_key
        / expected_sha256
        / Path(r2_uri).name
    )


@retry_external_io(retry_exceptions=(subprocess.CalledProcessError,))
def _download_verified(r2_uri: str, destination: Path, expected_sha256: str) -> None:
    """Publish downloaded bytes only after digest verification.

    :param r2_uri: Source object URI.
    :param destination: Final local cache path.
    :param expected_sha256: Required content identity.
    :raises ValueError: Downloaded bytes have the wrong digest.
    """
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        r2_io.download_to_path(r2_uri, temporary_path)
        actual_sha256 = file_sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"downloaded artifact SHA-256 is {actual_sha256}, "
                f"expected {expected_sha256}: {r2_uri}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def cache_r2_file(r2_uri: str, namespace: str, expected_sha256: str) -> Path:
    """Download and atomically cache one digest-pinned R2 object.

    :param r2_uri: Source object URI.
    :param namespace: Stable cache namespace without path separators.
    :param expected_sha256: Required lowercase SHA-256 content identity.
    :returns: Digest-verified local artifact path.
    """
    _validate_cache_request(r2_uri, namespace, expected_sha256)
    destination = _cache_destination(r2_uri, namespace, expected_sha256)
    lock_path = destination.with_name(f".{destination.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        if destination.is_file() and file_sha256(destination) == expected_sha256:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        _download_verified(r2_uri, destination, expected_sha256)
        return destination


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
