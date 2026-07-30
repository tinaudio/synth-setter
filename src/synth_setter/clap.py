"""Shared CLAP checkpoint identity and local materialization."""

from __future__ import annotations

import fcntl
import hashlib
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from synth_setter.model_cache import embedding_model_dir, synth_setter_cache_dir
from synth_setter.pipeline import r2_io

DEFAULT_CLAP_CHECKPOINT: str = "laion/clap-htsat-unfused"
DEFAULT_CLAP_TRAINING_CHECKPOINT: str = "r2://intermediate-data/models/encoders/clap-htsat-unfused"
DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256: str = (
    "ca1bad56747e413b34ac0b722a9d5adc5e479d64321440da1227f716a7a44ada"
)
_REQUIRED_CHECKPOINT_FILES: tuple[str, ...] = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


def _is_complete_checkpoint(checkpoint_dir: Path) -> bool:
    """Return whether every required CLAP file contains data.

    :param checkpoint_dir: Directory whose required CLAP files are checked.
    :returns: True when every required file is non-empty.
    """
    return all(
        (checkpoint_dir / filename).is_file() and (checkpoint_dir / filename).stat().st_size > 0
        for filename in _REQUIRED_CHECKPOINT_FILES
    )


def clap_checkpoint_sha256(checkpoint_dir: Path) -> str:
    """Hash checkpoint paths and contents with stable framing.

    :param checkpoint_dir: Directory hashed with stable path framing.
    :returns: SHA-256 identity for a non-empty checkpoint tree.
    :raises ValueError: The checkpoint directory is absent or contains no files.
    """
    files = sorted(
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and not path.relative_to(checkpoint_dir).parts[:2] == (".cache", "huggingface")
    )
    if not files:
        raise ValueError(f"CLAP checkpoint {checkpoint_dir} has no files")

    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(checkpoint_dir).as_posix().encode()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _r2_cache_dir(checkpoint: str) -> Path:
    """Return the source-specific cache directory for an R2 checkpoint.

    :param checkpoint: R2 checkpoint URI.
    :returns: Cache path namespaced by the complete bucket and object prefix.
    """
    cache_key = checkpoint.removeprefix("r2://").strip("/")
    return synth_setter_cache_dir() / "models" / "r2" / cache_key


@contextmanager
def _process_lock(lock_path: Path) -> Iterator[None]:
    """Serialize checkpoint resolution across processes.

    :param lock_path: Sibling lock file shared by checkpoint resolvers.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _publish_checkpoint(staging_dir: Path, cache_dir: Path) -> None:
    """Atomically replace incomplete cache data with a validated staging tree.

    :param staging_dir: Complete sibling staging directory.
    :param cache_dir: Published checkpoint path.
    :raises BaseException: Publication fails after restoring prior cache data.
    """
    backup_dir = staging_dir.with_name(f"{staging_dir.name}.replaced")
    if cache_dir.exists():
        cache_dir.replace(backup_dir)
    try:
        staging_dir.replace(cache_dir)
    except BaseException:
        if backup_dir.exists() and not cache_dir.exists():
            backup_dir.replace(cache_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _hydrate_r2_checkpoint(checkpoint: str, cache_dir: Path) -> str:
    """Download, validate, and atomically publish an R2 checkpoint.

    :param checkpoint: R2 checkpoint URI.
    :param cache_dir: Source-specific publication path.
    :returns: Published checkpoint directory.
    :raises RuntimeError: The downloaded checkpoint is incomplete.
    """
    lock_path = cache_dir.with_name(f".{cache_dir.name}.lock")
    with _process_lock(lock_path):
        if _is_complete_checkpoint(cache_dir):
            return str(cache_dir)

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{cache_dir.name}.staging-", dir=cache_dir.parent)
        )
        try:
            r2_io.ensure_r2_env_loaded()
            r2_io.download_dir_no_overwrite(checkpoint, staging_dir)
            if not _is_complete_checkpoint(staging_dir):
                raise RuntimeError(f"downloaded CLAP checkpoint is incomplete: {staging_dir}")
            _publish_checkpoint(staging_dir, cache_dir)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
        return str(cache_dir)


def resolve_clap_checkpoint(checkpoint: str) -> str:
    """Resolve a local, R2, or Hugging Face CLAP checkpoint directory.

    :param checkpoint: Local directory, R2 prefix, or Hugging Face model id.
    :returns: Local directory accepted by the Transformers loaders.
    """
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        return str(local)
    if r2_io.is_r2_uri(checkpoint):
        return _hydrate_r2_checkpoint(checkpoint, _r2_cache_dir(checkpoint))

    from huggingface_hub import snapshot_download

    if checkpoint == DEFAULT_CLAP_CHECKPOINT:
        cache_dir = embedding_model_dir("clap-htsat-unfused")
        return snapshot_download(checkpoint, local_dir=str(cache_dir))
    return snapshot_download(checkpoint)
