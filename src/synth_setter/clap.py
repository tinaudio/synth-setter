"""Shared CLAP checkpoint identity and local materialization.

Typical usage: ``checkpoint = resolve_clap_checkpoint(DEFAULT_CLAP_CHECKPOINT)`` before
constructing a Transformers CLAP model.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from synth_setter.model_cache import embedding_model_dir, synth_setter_cache_dir
from synth_setter.pipeline import r2_io

CLAP_SAMPLE_RATE: int = 48_000
DEFAULT_CLAP_CHECKPOINT: str = "r2://intermediate-data/models/encoders/clap-htsat-unfused"
DEFAULT_CLAP_CHECKPOINT_SHA256: str = (
    "ca1bad56747e413b34ac0b722a9d5adc5e479d64321440da1227f716a7a44ada"
)
_REQUIRED_CHECKPOINT_FILES: tuple[str, ...] = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
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
    :returns: SHA-256 identity for the complete checkpoint tree.
    """
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and not path.relative_to(checkpoint_dir).parts[:2] == (".cache", "huggingface")
    )
    for path in files:
        relative_path = path.relative_to(checkpoint_dir).as_posix().encode()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_clap_checkpoint(checkpoint: str) -> str:
    """Resolve a local, R2, or Hugging Face CLAP checkpoint directory.

    :param checkpoint: Local directory, R2 prefix, or Hugging Face model id.
    :returns: Local directory accepted by the Transformers loaders.
    :raises RuntimeError: The downloaded R2 checkpoint is incomplete.
    """
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        return str(local)
    if r2_io.is_r2_uri(checkpoint):
        if checkpoint == DEFAULT_CLAP_CHECKPOINT:
            cache_dir = embedding_model_dir("clap-htsat-unfused")
        else:
            cache_key = checkpoint.removeprefix("r2://").strip("/")
            cache_dir = synth_setter_cache_dir() / "models" / cache_key
        if _is_complete_checkpoint(cache_dir):
            return str(cache_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        r2_io.ensure_r2_env_loaded()
        r2_io.download_dir_no_overwrite(checkpoint, cache_dir)
        if not _is_complete_checkpoint(cache_dir):
            raise RuntimeError(f"downloaded CLAP checkpoint is incomplete: {cache_dir}")
        return str(cache_dir)

    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)
