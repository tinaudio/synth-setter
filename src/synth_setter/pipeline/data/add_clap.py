#!/usr/bin/env python
"""Add LAION-CLAP audio embeddings to existing HDF5 or webdataset shards.

CLAP expects mono audio at 48 kHz; each shard's stored audio is downmixed and
resampled before encoding. The 512-d embedding is written back into the same
shard — as a ``clap`` dataset for HDF5, or a ``<key>.clap.npy`` member for
webdataset tars (rewritten in place). Both paths are idempotent: a shard that
already carries the embedding is skipped, so a re-run only fills the gaps.

The CLAP model is imported lazily (see :class:`LaionClapEmbedder`) so importing
this module — and running its tests against an injected fake — needs neither the
``laion_clap`` package nor a checkpoint download.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Protocol, cast

import click
import h5py
import librosa
import numpy as np
import structlog

from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata

CLAP_FIELD = "clap"
CLAP_SAMPLE_RATE = 48_000
CLAP_EMBED_DIM = 512
_AUDIO_MEMBER_SUFFIX = ".audio.npy"
_CLAP_MEMBER_SUFFIX = ".clap.npy"
_METADATA_MEMBER = "metadata.json"

log = structlog.get_logger(__name__)


class ClapEmbedder(Protocol):
    """Encodes a batch of mono 48 kHz waveforms into CLAP embeddings."""

    def embed(self, audio_48k_mono: np.ndarray) -> np.ndarray:
        """Encode mono 48 kHz waveforms into CLAP embeddings.

        :param audio_48k_mono: ``(B, T)`` mono audio at 48 kHz.
        :returns: ``(B, CLAP_EMBED_DIM)`` float embeddings.
        """
        ...


def to_clap_input(audio: np.ndarray, source_sr: int) -> np.ndarray:
    """Downmix to mono and resample to CLAP's 48 kHz rate.

    :param audio: Shard audio of shape ``(B, C, T)`` at ``source_sr``.
    :param source_sr: Sample rate of ``audio`` in Hz.
    :returns: ``(B, T')`` float32 mono audio at :data:`CLAP_SAMPLE_RATE`.
    """
    mono = audio.astype(np.float32).mean(axis=1)
    if source_sr != CLAP_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=CLAP_SAMPLE_RATE, axis=-1)
    return np.ascontiguousarray(mono, dtype=np.float32)


def embed_audio_batch(audio: np.ndarray, source_sr: int, embedder: ClapEmbedder) -> np.ndarray:
    """Encode one batch of shard audio into CLAP embeddings.

    :param audio: Shard audio of shape ``(B, C, T)`` at ``source_sr``.
    :param source_sr: Sample rate of ``audio`` in Hz.
    :param embedder: CLAP encoder applied to the mono 48 kHz batch.
    :returns: ``(B, CLAP_EMBED_DIM)`` float32 embeddings.
    :raises ValueError: If the embedder returns an unexpected shape.
    """
    clap_input = to_clap_input(audio, source_sr)
    embeddings = np.asarray(embedder.embed(clap_input))
    expected = (clap_input.shape[0], CLAP_EMBED_DIM)
    if embeddings.shape != expected:
        raise ValueError(
            f"embedder returned {embeddings.shape}, expected {expected} (dim {CLAP_EMBED_DIM})"
        )
    return embeddings.astype(np.float32)


def add_clap_to_h5(path: Path, embedder: ClapEmbedder, *, batch_size: int) -> int:
    """Add a ``clap`` dataset to an HDF5 shard, reading the rate from ``audio.attrs``.

    :param path: HDF5 shard opened read/write.
    :param embedder: CLAP encoder.
    :param batch_size: Rows encoded per embedder call.
    :returns: Rows embedded, or 0 if the shard already carries the field.
    :raises ValueError: If the audio dataset has no ``sample_rate`` attr.
    """
    with h5py.File(path, "r+") as f:
        if CLAP_FIELD in f:
            log.info("clap.h5.skip", shard=path.name, reason="field already present")
            return 0
        audio = cast(h5py.Dataset, f["audio"])
        num_rows = audio.shape[0]
        if "sample_rate" not in audio.attrs:
            raise ValueError(f"{path.name}: audio dataset is missing the 'sample_rate' attr")
        source_sr = int(cast(int, audio.attrs["sample_rate"]))
        clap = f.create_dataset(CLAP_FIELD, shape=(num_rows, CLAP_EMBED_DIM), dtype=np.float32)
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            clap[start:end] = embed_audio_batch(audio[start:end], source_sr, embedder)
    log.info("clap.h5.done", shard=path.name, rows=num_rows)
    return num_rows


def add_clap_to_wds(path: Path, embedder: ClapEmbedder, *, batch_size: int) -> int:
    """Rewrite a webdataset tar in place, adding a ``<key>.clap.npy`` per audio member.

    :param path: Webdataset tar shard.
    :param embedder: CLAP encoder.
    :param batch_size: Rows encoded per embedder call.
    :returns: Rows embedded, or 0 if the shard already carries CLAP members.
    """
    if _wds_has_clap(path):
        log.info("clap.wds.skip", shard=path.name, reason="clap members already present")
        return 0
    source_sr = _read_wds_sample_rate(path)

    rows = 0
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tarfile.open(path, "r") as src, tarfile.open(tmp_path, "w") as dst:
            for member in src.getmembers():
                extracted = src.extractfile(member) if member.isfile() else None
                payload = extracted.read() if extracted is not None else b""
                dst.addfile(member, io.BytesIO(payload))
                if not member.name.endswith(_AUDIO_MEMBER_SUFFIX):
                    continue
                key = member.name[: -len(_AUDIO_MEMBER_SUFFIX)]
                audio = np.load(io.BytesIO(payload))
                embeddings = _embed_in_batches(audio, source_sr, embedder, batch_size)
                _add_npy_member(dst, f"{key}{_CLAP_MEMBER_SUFFIX}", embeddings)
                rows += audio.shape[0]
        os.replace(tmp_path, path)
    finally:
        # On success os.replace consumed tmp_path; on failure remove the partial rewrite.
        tmp_path.unlink(missing_ok=True)
    log.info("clap.wds.done", shard=path.name, rows=rows)
    return rows


def add_clap_to_shard(path: Path, embedder: ClapEmbedder, *, batch_size: int) -> int:
    """Dispatch a shard to the HDF5 or webdataset path by file suffix.

    :param path: Shard path ending in ``.h5`` or ``.tar``.
    :param embedder: CLAP encoder.
    :param batch_size: Rows encoded per embedder call.
    :returns: Rows embedded.
    :raises ValueError: If the suffix is neither ``.h5`` nor ``.tar``.
    """
    if path.suffix == ".h5":
        return add_clap_to_h5(path, embedder, batch_size=batch_size)
    if path.suffix == ".tar":
        return add_clap_to_wds(path, embedder, batch_size=batch_size)
    raise ValueError(f"unsupported shard type {path.suffix!r}: expected .h5 or .tar")


def _embed_in_batches(
    audio: np.ndarray, source_sr: int, embedder: ClapEmbedder, batch_size: int
) -> np.ndarray:
    """Encode ``(N, C, T)`` audio in ``batch_size`` chunks.

    :param audio: ``(N, C, T)`` shard audio at ``source_sr``.
    :param source_sr: Sample rate of ``audio`` in Hz.
    :param embedder: CLAP encoder.
    :param batch_size: Rows per embedder call.
    :returns: ``(N, CLAP_EMBED_DIM)`` float32 embeddings.
    """
    batches = [
        embed_audio_batch(audio[start : start + batch_size], source_sr, embedder)
        for start in range(0, audio.shape[0], batch_size)
    ]
    return np.concatenate(batches, axis=0)


def _read_wds_sample_rate(path: Path) -> int:
    """Read the shard sample rate from the tar's ``metadata.json`` member.

    :param path: Webdataset tar shard.
    :returns: Sample rate in Hz.
    :raises ValueError: If the tar has no ``metadata.json`` member.
    """
    with tarfile.open(path, "r") as tar:
        member = next(
            (m for m in tar.getmembers() if m.isfile() and m.name == _METADATA_MEMBER), None
        )
        extracted = tar.extractfile(member) if member is not None else None
        if extracted is None:
            raise ValueError(f"{path.name}: missing {_METADATA_MEMBER}")
        return ShardMetadata.model_validate_json(extracted.read()).sample_rate


def _wds_has_clap(path: Path) -> bool:
    """Whether the tar already contains any ``<key>.clap.npy`` member.

    :param path: Webdataset tar shard.
    :returns: True if any CLAP member is present.
    """
    with tarfile.open(path, "r") as tar:
        return any(name.endswith(_CLAP_MEMBER_SUFFIX) for name in tar.getnames())


def _add_npy_member(tar: tarfile.TarFile, name: str, array: np.ndarray) -> None:
    """Append ``array`` to ``tar`` as a ``.npy`` member.

    :param tar: Open tar archive in write mode.
    :param name: Member name (e.g. ``00000000.clap.npy``).
    :param array: Array serialized via ``numpy.save``.
    """
    buf = io.BytesIO()
    np.save(buf, array)
    info = tarfile.TarInfo(name=name)
    info.size = buf.tell()
    buf.seek(0)
    tar.addfile(info, buf)


def _build_embedder(ckpt: str | None) -> ClapEmbedder:
    """Construct the real CLAP encoder; indirected so tests can inject a fake.

    :param ckpt: CLAP checkpoint path, or None for the default general audioset model.
    :returns: A ready-to-use CLAP encoder.
    """
    return LaionClapEmbedder(ckpt)


class LaionClapEmbedder:
    """:class:`ClapEmbedder` backed by ``laion_clap.CLAP_Module`` (general audioset)."""

    def __init__(self, ckpt: str | None = None, *, enable_fusion: bool = False) -> None:
        """Load the CLAP checkpoint (the default general audioset model when ``ckpt`` is None).

        :param ckpt: Path to a CLAP checkpoint, or None for the bundled default.
        :param enable_fusion: Whether to enable CLAP's feature-fusion variant.
        """
        import laion_clap  # noqa: PLC0415 — heavy optional dependency, imported on demand

        self._model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        self._model.load_ckpt(ckpt)

    def embed(self, audio_48k_mono: np.ndarray) -> np.ndarray:
        """Encode a mono 48 kHz batch into CLAP embeddings.

        :param audio_48k_mono: ``(B, T)`` mono audio at 48 kHz.
        :returns: ``(B, CLAP_EMBED_DIM)`` float embeddings.
        """
        return self._model.get_audio_embedding_from_data(x=audio_48k_mono, use_tensor=False)


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--batch-size", "-b", type=int, default=256, help="Rows encoded per CLAP call.")
@click.option(
    "--ckpt", type=str, default=None, help="CLAP checkpoint path (default: general audioset)."
)
def main(data_dir: Path, batch_size: int, ckpt: str | None) -> None:
    """Add CLAP embeddings to every ``shard-*.h5`` / ``shard-*.tar`` under DATA_DIR.

    :param data_dir: Directory of dataset shards.
    :param batch_size: Rows encoded per CLAP call.
    :param ckpt: CLAP checkpoint path, or None for the default general audioset model.
    :raises click.ClickException: If no matching shards are found.
    """
    shards = sorted(p for p in data_dir.glob("shard-*") if p.suffix in {".h5", ".tar"})
    if not shards:
        raise click.ClickException(f"no shard-*.h5 or shard-*.tar found under {data_dir}")

    embedder = _build_embedder(ckpt)
    for shard in shards:
        add_clap_to_shard(shard, embedder, batch_size=batch_size)


if __name__ == "__main__":
    main()
