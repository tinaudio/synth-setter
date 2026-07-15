#!/usr/bin/env python
"""Append a ``music2latent`` latent column to finalized Lance dataset shards.

Backfills one column onto every ``shard-*.lance`` dataset under a directory, in a
single Lance ``add_columns`` transaction per shard:

- ``music2latent`` — a ``fixed_shape_tensor<float32, (C*D, T)>`` music2latent
  latent derived from each row's ``audio`` tensor. A sequence latent (not a
  search vector), so it is kept retrievable but un-indexed; this requires a
  constant ``(channels, latent-dim, time)`` across a shard's rows.

The encoder is an injected callable, so the core runs without a checkpoint;
:func:`load_m2l_audio_encoder` builds the real encoder behind a lazy
``music2latent`` import.

This is a sanctioned post-finalize augmenter: it commits a new Lance version of
each existing shard rather than writing fresh ``data/`` shards, so it does not
cross the worker/finalize write boundary.

CLI: ``python -m synth_setter.pipeline.data.add_music2latent DATA_DIR``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import click
import lance
import numpy as np
import pyarrow as pa
import structlog
from einops import rearrange

from synth_setter.data.vst.shapes import AUDIO_FIELD
from synth_setter.pipeline.data.lance_shard import tensor_array

logger = structlog.get_logger(__name__)

MUSIC2LATENT_FIELD: str = "music2latent"
# Per-forward GPU batch cap, independent of Lance's read batch size — caps GPU
# memory per encode call.
M2L_ENCODE_MAX_BATCH: int = 64

M2LEncodeFn: TypeAlias = Callable[[np.ndarray], np.ndarray]


def get_shard_id(shard_path: Path) -> int:
    """Parse the integer shard id from a ``shard-<id>.lance`` dataset directory name.

    :param shard_path: Path whose stem is ``shard-<id>`` (e.g. ``shard-000042``).
    :returns: The shard's integer id (leading zeros stripped).
    """
    return int(shard_path.stem.split("-")[1])


def music2latent_record_batch(audio: np.ndarray, m2l_encode: M2LEncodeFn) -> pa.RecordBatch:
    """Encode one audio batch into the single ``music2latent`` record batch.

    :param audio: ``(B, C, T)`` audio batch (any float dtype).
    :param m2l_encode: Maps the batch to a ``(B, C*D, T)`` latent batch.
    :returns: A ``B``-row batch with a ``music2latent`` fixed-shape-tensor column.
    :raises ValueError: The encoder returns a non-rank-3 latent (the fixed-shape
        tensor schema is permanent — a wrong-rank column fails only at model
        loading), the wrong row count, or a non-finite latent.
    """
    rows = len(audio)
    latents = np.ascontiguousarray(m2l_encode(audio), dtype=np.float32)
    if latents.ndim != 3:
        raise ValueError(
            f"encoder must return a rank-3 (B, C*D, T) latent batch, got shape {latents.shape}"
        )
    if len(latents) != rows:
        raise ValueError(f"encoder produced {len(latents)} latent rows, expected {rows}")
    if not np.isfinite(latents).all():
        raise ValueError(f"{MUSIC2LATENT_FIELD} latents contain non-finite values")
    return pa.record_batch(
        {MUSIC2LATENT_FIELD: tensor_array(latents, np.dtype("float32"), latents.shape[1:])}
    )


def add_music2latent(
    dataset: lance.LanceDataset,
    m2l_encode: M2LEncodeFn,
    *,
    batch_size: int | None = None,
) -> None:
    """Append a ``music2latent`` column to ``dataset``.

    Commits a new dataset version; existing columns are untouched and only
    ``audio`` is read per batch. ``add_columns`` commits in a single Lance
    transaction, so an interrupted run leaves the dataset on its prior version.

    :param dataset: Open Lance dataset carrying an ``audio`` column.
    :param m2l_encode: Maps an ``(B, C, T)`` batch to a ``(B, C*D, T)`` latent batch.
    :param batch_size: Rows per UDF call; ``None`` uses the Lance default. Ignored
        for legacy (v1) Lance datasets, which Lance rewrites whole.
    :raises ValueError: ``dataset`` already has a ``music2latent`` column
        (re-running on an augmented dataset would hit Lance's opaque "column
        already exists" error), or lacks the ``audio`` column the UDF reads.
    """
    if MUSIC2LATENT_FIELD in dataset.schema.names:
        raise ValueError(f"dataset already has a {MUSIC2LATENT_FIELD!r} column")
    if AUDIO_FIELD not in dataset.schema.names:
        raise ValueError(f"dataset has no {AUDIO_FIELD!r} column to encode")

    @lance.batch_udf()
    def udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        audio = batch.column(AUDIO_FIELD).to_numpy_ndarray()
        return music2latent_record_batch(audio, m2l_encode)

    dataset.add_columns(udf, read_columns=[AUDIO_FIELD], batch_size=batch_size)


def load_m2l_audio_encoder() -> M2LEncodeFn:
    """Build a music2latent encode callable over an ``(B, C, T)`` batch.

    The ``music2latent`` import is deferred so this module stays importable (and
    its core testable with a fake encoder) without loading a checkpoint. No device
    knob: music2latent owns its own placement. The returned callable holds a single
    encoder, so it is reused across shards without reloading the model.

    :returns: Encode callable mapping ``(B, C, T)`` to ``(B, C*D, T)`` float32.
    """
    from music2latent import EncoderDecoder

    encoder = EncoderDecoder()

    def encode(audio: np.ndarray) -> np.ndarray:
        batch, channels = audio.shape[0], audio.shape[1]
        flat = np.ascontiguousarray(rearrange(audio, "b c t -> (b c) t"), dtype=np.float32)
        latents = encoder.encode(flat, max_batch_size=M2L_ENCODE_MAX_BATCH)
        latents = rearrange(latents, "(b c) d t -> b (c d) t", b=batch, c=channels)
        return latents.cpu().numpy()

    return encode


def discover_shards(
    data_dir: Path,
    *,
    shard_range: tuple[int, int] | None = None,
    shard: int | None = None,
) -> list[Path]:
    """List ``shard-*.lance`` datasets under ``data_dir``, filtered and id-sorted.

    :param data_dir: Directory holding ``shard-<id>.lance`` dataset directories.
    :param shard_range: Half-open ``[lo, hi)`` id range to keep; ``None`` keeps all.
    :param shard: Single shard id to keep; ``None`` keeps all.
    :returns: Matching shard paths in ascending id order.
    :raises ValueError: Both ``shard_range`` and ``shard`` are given.
    """
    if shard_range is not None and shard is not None:
        raise ValueError("cannot specify both shard_range and shard")
    shards = list(data_dir.glob("shard-*.lance"))
    if shard_range is not None:
        shards = [s for s in shards if get_shard_id(s) in range(*shard_range)]
    if shard is not None:
        shards = [s for s in shards if get_shard_id(s) == shard]
    shards.sort(key=get_shard_id)
    return shards


@click.command()
@click.argument("data_dir", type=str)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Rows per UDF call; defaults to the Lance default (ignored for v1 datasets).",
)
@click.option(
    "--shard-range",
    type=int,
    nargs=2,
    default=None,
    help="Half-open [lo, hi) shard-id range to process.",
)
@click.option("--shard", type=int, default=None, help="Single shard id to process.")
def main(
    data_dir: str,
    batch_size: int | None,
    shard_range: tuple[int, int] | None,
    shard: int | None,
) -> None:
    """Add a ``music2latent`` column to every ``shard-*.lance`` dataset under DATA_DIR.

    DATA_DIR is a local directory of shards from generate_dataset or
    finalize_dataset. Shards that already carry a ``music2latent`` column are
    skipped, so a re-run only fills the gaps.

    :param data_dir: Local directory holding ``shard-<id>.lance`` dataset directories.
    :param batch_size: Rows per UDF call; ``None`` uses the Lance default.
    :param shard_range: Half-open ``[lo, hi)`` shard-id range to process.
    :param shard: Single shard id to process (mutually exclusive with ``shard_range``).
    """
    try:
        shards = discover_shards(Path(data_dir), shard_range=shard_range, shard=shard)
    except ValueError as exc:
        logger.error("shard_selection_invalid", data_dir=data_dir, error=str(exc))
        sys.exit(1)
    if not shards:
        logger.warning("no_shards_found", data_dir=data_dir)
        return

    try:
        m2l_encode = load_m2l_audio_encoder()
    except (OSError, RuntimeError, ImportError) as exc:
        logger.error("encoder_load_failed", error=str(exc))
        sys.exit(1)

    failures = 0
    for shard_path in shards:
        try:
            dataset = lance.dataset(str(shard_path))
            if MUSIC2LATENT_FIELD in dataset.schema.names:
                logger.info("music2latent_present_skipping", shard=shard_path.name)
                continue
            logger.info("adding_music2latent", shard=shard_path.name, rows=dataset.count_rows())
            add_music2latent(dataset, m2l_encode, batch_size=batch_size)
        # One bad shard (unreadable dataset, validation, Lance, or CUDA failure)
        # must not abort the remaining shards; log it, count it, and keep going.
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("add_music2latent_failed", shard=shard_path.name, error=str(exc))
            failures += 1
            continue
        logger.info("added_music2latent", shard=shard_path.name)

    if failures:
        logger.error("add_music2latent_incomplete", failed=failures, total=len(shards))
        sys.exit(1)


if __name__ == "__main__":
    main()
