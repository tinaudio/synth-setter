#!/usr/bin/env python
"""Append ``m2l`` + CLAP audio-embedding columns to a finalized Lance dataset.

The functional core (:func:`embeddings_record_batch`, :func:`add_embeddings`)
maps a Lance dataset's ``audio`` column to two columns:

- ``clap`` — a ``FixedSizeList<float32, CLAP_EMBEDDING_DIM>`` LAION-CLAP audio
  embedding. Stored as a fixed-size list (not opaque bytes) so Lance can build a
  vector (IVF_PQ) index over it and serve ``nearest=`` queries.
- ``m2l`` — a ``fixed_shape_tensor<float32, (C*D, T)>`` music2latent latent. A
  sequence latent, not a search vector, so it is kept retrievable but
  un-indexed; this requires a constant ``(channels, latent-dim, time)`` across
  the dataset.

A separate SAME mode (``--same s`` / ``--same l``) appends ``same_s`` /
``same_l`` — ``fixed_shape_tensor<float32, (256, T)>`` Stability AI SAME latent
sequences, un-indexed like ``m2l`` — so an already-augmented dataset can be
extended without touching its m2l/clap columns.

All embedders are injected callables, so the core runs without a checkpoint;
:func:`load_m2l_audio_encoder` / :func:`load_clap_audio_encoder` /
:func:`load_same_audio_encoder` build the real encoders behind lazy
``music2latent`` / ``transformers`` / ``stable_audio_tools`` imports
(``stable_audio_tools`` ships in the optional ``same`` extra — install with
``uv sync --extra same``; its stale upstream pins are relaxed via the
``[[tool.uv.dependency-metadata]]`` block in pyproject.toml).

This is a sanctioned post-finalize augmenter: it commits a new Lance version of
an existing dataset rather than writing fresh ``data/`` shards, so it does not
cross the worker/finalize write boundary.

CLI: ``python -m synth_setter.pipeline.data.add_embeddings DATASET.lance``.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import click
import numpy as np
import pyarrow as pa
import structlog
from einops import rearrange

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    M2L_FIELD,
    SAME_L_FIELD,
    SAME_S_FIELD,
)
from synth_setter.pipeline import r2_io

# lance (and lance_shard, which imports it) is imported lazily everywhere below:
# native Lance logging locks in LANCE_LOG at first import, so _configure_lance_logging
# must run first. Hoisting any of these imports silently breaks --debug telemetry.
if TYPE_CHECKING:
    import lance

logger = structlog.get_logger(__name__)

DEFAULT_CLAP_CHECKPOINT: str = "laion/clap-htsat-unfused"
# CLAP's feature extractor rejects any other input rate.
CLAP_SAMPLE_RATE: int = 48000
# Projected audio-embedding width of the default checkpoint; the ``clap`` column's
# fixed-size-list length, and the dim a vector query must match.
CLAP_EMBEDDING_DIM: int = 512
# Per-forward sub-batch caps for the encoders, independent of Lance's read batch:
# the batch_udf hands an encoder the whole read batch at once, so without these a
# large dataset OOMs the GPU (a 256-row CLAP forward alone needs ~5 GiB).
M2L_ENCODE_MAX_BATCH: int = 64
CLAP_ENCODE_MAX_BATCH: int = 32
# Bounds each Lance read/UDF call while amortizing remote-I/O and Python overhead.
DEFAULT_LANCE_BATCH_SIZE: int = 128
MAX_PROGRESS_LOGS: int = 20
# IVF_PQ needs ~256 training vectors (256 PQ centroids); below this the index is
# skipped and callers fall back to Lance's exact (brute-force) ``nearest`` scan.
MIN_ROWS_FOR_INDEX: int = 256
# 512 % 16 == 0; each PQ sub-quantizer covers a 32-d slice of the CLAP vector.
DEFAULT_NUM_SUB_VECTORS: int = 16
# Cosine matches CLAP's L2-normalized audio embeddings.
DEFAULT_INDEX_METRIC: str = "cosine"
DEFAULT_LANCE_LOG: str = "warn"
# Checked at batch boundaries only: forces a progress line on the next batch when
# batches are slow, so early progress isn't silent. Deliberately not a stall
# detector — a hard stall shows as log silence, plus LANCE_LOG telemetry under --debug.
PROGRESS_LOG_INTERVAL_SECONDS: float = 30.0

# SAME (Stability AI Semantic-Acoustic Music Encoder) contract, per the
# checkpoints' model_config.json: stereo 44.1 kHz input, 256-dim latents, one
# latent frame per 4096 input samples.
SAME_EMBEDDING_DIM: int = 256
SAME_SAMPLE_RATE: int = 44100
SAME_DOWNSAMPLING_RATIO: int = 4096
# The encoder zero-pads input to a multiple of two latent hops before striding
# (verified empirically on SAME-S across lengths 4096..180224), so frame counts
# come in even blocks of this many samples.
SAME_PAD_BLOCK_SAMPLES: int = 2 * SAME_DOWNSAMPLING_RATIO
# Latent frames for the project's standard 4 s render:
# ceil(176400 / 8192) * 2 == 44.
SAME_LATENT_FRAMES: int = 44
# Per-forward sub-batch cap: SAME-L is a ~3.4 GB model, so the whole Lance read
# batch cannot hit the GPU at once.
SAME_ENCODE_MAX_BATCH: int = 16
# R2 mirrors of the public stabilityai/SAME-S and stabilityai/SAME-L HF repos.
DEFAULT_SAME_S_CHECKPOINT: str = "r2://intermediate-data/models/same-s"
DEFAULT_SAME_L_CHECKPOINT: str = "r2://intermediate-data/models/same-l"

M2LEncodeFn: TypeAlias = Callable[[np.ndarray], np.ndarray]
ClapEncodeFn: TypeAlias = Callable[[np.ndarray, int], np.ndarray]
# Maps prepared stereo 44.1 kHz ``(B, 2, T)`` audio to ``(B, 256, T_lat)`` latents.
SameEncodeFn: TypeAlias = Callable[[np.ndarray], np.ndarray]


def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Average an ``(B, C, T)`` batch's channels into ``(B, T)`` mono float32.

    :param audio: ``(B, C, T)`` batch, ``C >= 1`` (on-disk float16). ``float32``
        upcasts in the reduction so the encoder never sees half-precision.
    :returns: ``(B, T)`` float32 mono batch.
    """
    return audio.mean(axis=1, dtype=np.float32)


def _clap_fixed_size_list(clap: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    """Pack a ``(B, dim)`` float32 batch as a ``FixedSizeList<float32, dim>`` array.

    :param clap: ``(B, dim)`` embedding batch.
    :param dim: Fixed list length (the vector dimensionality).
    :returns: A Lance-indexable fixed-size-list array.
    """
    flat = pa.array(np.ascontiguousarray(clap, dtype=np.float32).reshape(-1), pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def embeddings_record_batch(
    audio: np.ndarray,
    m2l_encode: M2LEncodeFn,
    clap_encode: ClapEncodeFn,
    sample_rate: int,
    *,
    clap_dim: int = CLAP_EMBEDDING_DIM,
) -> pa.RecordBatch:
    """Encode one audio batch into the ``(m2l, clap)`` record batch.

    :param audio: ``(B, C, T)`` audio batch (any float dtype).
    :param m2l_encode: Maps the batch to a ``(B, C*D, T)`` latent batch.
    :param clap_encode: Maps a mono ``(B, T)`` batch and ``sample_rate`` to ``(B, clap_dim)``.
    :param sample_rate: Passed through to ``clap_encode``.
    :param clap_dim: Expected CLAP width; the ``clap`` fixed-size-list length.
    :returns: A ``B``-row batch with a ``m2l`` fixed-shape-tensor column and a
        ``clap`` ``FixedSizeList<float32, clap_dim>`` column.
    :raises ValueError: An encoder returns the wrong row count, ``clap`` is not
        ``(B, clap_dim)``, or either embedding is non-finite (the columns are
        permanent — a NaN/inf row from a degenerate clip must not be written).
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    rows = len(audio)
    m2l = np.ascontiguousarray(m2l_encode(audio), dtype=np.float32)
    clap = np.ascontiguousarray(
        clap_encode(_downmix_to_mono(audio), sample_rate), dtype=np.float32
    )
    if len(m2l) != rows or len(clap) != rows:
        raise ValueError(
            f"encoders produced {len(m2l)} m2l and {len(clap)} clap rows, expected {rows}"
        )
    if clap.ndim != 2 or clap.shape[1] != clap_dim:
        raise ValueError(
            f"clap encoder produced shape {clap.shape}, expected ({rows}, {clap_dim})"
        )
    for field, embedding in ((M2L_FIELD, m2l), (CLAP_FIELD, clap)):
        if not np.isfinite(embedding).all():
            raise ValueError(f"{field} embeddings contain non-finite values")
    return pa.record_batch(
        {
            M2L_FIELD: tensor_array(m2l, np.dtype("float32"), m2l.shape[1:]),
            CLAP_FIELD: _clap_fixed_size_list(clap, clap_dim),
        }
    )


def build_clap_index(
    dataset: lance.LanceDataset,
    *,
    num_partitions: int | None = None,
    num_sub_vectors: int = DEFAULT_NUM_SUB_VECTORS,
    metric: str = DEFAULT_INDEX_METRIC,
) -> bool:
    """Build an IVF_PQ vector index on the ``clap`` column, if the dataset is large enough.

    :param dataset: Dataset already carrying a ``clap`` fixed-size-list column.
    :param num_partitions: IVF partition count; ``None`` uses ``round(sqrt(rows))``.
    :param num_sub_vectors: PQ sub-vector count (must divide the CLAP dim).
    :param metric: Distance metric for the index.
    :returns: ``True`` if an index was built, ``False`` if skipped (too few rows
        to train PQ — exact ``nearest`` still works without an index).
    :raises ValueError: ``num_sub_vectors`` or ``num_partitions`` is non-positive,
        or ``num_sub_vectors`` does not divide the ``clap`` column's vector width
        (Lance's own failures here are a ``ZeroDivisionError`` / opaque PQ error).
    """
    if num_sub_vectors < 1:
        raise ValueError(f"num_sub_vectors must be >= 1, got {num_sub_vectors}")
    if num_partitions is not None and num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")
    clap_dim = dataset.schema.field(CLAP_FIELD).type.list_size
    if clap_dim % num_sub_vectors != 0:
        raise ValueError(f"num_sub_vectors={num_sub_vectors} does not divide clap dim {clap_dim}")
    rows = dataset.count_rows()
    if rows < MIN_ROWS_FOR_INDEX:
        logger.warning("clap_index_skipped_too_few_rows", rows=rows, minimum=MIN_ROWS_FOR_INDEX)
        return False
    partitions = max(1, round(rows**0.5)) if num_partitions is None else num_partitions
    dataset.create_index(
        CLAP_FIELD,
        index_type="IVF_PQ",
        num_partitions=partitions,
        num_sub_vectors=num_sub_vectors,
        metric=metric,
    )
    logger.info("clap_index_built", rows=rows, num_partitions=partitions, metric=metric)
    return True


def add_embeddings(
    dataset: lance.LanceDataset,
    m2l_encode: M2LEncodeFn,
    clap_encode: ClapEncodeFn,
    sample_rate: int,
    *,
    clap_dim: int = CLAP_EMBEDDING_DIM,
    batch_size: int = DEFAULT_LANCE_BATCH_SIZE,
    log_every_batch: bool = False,
    resume_cache: Path | None = None,
    build_index: bool = True,
    num_partitions: int | None = None,
    num_sub_vectors: int = DEFAULT_NUM_SUB_VECTORS,
    metric: str = DEFAULT_INDEX_METRIC,
) -> None:
    """Append ``m2l`` + ``clap`` columns to ``dataset`` and (optionally) index ``clap``.

    Commits a new dataset version; existing columns are untouched and only
    ``audio`` is read per batch. When ``build_index`` and the dataset has enough
    rows, an IVF_PQ index is built on ``clap`` so ``nearest=`` uses ANN.

    :param dataset: Open Lance dataset carrying an ``audio`` column.
    :param m2l_encode: Maps an ``(B, C, T)`` batch to a ``(B, C*D, T)`` latent batch.
    :param clap_encode: Maps a mono ``(B, T)`` batch and ``sample_rate`` to ``(B, clap_dim)``.
    :param sample_rate: Passed through to ``clap_encode``.
    :param clap_dim: CLAP embedding width; the ``clap`` fixed-size-list length.
    :param batch_size: Rows per UDF call. Ignored for legacy (v1) Lance datasets,
        which Lance rewrites whole.
    :param log_every_batch: Emit an ``embedding_progress`` line for every UDF
        batch instead of only at row/time intervals.
    :param resume_cache: Lance-managed resume cache of per-batch UDF outputs; a rerun
        with the same file, dataset version, and ``batch_size`` skips already
        encoded batches (cached batches bypass the UDF, so resumed-run progress
        counts freshly encoded rows only). Deleted after a successful commit.
    :param build_index: Build an IVF_PQ index on ``clap`` after the column lands.
    :param num_partitions: IVF partition count; ``None`` uses ``round(sqrt(rows))``.
    :param num_sub_vectors: PQ sub-vector count (must divide ``clap_dim``).
    :param metric: Vector-index distance metric.
    :raises ValueError: ``dataset`` already has an ``m2l`` or ``clap`` column
        (re-running on an augmented dataset would hit Lance's opaque "column
        already exists" error), or lacks the ``audio`` column the UDF reads
        (an absent source column would otherwise fail opaquely mid-transaction).
    """
    import lance

    existing = {M2L_FIELD, CLAP_FIELD} & set(dataset.schema.names)
    if existing:
        raise ValueError(f"dataset already has embedding column(s): {sorted(existing)}")
    if AUDIO_FIELD not in dataset.schema.names:
        raise ValueError(f"dataset has no {AUDIO_FIELD!r} column to embed")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    total_rows = dataset.count_rows()
    if total_rows < 1:
        raise ValueError("dataset has no rows to embed")
    logger.info("inferring_embedding_schema")
    sample = next(dataset.to_batches(columns=[AUDIO_FIELD], limit=1))
    sample_output = embeddings_record_batch(
        sample.column(AUDIO_FIELD).to_numpy_ndarray(),
        m2l_encode,
        clap_encode,
        sample_rate,
        clap_dim=clap_dim,
    )
    logger.info("inferred_embedding_schema")
    progress_interval = max(batch_size, (total_rows + MAX_PROGRESS_LOGS - 1) // MAX_PROGRESS_LOGS)
    next_progress_row = progress_interval
    rows_processed = 0
    started_at = time.monotonic()
    last_progress_at = started_at
    last_udf_end = started_at
    stage_ms: dict[str, float] = {}

    def timed(stage: str, encode: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        """Record ``encode``'s wall-clock per call under ``stage`` in ``stage_ms``.

        :param stage: Key the duration is stored under.
        :param encode: Encoder to wrap.
        :returns: Wrapped encoder with unchanged signature and result.
        """

        def wrapper(*args: object) -> np.ndarray:
            encode_started = time.monotonic()
            result = encode(*args)
            stage_ms[stage] = (time.monotonic() - encode_started) * 1000
            return result

        return wrapper

    @lance.batch_udf(
        output_schema=sample_output.schema,
        checkpoint_file=None if resume_cache is None else str(resume_cache),
    )
    def udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        nonlocal next_progress_row, rows_processed, last_progress_at, last_udf_end
        udf_started = time.monotonic()
        audio = batch.column(AUDIO_FIELD).to_numpy_ndarray()
        output = embeddings_record_batch(
            audio,
            timed("m2l", m2l_encode),
            timed("clap", clap_encode),
            sample_rate,
            clap_dim=clap_dim,
        )
        rows_processed += batch.num_rows
        now = time.monotonic()
        interval_due = rows_processed >= next_progress_row or rows_processed == total_rows
        time_due = now - last_progress_at >= PROGRESS_LOG_INTERVAL_SECONDS
        if log_every_batch or interval_due or time_due:
            logger.info(
                "embedding_progress",
                rows_processed=rows_processed,
                total_rows=total_rows,
                percent=round(rows_processed / total_rows * 100, 1),
                rows_per_second=round(rows_processed / max(now - started_at, 1e-9), 1),
                batch_rows=batch.num_rows,
                m2l_ms=round(stage_ms["m2l"], 1),
                clap_ms=round(stage_ms["clap"], 1),
                batch_ms=round((now - udf_started) * 1000, 1),
                # Gap since the previous batch returned: Lance read/write + native
                # overhead — high values point at I/O, not the encoders.
                interbatch_ms=round((udf_started - last_udf_end) * 1000, 1),
            )
            last_progress_at = now
        if interval_due:
            next_progress_row = (rows_processed // progress_interval + 1) * progress_interval
        last_udf_end = time.monotonic()
        return output

    logger.info(
        "embedding_write_started",
        total_rows=total_rows,
        batch_size=batch_size,
        source_version=dataset.version,
    )
    dataset.add_columns(udf, read_columns=[AUDIO_FIELD], batch_size=batch_size)
    if resume_cache is not None:
        # Only useful for resuming the just-committed run; Lance leaves deletion
        # to the caller. The columns are committed, so a failed delete must not
        # fail the run (a rerun would hit the existing-column guard).
        try:
            resume_cache.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "resume_cache_cleanup_failed",
                resume_cache=str(resume_cache),
                error=str(exc),
            )
    logger.info(
        "wrote_embeddings",
        total_rows=total_rows,
        committed_version=dataset.version,
    )
    if build_index:
        build_clap_index(
            dataset, num_partitions=num_partitions, num_sub_vectors=num_sub_vectors, metric=metric
        )


def same_num_latent_frames(num_samples: int, sample_rate: int) -> int:
    """Latent frames SAME emits for a clip, after any resample to 44.1 kHz.

    :param num_samples: Clip length in samples at ``sample_rate``.
    :param sample_rate: Clip sample rate in Hz.
    :returns: ``ceil(resampled / SAME_PAD_BLOCK_SAMPLES) * 2`` — the resampled
        length (torchaudio's ``ceil`` convention) zero-padded to a whole
        two-hop block, then one frame per ``SAME_DOWNSAMPLING_RATIO`` samples.
    :raises ValueError: ``num_samples`` or ``sample_rate`` is non-positive.
    """
    if num_samples < 1 or sample_rate < 1:
        raise ValueError(f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}")
    resampled = math.ceil(num_samples * SAME_SAMPLE_RATE / sample_rate)
    return 2 * math.ceil(resampled / SAME_PAD_BLOCK_SAMPLES)


def same_encoder_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare an audio batch for SAME: float32 stereo at 44.1 kHz.

    :param audio: ``(B, C, T)`` batch, ``C`` 1 (duplicated) or 2 (on-disk float16).
    :param sample_rate: Batch sample rate; non-44.1 kHz input is resampled.
    :returns: ``(B, 2, T')`` float32 batch at ``SAME_SAMPLE_RATE``.
    :raises ValueError: ``audio`` is not rank 3 or has more than two channels.
    """
    if audio.ndim != 3 or audio.shape[1] not in (1, 2):
        raise ValueError(
            f"expected a (B, C, T) batch with 1 or 2 channels for a stereo encoder, "
            f"got shape {audio.shape}"
        )
    prepared = np.ascontiguousarray(audio, dtype=np.float32)
    if prepared.shape[1] == 1:
        prepared = np.repeat(prepared, 2, axis=1)
    if sample_rate != SAME_SAMPLE_RATE:
        import torch
        import torchaudio.functional as audio_fn

        prepared = audio_fn.resample(
            torch.from_numpy(prepared), sample_rate, SAME_SAMPLE_RATE
        ).numpy()
    return prepared


def same_record_batch(
    audio: np.ndarray,
    encoders: Mapping[str, SameEncodeFn],
    sample_rate: int,
    *,
    num_frames: int,
) -> pa.RecordBatch:
    """Encode one audio batch into fixed-shape SAME latent columns.

    :param audio: ``(B, C, T)`` audio batch (any float dtype).
    :param encoders: Column name to encode callable over prepared stereo audio.
    :param sample_rate: Batch sample rate; drives the 44.1 kHz preparation.
    :param num_frames: Expected latent frame count ``T_lat`` for every column.
    :returns: A ``B``-row batch with one ``(SAME_EMBEDDING_DIM, num_frames)``
        fixed-shape-tensor column per encoder.
    :raises ValueError: An encoder output is not ``(B, 256, num_frames)`` or is
        non-finite (the columns are permanent — a NaN/inf row must not land).
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    prepared = same_encoder_input(audio, sample_rate)
    expected_shape = (len(audio), SAME_EMBEDDING_DIM, num_frames)
    columns: dict[str, pa.Array] = {}
    for field, encode in encoders.items():
        latents = np.ascontiguousarray(encode(prepared), dtype=np.float32)
        if latents.shape != expected_shape:
            raise ValueError(
                f"{field} encoder produced shape {latents.shape}, expected {expected_shape}"
            )
        if not np.isfinite(latents).all():
            raise ValueError(f"{field} embeddings contain non-finite values")
        columns[field] = tensor_array(latents, np.dtype("float32"), expected_shape[1:])
    return pa.record_batch(columns)


def add_same_embeddings(
    dataset: lance.LanceDataset,
    encoders: Mapping[str, SameEncodeFn],
    sample_rate: int,
    *,
    batch_size: int = DEFAULT_LANCE_BATCH_SIZE,
) -> None:
    """Append fixed-shape SAME latent columns to ``dataset``.

    Commits a new dataset version; existing columns are untouched and only
    ``audio`` is read per batch. The latent frame count is derived from the
    dataset's fixed audio length, so every row shares one column shape. SAME
    latents are sequences, not search vectors, so no index is built (like m2l).

    :param dataset: Open Lance dataset carrying an ``audio`` column.
    :param encoders: Column name (``same_s`` / ``same_l``) to encode callable.
    :param sample_rate: Dataset audio sample rate in Hz.
    :param batch_size: Rows per UDF call. Ignored for legacy (v1) Lance datasets,
        which Lance rewrites whole.
    :raises ValueError: ``encoders`` is empty, a target column already exists,
        the ``audio`` column is absent, the dataset is empty, or
        ``batch_size < 1``.
    """
    import lance

    if not encoders:
        raise ValueError("no SAME encoders given; nothing to write")
    existing = set(encoders) & set(dataset.schema.names)
    if existing:
        raise ValueError(f"dataset already has embedding column(s): {sorted(existing)}")
    if AUDIO_FIELD not in dataset.schema.names:
        raise ValueError(f"dataset has no {AUDIO_FIELD!r} column to embed")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    total_rows = dataset.count_rows()
    if total_rows < 1:
        raise ValueError("dataset has no rows to embed")

    audio_shape = tuple(dataset.schema.field(AUDIO_FIELD).type.shape)
    num_frames = same_num_latent_frames(audio_shape[-1], sample_rate)
    sample = next(dataset.to_batches(columns=[AUDIO_FIELD], limit=1))
    sample_output = same_record_batch(
        sample.column(AUDIO_FIELD).to_numpy_ndarray(), encoders, sample_rate, num_frames=num_frames
    )
    progress_interval = max(batch_size, (total_rows + MAX_PROGRESS_LOGS - 1) // MAX_PROGRESS_LOGS)
    next_progress_row = progress_interval
    rows_processed = 0
    started_at = time.monotonic()

    @lance.batch_udf(output_schema=sample_output.schema)
    def udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        nonlocal next_progress_row, rows_processed
        audio = batch.column(AUDIO_FIELD).to_numpy_ndarray()
        output = same_record_batch(audio, encoders, sample_rate, num_frames=num_frames)
        rows_processed += batch.num_rows
        if rows_processed >= next_progress_row or rows_processed == total_rows:
            elapsed_seconds = time.monotonic() - started_at
            logger.info(
                "embedding_progress",
                rows_processed=rows_processed,
                total_rows=total_rows,
                percent=round(rows_processed / total_rows * 100, 1),
                rows_per_second=round(rows_processed / max(elapsed_seconds, 1e-9), 1),
                batch_size=batch_size,
            )
            next_progress_row = (rows_processed // progress_interval + 1) * progress_interval
        return output

    dataset.add_columns(udf, read_columns=[AUDIO_FIELD], batch_size=batch_size)


def _configure_lance_logging(*, debug: bool) -> None:
    """Configure native Lance warnings or full debug telemetry before import.

    :param debug: Replace ambient Lance logging with debug-level native telemetry.
    """
    if debug:
        os.environ["LANCE_LOG"] = "debug"
    else:
        os.environ.setdefault("LANCE_LOG", DEFAULT_LANCE_LOG)


def _resolve_torch_device(device: str | None) -> str:
    """Resolve an explicit device or select the fastest available backend.

    :param device: Explicit Torch device, or ``None`` to auto-select.
    :returns: Explicit device, otherwise ``cuda``, ``mps``, or ``cpu`` in priority order.
    """
    import torch

    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_m2l_audio_encoder(device: str | None = None) -> M2LEncodeFn:
    """Build a music2latent encode callable over an ``(B, C, T)`` batch.

    The ``music2latent`` import is deferred so this module stays importable (and
    its core testable with a fake encoder) without loading a checkpoint.

    :param device: Torch device; ``None`` selects cuda, MPS, then cpu.
    :returns: Encode callable mapping ``(B, C, T)`` to ``(B, C*D, T)`` float32.
    """
    from music2latent import EncoderDecoder

    device = _resolve_torch_device(device)
    logger.info("loading_m2l_checkpoint", device=device)
    encoder = EncoderDecoder(device=device)

    def encode(audio: np.ndarray) -> np.ndarray:
        batch, channels = audio.shape[0], audio.shape[1]
        flat = np.ascontiguousarray(rearrange(audio, "b c t -> (b c) t"), dtype=np.float32)
        latents = encoder.encode(flat, max_batch_size=M2L_ENCODE_MAX_BATCH)
        latents = rearrange(latents, "(b c) d t -> b (c d) t", b=batch, c=channels)
        return latents.cpu().numpy()

    return encode


def load_clap_audio_encoder(
    checkpoint: str = DEFAULT_CLAP_CHECKPOINT,
    device: str | None = None,
) -> ClapEncodeFn:
    """Load a CLAP checkpoint and return a mono-batch encode callable.

    The ``torch`` / ``transformers`` imports are deferred so this module stays
    importable without loading a model.

    :param checkpoint: HuggingFace CLAP model id; its audio tower sets the embedding width.
    :param device: Torch device; ``None`` selects cuda, MPS, then cpu.
    :returns: Encode callable mapping a mono ``(B, T)`` batch and sample rate to
        a ``(B, D)`` float32 embedding batch.
    """
    import torch
    import torchaudio.functional as audio_fn
    from transformers import ClapModel, ClapProcessor

    device = _resolve_torch_device(device)
    logger.info("loading_clap_checkpoint", checkpoint=checkpoint, device=device)
    # transformers' own types are too loose for the surface used below, so pyright
    # is scoped off the two offending calls (the from_pretrained `.to()` chain and
    # the get_audio_features output access) rather than widened to Any; the
    # processor's audio kwargs are dict-splatted, which the stub can't object to.
    model = ClapModel.from_pretrained(checkpoint).to(device).eval()  # pyright: ignore
    processor = ClapProcessor.from_pretrained(checkpoint)

    @torch.no_grad()
    def _encode_chunk(chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        if sample_rate != CLAP_SAMPLE_RATE:
            wav = audio_fn.resample(wav, sample_rate, CLAP_SAMPLE_RATE)
        # `audio=` (singular) per transformers>=5; the splat also hides the stub's
        # too-narrow __call__ signature from pyright.
        clap_kwargs = {
            "audio": list(wav.numpy()),
            "sampling_rate": CLAP_SAMPLE_RATE,
            "return_tensors": "pt",
        }
        inputs = processor(**clap_kwargs)
        device_inputs = {key: value.to(device) for key, value in inputs.items()}
        # transformers>=5 returns a BaseModelOutputWithPooling; the projected
        # (B, CLAP_EMBEDDING_DIM) audio embedding is its pooler_output.
        features = model.get_audio_features(**device_inputs)
        return features.pooler_output.cpu().numpy()  # pyright: ignore

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        # Sub-batch the forward so the whole UDF read batch never hits the GPU at
        # once; each chunk's result is pulled to CPU before the next runs.
        chunks = [
            _encode_chunk(mono[start : start + CLAP_ENCODE_MAX_BATCH], sample_rate)
            for start in range(0, len(mono), CLAP_ENCODE_MAX_BATCH)
        ]
        return np.concatenate(chunks, axis=0)

    return encode


def _resolve_same_checkpoint_dir(checkpoint: str) -> Path:
    """Resolve a SAME checkpoint reference to a local directory.

    :param checkpoint: Local directory, ``r2://`` mirror prefix (fetched into a
        local cache via credentialed rclone), or HuggingFace repo id (fetched
        through ``huggingface_hub``'s own cache).
    :returns: Directory holding ``model.safetensors`` + ``model_config.json``.
    """
    if r2_io.is_r2_uri(checkpoint):
        # Key on the full bucket/key path: distinct URIs sharing a final path
        # component must not collide (download_dir_no_overwrite hard-fails on
        # a populated directory holding a different checkpoint).
        cache_key = checkpoint.removeprefix("r2://").strip("/")
        cache_dir = Path.home() / ".cache" / "synth-setter" / "models" / cache_key
        r2_io.ensure_r2_env_loaded()
        r2_io.download_dir_no_overwrite(checkpoint, cache_dir)
        return cache_dir
    local = Path(checkpoint)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(checkpoint))


def load_same_audio_encoder(checkpoint: str, device: str | None = None) -> SameEncodeFn:
    """Load a SAME checkpoint and return a prepared-stereo-batch encode callable.

    The ``stable_audio_tools`` / ``torch`` imports are deferred so this module
    stays importable (and its core testable with a fake encoder) without the
    optional dependency or a checkpoint.

    :param checkpoint: Local directory, ``r2://`` mirror, or HuggingFace repo id.
    :param device: Torch device; ``None`` selects cuda, MPS, then cpu.
    :returns: Encode callable mapping prepared ``(B, 2, T)`` 44.1 kHz audio to
        ``(B, SAME_EMBEDDING_DIM, T_lat)`` float32 latents.
    :raises ImportError: ``stable_audio_tools`` is not installed.
    """
    import json

    import torch
    from safetensors.torch import load_file

    try:
        from stable_audio_tools.models.factory import create_model_from_config
    except ImportError as exc:
        raise ImportError(
            "loading SAME encoders requires the optional `same` extra — "
            "install it with `uv sync --extra same`"
        ) from exc

    checkpoint_dir = _resolve_same_checkpoint_dir(checkpoint)
    device = _resolve_torch_device(device)
    logger.info("loading_same_checkpoint", checkpoint=checkpoint, device=device)
    model_config = json.loads((checkpoint_dir / "model_config.json").read_text())
    model = create_model_from_config(model_config)
    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"))
    model = model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def _encode_chunk(chunk: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)).to(device)
        # The factory returns an untyped union of model classes whose `encode`
        # pyright cannot resolve; the SAME autoencoder returns a plain Tensor
        # (pinned by the real-weights e2e tests), so the call is scoped off.
        latents: torch.Tensor = model.encode(wav)  # pyright: ignore
        return latents.float().cpu().numpy()

    def encode(stereo: np.ndarray) -> np.ndarray:
        # Sub-batch the forward so the whole UDF read batch never hits the GPU
        # at once; each chunk's result is pulled to CPU before the next runs.
        chunks = [
            _encode_chunk(stereo[start : start + SAME_ENCODE_MAX_BATCH])
            for start in range(0, len(stereo), SAME_ENCODE_MAX_BATCH)
        ]
        return np.concatenate(chunks, axis=0)

    return encode


def _run_same_mode(
    dataset: lance.LanceDataset,
    *,
    uri: str,
    checkpoints: Mapping[str, str],
    sample_rate: int,
    device: str | None,
    batch_size: int,
) -> None:
    """Drive the SAME-only CLI mode: load encoders, append columns, exit on failure.

    :param dataset: Open Lance dataset to augment.
    :param uri: Dataset URI, for logging only.
    :param checkpoints: Target column name to checkpoint source.
    :param sample_rate: Dataset audio sample rate in Hz.
    :param device: Torch device for the encoders; ``None`` auto-selects.
    :param batch_size: Rows per UDF call.
    """
    existing = set(checkpoints) & set(dataset.schema.names)
    if existing:
        logger.error("embeddings_already_present", uri=uri, columns=sorted(existing))
        sys.exit(1)
    logger.info(
        "adding_same_embeddings",
        uri=uri,
        columns=sorted(checkpoints),
        sample_rate=sample_rate,
        rows=dataset.count_rows(),
        batch_size=batch_size,
    )
    try:
        encoders = {
            field: load_same_audio_encoder(checkpoint, device)
            for field, checkpoint in checkpoints.items()
        }
        add_same_embeddings(dataset, encoders, sample_rate, batch_size=batch_size)
    # Encoder load (missing dep / weights) or the add step (validation, Lance,
    # CUDA) should exit cleanly with a logged cause, not a raw CLI traceback.
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.error("add_embeddings_failed", uri=uri, error=str(exc))
        sys.exit(1)
    logger.info("added_embeddings", uri=uri, columns=sorted(checkpoints))


def _open_lance_dataset(uri: str) -> lance.LanceDataset:
    """Open a Lance dataset, attaching R2 ``storage_options`` for cloud URIs.

    Any ``s3://`` URI is treated as the project's R2 (S3-compatible) endpoint and
    credentialed via :func:`r2_io.r2_storage_options`; generic non-R2 S3 buckets
    are not a supported input.

    :param uri: Local path, ``r2://bucket/key``, or ``s3://bucket/key`` (R2).
    :returns: The opened dataset, credentialed when ``uri`` is on R2.
    """
    import lance

    if r2_io.is_r2_uri(uri):
        uri = r2_io.to_s3_uri(uri)
    if uri.startswith("s3://"):
        r2_io.ensure_r2_env_loaded()
        return lance.dataset(uri, storage_options=r2_io.r2_storage_options())
    return lance.dataset(uri)


@click.command()
@click.argument("lance_uri", type=str)
@click.option(
    "--debug",
    is_flag=True,
    help="Log every batch with stage timings and enable native Lance debug telemetry.",
)
@click.option(
    "--clap-checkpoint",
    default=DEFAULT_CLAP_CHECKPOINT,
    show_default=True,
    help="HuggingFace CLAP model id.",
)
@click.option(
    "--device",
    default=None,
    help="Torch device for both encoders (defaults to cuda, MPS, then cpu).",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_LANCE_BATCH_SIZE,
    show_default=True,
    help="Rows per UDF call (ignored for v1 datasets).",
)
@click.option(
    "--resume-cache",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help=(
        "Cache per-batch encoder outputs here; rerunning with the same file "
        "resumes an interrupted run. Deleted on success."
    ),
)
@click.option(
    "--build-index/--no-build-index",
    default=True,
    show_default=True,
    help="Build an IVF_PQ vector index on the clap column.",
)
@click.option(
    "--num-partitions",
    type=int,
    default=None,
    help="IVF partition count; defaults to round(sqrt(rows)).",
)
@click.option(
    "--num-sub-vectors",
    type=int,
    default=DEFAULT_NUM_SUB_VECTORS,
    show_default=True,
    help="PQ sub-vector count (must divide the clap dim).",
)
@click.option(
    "--metric",
    default=DEFAULT_INDEX_METRIC,
    show_default=True,
    help="Vector-index distance metric.",
)
@click.option(
    "--same",
    "same_variants",
    type=click.Choice(["s", "l"]),
    multiple=True,
    help="Append only the selected SAME latent column(s) instead of m2l+clap.",
)
@click.option(
    "--same-s-checkpoint",
    default=DEFAULT_SAME_S_CHECKPOINT,
    show_default=True,
    help="SAME-S weights: local dir, r2:// mirror, or HuggingFace repo id.",
)
@click.option(
    "--same-l-checkpoint",
    default=DEFAULT_SAME_L_CHECKPOINT,
    show_default=True,
    help="SAME-L weights: local dir, r2:// mirror, or HuggingFace repo id.",
)
def main(
    lance_uri: str,
    debug: bool,
    clap_checkpoint: str,
    device: str | None,
    batch_size: int,
    resume_cache: Path | None,
    build_index: bool,
    num_partitions: int | None,
    num_sub_vectors: int,
    metric: str,
    same_variants: tuple[str, ...],
    same_s_checkpoint: str,
    same_l_checkpoint: str,
) -> None:
    """Add ``m2l`` + ``clap`` embedding columns to the Lance dataset at LANCE_URI.

    LANCE_URI is a dataset from generate_dataset or finalize_dataset (local path,
    ``r2://``, or ``s3://``); its audio sample rate is read from the shard
    metadata in the schema. The ``clap`` column is a vector-searchable
    fixed-size list; with ``--build-index`` it also gets an IVF_PQ index.

    :param lance_uri: Dataset directory to augment in place.
    :param debug: Log every batch with stage timings and enable native Lance
        debug telemetry.
    :param clap_checkpoint: HuggingFace CLAP model id.
    :param device: Torch device for both encoders; ``None`` selects cuda, MPS, then cpu.
    :param batch_size: Rows per UDF call.
    :param resume_cache: Optional per-batch output cache enabling resume of an
        interrupted run; deleted after a successful commit.
    :param build_index: Build an IVF_PQ index on the clap column after writing it.
    :param num_partitions: IVF partition count; ``None`` uses ``round(sqrt(rows))``.
    :param num_sub_vectors: PQ sub-vector count; must divide the clap dim.
    :param metric: Vector-index distance metric.
    :param same_variants: SAME variants to append; non-empty switches to
        SAME-only mode (so an m2l/clap-augmented dataset can be extended).
    :param same_s_checkpoint: SAME-S weight source.
    :param same_l_checkpoint: SAME-L weight source.
    """
    _configure_lance_logging(debug=debug)
    logger.info("lance_logging_configured", native_level=os.environ["LANCE_LOG"])
    try:
        from synth_setter.pipeline.data.lance_shard import read_shard_metadata

        dataset = _open_lance_dataset(lance_uri)
        sample_rate = int(read_shard_metadata(dataset.schema).sample_rate)
    # RuntimeError covers missing R2 creds / rclone from the cloud-URI path.
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("open_dataset_failed", uri=lance_uri, error=str(exc))
        sys.exit(1)
    if same_variants:
        checkpoints = {SAME_S_FIELD: same_s_checkpoint, SAME_L_FIELD: same_l_checkpoint}
        selected = {f"same_{variant}" for variant in same_variants}
        _run_same_mode(
            dataset,
            uri=lance_uri,
            checkpoints={field: checkpoints[field] for field in sorted(selected)},
            sample_rate=sample_rate,
            device=device,
            batch_size=batch_size,
        )
        return
    existing = {M2L_FIELD, CLAP_FIELD} & set(dataset.schema.names)
    if existing:
        logger.error("embeddings_already_present", uri=lance_uri, columns=sorted(existing))
        sys.exit(1)
    logger.info(
        "adding_embeddings",
        uri=lance_uri,
        sample_rate=sample_rate,
        rows=dataset.count_rows(),
        batch_size=batch_size,
    )
    try:
        m2l_encode = load_m2l_audio_encoder(device)
        clap_encode = load_clap_audio_encoder(clap_checkpoint, device)
        add_embeddings(
            dataset,
            m2l_encode,
            clap_encode,
            sample_rate,
            batch_size=batch_size,
            log_every_batch=debug,
            resume_cache=resume_cache,
            build_index=build_index,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            metric=metric,
        )
    # Encoder load (missing dep) or the add/index step (validation, Lance, CUDA)
    # should exit cleanly with a logged cause, not a raw CLI traceback.
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.error("add_embeddings_failed", uri=lance_uri, error=str(exc))
        sys.exit(1)
    logger.info("added_embeddings", uri=lance_uri, columns=[M2L_FIELD, CLAP_FIELD])


if __name__ == "__main__":
    main()
