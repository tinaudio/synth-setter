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

Both embedders are injected callables, so the core runs without a checkpoint;
:func:`load_m2l_audio_encoder` / :func:`load_clap_audio_encoder` build the real
encoders behind lazy ``music2latent`` / ``transformers`` imports.

This is a sanctioned post-finalize augmenter: it commits a new Lance version of
an existing dataset rather than writing fresh ``data/`` shards, so it does not
cross the worker/finalize write boundary.

CLI: ``python -m synth_setter.pipeline.data.add_embeddings DATASET.lance``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TypeAlias

import click
import lance
import numpy as np
import pyarrow as pa
import structlog
from einops import rearrange

from synth_setter.data.vst.shapes import AUDIO_FIELD, CLAP_FIELD, M2L_FIELD
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_shard import read_shard_metadata, tensor_array

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
# IVF_PQ needs ~256 training vectors (256 PQ centroids); below this the index is
# skipped and callers fall back to Lance's exact (brute-force) ``nearest`` scan.
MIN_ROWS_FOR_INDEX: int = 256
# 512 % 16 == 0; each PQ sub-quantizer covers a 32-d slice of the CLAP vector.
DEFAULT_NUM_SUB_VECTORS: int = 16
# Cosine matches CLAP's L2-normalized audio embeddings.
DEFAULT_INDEX_METRIC: str = "cosine"

M2LEncodeFn: TypeAlias = Callable[[np.ndarray], np.ndarray]
ClapEncodeFn: TypeAlias = Callable[[np.ndarray, int], np.ndarray]


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
    batch_size: int | None = None,
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
    :param batch_size: Rows per UDF call; ``None`` uses the Lance default. Ignored
        for legacy (v1) Lance datasets, which Lance rewrites whole.
    :param build_index: Build an IVF_PQ index on ``clap`` after the column lands.
    :param num_partitions: IVF partition count; ``None`` uses ``round(sqrt(rows))``.
    :param num_sub_vectors: PQ sub-vector count (must divide ``clap_dim``).
    :param metric: Vector-index distance metric.
    :raises ValueError: ``dataset`` already has an ``m2l`` or ``clap`` column
        (re-running on an augmented dataset would hit Lance's opaque "column
        already exists" error), or lacks the ``audio`` column the UDF reads
        (an absent source column would otherwise fail opaquely mid-transaction).
    """
    existing = {M2L_FIELD, CLAP_FIELD} & set(dataset.schema.names)
    if existing:
        raise ValueError(f"dataset already has embedding column(s): {sorted(existing)}")
    if AUDIO_FIELD not in dataset.schema.names:
        raise ValueError(f"dataset has no {AUDIO_FIELD!r} column to embed")

    @lance.batch_udf()
    def udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        audio = batch.column(AUDIO_FIELD).to_numpy_ndarray()
        return embeddings_record_batch(
            audio, m2l_encode, clap_encode, sample_rate, clap_dim=clap_dim
        )

    dataset.add_columns(udf, read_columns=[AUDIO_FIELD], batch_size=batch_size)
    if build_index:
        build_clap_index(
            dataset, num_partitions=num_partitions, num_sub_vectors=num_sub_vectors, metric=metric
        )


def load_m2l_audio_encoder() -> M2LEncodeFn:
    """Build a music2latent encode callable over an ``(B, C, T)`` batch.

    The ``music2latent`` import is deferred so this module stays importable (and
    its core testable with a fake encoder) without loading a checkpoint. No
    device knob: music2latent owns its own placement.

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


def load_clap_audio_encoder(
    checkpoint: str = DEFAULT_CLAP_CHECKPOINT,
    device: str | None = None,
) -> ClapEncodeFn:
    """Load a CLAP checkpoint and return a mono-batch encode callable.

    The ``torch`` / ``transformers`` imports are deferred so this module stays
    importable without loading a model.

    :param checkpoint: HuggingFace CLAP model id; its audio tower sets the embedding width.
    :param device: Torch device; ``None`` selects cuda when available, else cpu.
    :returns: Encode callable mapping a mono ``(B, T)`` batch and sample rate to
        a ``(B, D)`` float32 embedding batch.
    """
    import torch
    import torchaudio.functional as audio_fn
    from transformers import ClapModel, ClapProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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


def _open_lance_dataset(uri: str) -> lance.LanceDataset:
    """Open a Lance dataset, attaching R2 ``storage_options`` for cloud URIs.

    Any ``s3://`` URI is treated as the project's R2 (S3-compatible) endpoint and
    credentialed via :func:`r2_io.r2_storage_options`; generic non-R2 S3 buckets
    are not a supported input.

    :param uri: Local path, ``r2://bucket/key``, or ``s3://bucket/key`` (R2).
    :returns: The opened dataset, credentialed when ``uri`` is on R2.
    """
    if r2_io.is_r2_uri(uri):
        uri = r2_io.to_s3_uri(uri)
    if uri.startswith("s3://"):
        r2_io.ensure_r2_env_loaded()
        return lance.dataset(uri, storage_options=r2_io.r2_storage_options())
    return lance.dataset(uri)


@click.command()
@click.argument("lance_uri", type=str)
@click.option(
    "--clap-checkpoint",
    default=DEFAULT_CLAP_CHECKPOINT,
    show_default=True,
    help="HuggingFace CLAP model id.",
)
@click.option(
    "--device",
    default=None,
    help="Torch device for CLAP (defaults to cuda when available, else cpu).",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Rows per UDF call; defaults to the Lance default (ignored for v1 datasets).",
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
def main(
    lance_uri: str,
    clap_checkpoint: str,
    device: str | None,
    batch_size: int | None,
    build_index: bool,
    num_partitions: int | None,
    num_sub_vectors: int,
    metric: str,
) -> None:
    """Add ``m2l`` + ``clap`` embedding columns to the Lance dataset at LANCE_URI.

    LANCE_URI is a dataset from generate_dataset or finalize_dataset (local path,
    ``r2://``, or ``s3://``); its audio sample rate is read from the shard
    metadata in the schema. The ``clap`` column is a vector-searchable
    fixed-size list; with ``--build-index`` it also gets an IVF_PQ index.

    :param lance_uri: Dataset directory to augment in place.
    :param clap_checkpoint: HuggingFace CLAP model id.
    :param device: Torch device for CLAP; ``None`` selects cuda when available, else cpu.
    :param batch_size: Rows per UDF call; ``None`` uses the Lance default.
    :param build_index: Build an IVF_PQ index on the clap column after writing it.
    :param num_partitions: IVF partition count; ``None`` uses ``round(sqrt(rows))``.
    :param num_sub_vectors: PQ sub-vector count; must divide the clap dim.
    :param metric: Vector-index distance metric.
    """
    try:
        dataset = _open_lance_dataset(lance_uri)
        sample_rate = int(read_shard_metadata(dataset.schema).sample_rate)
    # RuntimeError covers missing R2 creds / rclone from the cloud-URI path.
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("open_dataset_failed", uri=lance_uri, error=str(exc))
        sys.exit(1)
    existing = {M2L_FIELD, CLAP_FIELD} & set(dataset.schema.names)
    if existing:
        logger.error("embeddings_already_present", uri=lance_uri, columns=sorted(existing))
        sys.exit(1)
    logger.info(
        "adding_embeddings", uri=lance_uri, sample_rate=sample_rate, rows=dataset.count_rows()
    )
    m2l_encode = load_m2l_audio_encoder()
    clap_encode = load_clap_audio_encoder(clap_checkpoint, device)
    add_embeddings(
        dataset,
        m2l_encode,
        clap_encode,
        sample_rate,
        batch_size=batch_size,
        build_index=build_index,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        metric=metric,
    )
    logger.info("added_embeddings", uri=lance_uri, columns=[M2L_FIELD, CLAP_FIELD])


if __name__ == "__main__":
    main()
