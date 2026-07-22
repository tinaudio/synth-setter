#!/usr/bin/env python
"""Append registry-selected audio-embedding columns to a finalized Lance dataset.

The registry keeps checkpoint loading, Arrow encoding, residency, optional dependencies, and
index policy together for each embedding. Co-resident encoders share one Lance UDF pass; large
SAME encoders run in separate load-write-release passes.

CLI: ``synth-setter-add-embeddings lance_uri=DATASET.lance embeddings=[clap,m2l]``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
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
from synth_setter.workspace import operator_workspace

if TYPE_CHECKING:
    import lance
    from omegaconf import DictConfig

    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

logger = structlog.get_logger(__name__)
operator_workspace()

DEFAULT_CLAP_CHECKPOINT: str = "laion/clap-htsat-unfused"
DEFAULT_M2L_CHECKPOINT: str = ""
DEFAULT_SAME_S_CHECKPOINT: str = "r2://intermediate-data/models/same-s"
DEFAULT_SAME_L_CHECKPOINT: str = "r2://intermediate-data/models/same-l"
CLAP_SAMPLE_RATE: int = 48000
CLAP_EMBEDDING_DIM: int = 512
M2L_ENCODE_MAX_BATCH: int = 64
CLAP_ENCODE_MAX_BATCH: int = 32
DEFAULT_LANCE_BATCH_SIZE: int = 128
MAX_PROGRESS_LOGS: int = 20
MIN_ROWS_FOR_INDEX: int = 256
DEFAULT_NUM_SUB_VECTORS: int = 16
DEFAULT_INDEX_METRIC: str = "cosine"
DEFAULT_LANCE_LOG: str = "warn"
PROGRESS_LOG_INTERVAL_SECONDS: float = 30.0
SAME_EMBEDDING_DIM: int = 256
SAME_SAMPLE_RATE: int = 44100
SAME_DOWNSAMPLING_RATIO: int = 4096
SAME_PAD_BLOCK_SAMPLES: int = 2 * SAME_DOWNSAMPLING_RATIO
SAME_LATENT_FRAMES: int = 44
SAME_ENCODE_MAX_BATCH: int = 16

type M2LEncodeFn = Callable[[np.ndarray], np.ndarray]
type ClapEncodeFn = Callable[[np.ndarray, int], np.ndarray]
type SameEncodeFn = Callable[[np.ndarray], np.ndarray]
type Encoder = M2LEncodeFn | ClapEncodeFn | SameEncodeFn
type LoadEncoderFn = Callable[[str, str | None], Encoder]
type EncodeColumnFn = Callable[[np.ndarray, int, Encoder], pa.Array]


@dataclass(frozen=True)
class IndexSpec:
    """Declare the default vector-index policy for one embedding column.

    .. attribute :: metric

        Lance distance metric.

    .. attribute :: num_sub_vectors

        PQ sub-vector count.
    """

    metric: str = DEFAULT_INDEX_METRIC
    num_sub_vectors: int = DEFAULT_NUM_SUB_VECTORS


@dataclass(frozen=True)
class EmbeddingSpec:
    """Declare one selectable embedding's complete write policy.

    .. attribute :: name

        Registry key and config token.

    .. attribute :: column

        Lance column written by the encoder.

    .. attribute :: default_checkpoint

        Checkpoint source used without a keyed config override.

    .. attribute :: requires_extra

        Optional uv extra required before loading, or ``None``.

    .. attribute :: co_resident

        Whether the encoder may share a UDF pass with other selected encoders.

    .. attribute :: index

        Vector-index policy, or ``None`` for sequence embeddings.

    .. attribute :: load_encoder

        Checkpoint and device to encoder factory.

    .. attribute :: encode_column

        Audio batch, sample rate, and encoder to Arrow array transform.
    """

    name: str
    column: str
    default_checkpoint: str
    requires_extra: str | None
    co_resident: bool
    index: IndexSpec | None
    load_encoder: LoadEncoderFn
    encode_column: EncodeColumnFn


def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Average ``(B, C, T)`` audio into ``(B, T)`` float32 mono.

    :param audio: Audio with one or more channels.
    :returns: Float32 mono audio.
    """
    return audio.mean(axis=1, dtype=np.float32)


def _clap_fixed_size_list(clap: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    """Pack ``(B, dim)`` vectors as a Lance-indexable Arrow array.

    :param clap: Float-compatible CLAP vectors.
    :param dim: Fixed vector width.
    :returns: Fixed-size-list float32 array.
    """
    flat = pa.array(np.ascontiguousarray(clap, dtype=np.float32).reshape(-1), pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def _finite_embedding(field: str, embedding: np.ndarray) -> np.ndarray:
    """Reject non-finite values before a permanent column commit.

    :param field: Target column named in failures.
    :param embedding: Encoder output to validate.
    :returns: Contiguous float32 embedding.
    :raises ValueError: The embedding contains NaN or infinity.
    """
    contiguous = np.ascontiguousarray(embedding, dtype=np.float32)
    if not np.isfinite(contiguous).all():
        raise ValueError(f"{field} embeddings contain non-finite values")
    return contiguous


def _encode_m2l_column(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
    """Encode one audio batch as a fixed-shape m2l tensor column.

    :param audio: ``(B, C, T)`` audio batch.
    :param sample_rate: Unused source sample rate.
    :param encoder: m2l encoder over the original channel layout.
    :returns: Fixed-shape tensor array.
    :raises ValueError: The encoder returns the wrong row count, rank, or non-finite values.
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    del sample_rate
    encode = cast("M2LEncodeFn", encoder)
    latents = _finite_embedding(M2L_FIELD, encode(audio))
    if latents.ndim < 2 or len(latents) != len(audio):
        raise ValueError(
            f"{M2L_FIELD} encoder produced shape {latents.shape}, expected {len(audio)} rows "
            "with at least one embedding dimension"
        )
    return tensor_array(latents, np.dtype("float32"), latents.shape[1:])


def _encode_clap_column(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
    """Encode one audio batch as fixed-width CLAP vectors.

    :param audio: ``(B, C, T)`` audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: CLAP encoder over mono audio.
    :returns: Fixed-size-list float32 array.
    :raises ValueError: The encoder returns the wrong shape or non-finite values.
    """
    encode = cast("ClapEncodeFn", encoder)
    vectors = _finite_embedding(CLAP_FIELD, encode(_downmix_to_mono(audio), sample_rate))
    expected_shape = (len(audio), CLAP_EMBEDDING_DIM)
    if vectors.shape != expected_shape:
        raise ValueError(
            f"{CLAP_FIELD} encoder produced shape {vectors.shape}, expected {expected_shape}"
        )
    return _clap_fixed_size_list(vectors, CLAP_EMBEDDING_DIM)


def same_num_latent_frames(num_samples: int, sample_rate: int) -> int:
    """Return SAME's padded latent-frame count after resampling to 44.1 kHz.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Even latent-frame count after resampling and two-hop padding.
    :raises ValueError: Either input is non-positive.
    """
    if num_samples < 1 or sample_rate < 1:
        raise ValueError(f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}")
    resampled = math.ceil(num_samples * SAME_SAMPLE_RATE / sample_rate)
    return 2 * math.ceil(resampled / SAME_PAD_BLOCK_SAMPLES)


def same_encoder_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare ``(B, C, T)`` audio as float32 stereo at 44.1 kHz.

    :param audio: Audio with one or two channels.
    :param sample_rate: Source sample rate in Hz.
    :returns: Prepared stereo audio.
    :raises ValueError: Audio is not rank three or has an unsupported channel count.
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


def _encode_same_column(field: str, audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
    """Encode one audio batch as a fixed-shape SAME tensor column.

    :param field: SAME target column.
    :param audio: ``(B, C, T)`` source audio.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: SAME encoder over prepared stereo audio.
    :returns: Fixed-shape tensor array.
    :raises ValueError: The encoder returns the wrong shape or non-finite values.
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    prepared = same_encoder_input(audio, sample_rate)
    encode = cast("SameEncodeFn", encoder)
    latents = _finite_embedding(field, encode(prepared))
    expected_shape = (
        len(audio),
        SAME_EMBEDDING_DIM,
        same_num_latent_frames(prepared.shape[-1], SAME_SAMPLE_RATE),
    )
    if latents.shape != expected_shape:
        raise ValueError(f"{field} encoder produced shape {latents.shape}, expected {expected_shape}")
    return tensor_array(latents, np.dtype("float32"), expected_shape[1:])


def _encode_same_s_column(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
    """Encode a SAME-S Arrow column through the shared SAME contract.

    :param audio: Source audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: SAME-S encoder.
    :returns: Fixed-shape tensor array.
    """
    return _encode_same_column(SAME_S_FIELD, audio, sample_rate, encoder)


def _encode_same_l_column(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
    """Encode a SAME-L Arrow column through the shared SAME contract.

    :param audio: Source audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: SAME-L encoder.
    :returns: Fixed-shape tensor array.
    """
    return _encode_same_column(SAME_L_FIELD, audio, sample_rate, encoder)


def _load_m2l_spec_encoder(checkpoint: str, device: str | None) -> Encoder:
    """Load music2latent through the registry's uniform factory signature.

    :param checkpoint: Unused registry placeholder.
    :param device: Torch device, or ``None`` for automatic selection.
    :returns: m2l encoder.
    """
    del checkpoint
    return load_m2l_audio_encoder(device)


def _load_clap_spec_encoder(checkpoint: str, device: str | None) -> Encoder:
    """Load CLAP through the registry's uniform factory signature.

    :param checkpoint: HuggingFace CLAP model id.
    :param device: Torch device, or ``None`` for automatic selection.
    :returns: CLAP encoder.
    """
    return load_clap_audio_encoder(checkpoint, device)


def _load_same_spec_encoder(checkpoint: str, device: str | None) -> Encoder:
    """Load SAME through the registry's uniform factory signature.

    :param checkpoint: SAME checkpoint source.
    :param device: Torch device, or ``None`` for automatic selection.
    :returns: SAME encoder.
    """
    return load_same_audio_encoder(checkpoint, device)


EMBEDDING_REGISTRY: dict[str, EmbeddingSpec] = {
    "clap": EmbeddingSpec(
        name="clap",
        column=CLAP_FIELD,
        default_checkpoint=DEFAULT_CLAP_CHECKPOINT,
        requires_extra=None,
        co_resident=True,
        index=IndexSpec(),
        load_encoder=_load_clap_spec_encoder,
        encode_column=_encode_clap_column,
    ),
    "m2l": EmbeddingSpec(
        name="m2l",
        column=M2L_FIELD,
        default_checkpoint=DEFAULT_M2L_CHECKPOINT,
        requires_extra=None,
        co_resident=True,
        index=None,
        load_encoder=_load_m2l_spec_encoder,
        encode_column=_encode_m2l_column,
    ),
    "same_s": EmbeddingSpec(
        name="same_s",
        column=SAME_S_FIELD,
        default_checkpoint=DEFAULT_SAME_S_CHECKPOINT,
        requires_extra="same",
        co_resident=False,
        index=None,
        load_encoder=_load_same_spec_encoder,
        encode_column=_encode_same_s_column,
    ),
    "same_l": EmbeddingSpec(
        name="same_l",
        column=SAME_L_FIELD,
        default_checkpoint=DEFAULT_SAME_L_CHECKPOINT,
        requires_extra="same",
        co_resident=False,
        index=None,
        load_encoder=_load_same_spec_encoder,
        encode_column=_encode_same_l_column,
    ),
}


def _guard_existing_columns(
    dataset: lance.LanceDataset, specs: Sequence[EmbeddingSpec]
) -> None:
    """Reject selected columns already present in the dataset.

    :param dataset: Open Lance dataset.
    :param specs: Selected embedding policies.
    :raises ValueError: Any selected target column already exists.
    """
    existing = {spec.column for spec in specs} & set(dataset.schema.names)
    if existing:
        raise ValueError(f"dataset already has embedding column(s): {sorted(existing)}")


def _require_extras(specs: Sequence[EmbeddingSpec]) -> None:
    """Fail before checkpoint downloads when a selected optional extra is absent.

    :param specs: Selected embedding policies.
    :raises ImportError: A selected embedding's optional dependency is unavailable.
    """
    required = {spec.requires_extra for spec in specs if spec.requires_extra is not None}
    for extra in sorted(required):
        module = "stable_audio_tools" if extra == "same" else extra
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            raise ImportError(
                f"embedding selection requires the optional `{extra}` extra — "
                f"install it with `uv sync --extra {extra}`"
            )


def _validate_write_source(dataset: lance.LanceDataset, batch_size: int) -> int:
    """Validate source-column and row-count preconditions for one UDF commit.

    :param dataset: Open Lance dataset.
    :param batch_size: Requested rows per UDF call.
    :returns: Positive source row count.
    :raises ValueError: Audio is absent, the dataset is empty, or batch size is non-positive.
    """
    if AUDIO_FIELD not in dataset.schema.names:
        raise ValueError(f"dataset has no {AUDIO_FIELD!r} column to embed")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    total_rows = dataset.count_rows()
    if total_rows < 1:
        raise ValueError("dataset has no rows to embed")
    return total_rows


def _delete_resume_cache(resume_cache: Path | None) -> None:
    """Best-effort delete a consumed UDF resume cache after commit.

    :param resume_cache: Cache path, or ``None`` for a cacheless run.
    """
    if resume_cache is None:
        return
    try:
        resume_cache.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "resume_cache_cleanup_failed",
            resume_cache=str(resume_cache),
            error=str(exc),
        )


def _resume_cache_for_specs(
    resume_cache: Path | None,
    selected_names: tuple[str, ...],
    specs: Sequence[EmbeddingSpec],
) -> Path | None:
    """Give each multi-commit pass an output-schema-specific resume cache.

    :param resume_cache: User-selected cache path.
    :param selected_names: Full run selection.
    :param specs: Policies written by this pass.
    :returns: Unchanged single-pass cache or a pass-specific sibling path.
    """
    if resume_cache is None or len(specs) == len(selected_names):
        return resume_cache
    suffix = "-".join(spec.name for spec in specs)
    return resume_cache.with_name(f"{resume_cache.name}.{suffix}")


def _load_encoders(
    specs: Sequence[EmbeddingSpec], config: AddEmbeddingsConfig
) -> list[Encoder]:
    """Load selected encoders in policy order.

    :param specs: Policies sharing this UDF pass.
    :param config: Checkpoint overrides and device selection.
    :returns: Encoders aligned positionally with ``specs``.
    """
    encoders: list[Encoder] = []
    for spec in specs:
        checkpoint = config.checkpoints.get(spec.name, spec.default_checkpoint)
        encoders.append(spec.load_encoder(checkpoint, config.device))
    return encoders


def _encode_columns(
    audio: np.ndarray,
    sample_rate: int,
    specs: Sequence[EmbeddingSpec],
    encoders: Sequence[Encoder],
    stage_ms: dict[str, float] | None = None,
) -> pa.RecordBatch:
    """Encode one decoded audio batch through every policy in a UDF pass.

    :param audio: Decoded ``(B, C, T)`` audio shared by all policies.
    :param sample_rate: Dataset sample rate in Hz.
    :param specs: Policies sharing this pass.
    :param encoders: Encoders aligned with ``specs``.
    :param stage_ms: Optional destination for per-encoder wall times.
    :returns: Record batch containing each selected embedding column.
    """
    columns: dict[str, pa.Array] = {}
    for spec, encoder in zip(specs, encoders, strict=True):
        started_at = time.monotonic()
        columns[spec.column] = spec.encode_column(audio, sample_rate, encoder)
        if stage_ms is not None:
            stage_ms[spec.name] = (time.monotonic() - started_at) * 1000
    return pa.record_batch(columns)


def _write_columns(
    dataset: lance.LanceDataset,
    specs: Sequence[EmbeddingSpec],
    sample_rate: int,
    config: AddEmbeddingsConfig,
) -> None:
    """Append one co-resident policy group as a single Lance UDF commit.

    :param dataset: Open Lance dataset carrying fixed-shape audio.
    :param specs: Non-empty policy group whose encoders may coexist.
    :param sample_rate: Dataset sample rate in Hz.
    :param config: Batch, checkpoint, logging, and resume settings.
    :raises ValueError: Policies are empty or dataset write preconditions fail.
    """
    import lance

    if not specs:
        raise ValueError("no embedding specs given; nothing to write")
    _guard_existing_columns(dataset, specs)
    total_rows = _validate_write_source(dataset, config.batch_size)
    encoders = _load_encoders(specs, config)
    resume_cache = _resume_cache_for_specs(config.resume_cache, config.embeddings, specs)

    logger.info("inferring_embedding_schema", columns=[spec.column for spec in specs])
    sample = next(dataset.to_batches(columns=[AUDIO_FIELD], limit=1))
    sample_audio = sample.column(AUDIO_FIELD).to_numpy_ndarray()
    sample_output = _encode_columns(sample_audio, sample_rate, specs, encoders)
    logger.info("inferred_embedding_schema", columns=[spec.column for spec in specs])

    progress_interval = max(
        config.batch_size, (total_rows + MAX_PROGRESS_LOGS - 1) // MAX_PROGRESS_LOGS
    )
    next_progress_row = progress_interval
    rows_processed = 0
    started_at = time.monotonic()
    last_progress_at = started_at
    last_udf_end = started_at
    stage_ms: dict[str, float] = {}

    @lance.batch_udf(
        output_schema=sample_output.schema,
        checkpoint_file=None if resume_cache is None else str(resume_cache),
    )
    def udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        nonlocal next_progress_row, rows_processed, last_progress_at, last_udf_end
        udf_started = time.monotonic()
        audio = batch.column(AUDIO_FIELD).to_numpy_ndarray()
        output = _encode_columns(audio, sample_rate, specs, encoders, stage_ms)
        rows_processed += batch.num_rows
        now = time.monotonic()
        interval_due = rows_processed >= next_progress_row or rows_processed == total_rows
        time_due = now - last_progress_at >= PROGRESS_LOG_INTERVAL_SECONDS
        if config.debug or interval_due or time_due:
            timings = {f"{name}_ms": round(duration, 1) for name, duration in stage_ms.items()}
            logger.info(
                "embedding_progress",
                rows_processed=rows_processed,
                total_rows=total_rows,
                percent=round(rows_processed / total_rows * 100, 1),
                rows_per_second=round(rows_processed / max(now - started_at, 1e-9), 1),
                batch_rows=batch.num_rows,
                batch_ms=round((now - udf_started) * 1000, 1),
                interbatch_ms=round((udf_started - last_udf_end) * 1000, 1),
                **timings,
            )
            last_progress_at = now
        if interval_due:
            next_progress_row = (rows_processed // progress_interval + 1) * progress_interval
        last_udf_end = time.monotonic()
        return output

    logger.info(
        "embedding_write_started",
        columns=[spec.column for spec in specs],
        total_rows=total_rows,
        batch_size=config.batch_size,
        source_version=dataset.version,
    )
    dataset.add_columns(udf, read_columns=[AUDIO_FIELD], batch_size=config.batch_size)
    _delete_resume_cache(resume_cache)
    logger.info(
        "wrote_embeddings",
        columns=[spec.column for spec in specs],
        total_rows=total_rows,
        committed_version=dataset.version,
    )
    encoders.clear()


def build_index(
    dataset: lance.LanceDataset,
    column: str,
    *,
    index: IndexSpec,
    config: AddEmbeddingsConfig,
) -> bool:
    """Build one declared IVF_PQ index when the dataset has enough rows.

    :param dataset: Dataset carrying the target fixed-size-list column.
    :param column: Vector column to index.
    :param index: Registry index defaults.
    :param config: Per-run index overrides.
    :returns: Whether an index was built.
    :raises ValueError: Index parameters are invalid for the target vector width.
    """
    num_sub_vectors = config.num_sub_vectors or index.num_sub_vectors
    metric = config.metric or index.metric
    if num_sub_vectors < 1:
        raise ValueError(f"num_sub_vectors must be >= 1, got {num_sub_vectors}")
    if config.num_partitions is not None and config.num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {config.num_partitions}")
    vector_dim = dataset.schema.field(column).type.list_size
    if vector_dim % num_sub_vectors != 0:
        raise ValueError(
            f"num_sub_vectors={num_sub_vectors} does not divide {column} dim {vector_dim}"
        )
    rows = dataset.count_rows()
    if rows < MIN_ROWS_FOR_INDEX:
        logger.warning(
            "embedding_index_skipped_too_few_rows",
            column=column,
            rows=rows,
            minimum=MIN_ROWS_FOR_INDEX,
        )
        return False
    partitions = (
        max(1, round(rows**0.5))
        if config.num_partitions is None
        else config.num_partitions
    )
    dataset.create_index(
        column,
        index_type="IVF_PQ",
        num_partitions=partitions,
        num_sub_vectors=num_sub_vectors,
        metric=metric,
    )
    logger.info(
        "embedding_index_built",
        column=column,
        rows=rows,
        num_partitions=partitions,
        metric=metric,
    )
    return True


def add_embeddings(config: AddEmbeddingsConfig) -> None:
    """Append the registry entries selected by ``config.embeddings``.

    :param config: Validated dataset, embedding, checkpoint, and write settings.
    """
    from synth_setter.pipeline.data.lance_shard import read_shard_metadata

    specs = [EMBEDDING_REGISTRY[name] for name in config.embeddings]
    dataset = _open_lance_dataset(config.lance_uri)
    sample_rate = int(read_shard_metadata(dataset.schema).sample_rate)
    _guard_existing_columns(dataset, specs)
    _validate_write_source(dataset, config.batch_size)
    _require_extras(specs)

    logger.info(
        "adding_embeddings",
        uri=config.lance_uri,
        columns=[spec.column for spec in specs],
        sample_rate=sample_rate,
        rows=dataset.count_rows(),
        batch_size=config.batch_size,
    )
    co_resident = [spec for spec in specs if spec.co_resident]
    solo = [spec for spec in specs if not spec.co_resident]
    if co_resident:
        _write_columns(dataset, co_resident, sample_rate, config)
    for spec in solo:
        _write_columns(dataset, [spec], sample_rate, config)

    if config.build_index:
        for spec in specs:
            if spec.index is not None:
                build_index(dataset, spec.column, index=spec.index, config=config)
    logger.info("added_embeddings", uri=config.lance_uri, columns=[spec.column for spec in specs])


def _configure_lance_logging(*, debug: bool) -> None:
    """Set native Lance logging before its first import.

    :param debug: Whether to force debug-level native telemetry.
    """
    if debug:
        os.environ["LANCE_LOG"] = "debug"
    else:
        os.environ.setdefault("LANCE_LOG", DEFAULT_LANCE_LOG)


def _resolve_torch_device(device: str | None) -> str:
    """Resolve an explicit device or prefer CUDA, MPS, then CPU.

    :param device: Explicit Torch device, or ``None``.
    :returns: Resolved Torch device.
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
    """Load music2latent and return an encoder over ``(B, C, T)`` audio.

    :param device: Torch device, or ``None`` for automatic selection.
    :returns: Encoder producing ``(B, C*D, T_lat)`` float32 latents.
    """
    from music2latent import EncoderDecoder

    resolved_device = _resolve_torch_device(device)
    logger.info("loading_m2l_checkpoint", device=resolved_device)
    encoder = EncoderDecoder(device=resolved_device)

    def encode(audio: np.ndarray) -> np.ndarray:
        batch, channels = audio.shape[:2]
        flat = np.ascontiguousarray(rearrange(audio, "b c t -> (b c) t"), dtype=np.float32)
        latents = encoder.encode(flat, max_batch_size=M2L_ENCODE_MAX_BATCH)
        latents = rearrange(latents, "(b c) d t -> b (c d) t", b=batch, c=channels)
        return latents.cpu().numpy()

    return encode


def load_clap_audio_encoder(
    checkpoint: str = DEFAULT_CLAP_CHECKPOINT,
    device: str | None = None,
) -> ClapEncodeFn:
    """Load CLAP and return an encoder over mono audio.

    :param checkpoint: HuggingFace CLAP model id.
    :param device: Torch device, or ``None`` for automatic selection.
    :returns: Encoder producing ``(B, CLAP_EMBEDDING_DIM)`` vectors.
    """
    import torch
    import torchaudio.functional as audio_fn
    from transformers import ClapModel, ClapProcessor

    resolved_device = _resolve_torch_device(device)
    logger.info(
        "loading_embedding_checkpoint",
        embedding="clap",
        checkpoint=checkpoint,
        device=resolved_device,
    )
    model = ClapModel.from_pretrained(checkpoint).to(resolved_device).eval()  # pyright: ignore
    processor = ClapProcessor.from_pretrained(checkpoint)

    @torch.no_grad()
    def _encode_chunk(chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        if sample_rate != CLAP_SAMPLE_RATE:
            wav = audio_fn.resample(wav, sample_rate, CLAP_SAMPLE_RATE)
        processor_kwargs = {
            "audio": list(wav.numpy()),
            "sampling_rate": CLAP_SAMPLE_RATE,
            "return_tensors": "pt",
        }
        inputs = processor(**processor_kwargs)
        device_inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
        features = model.get_audio_features(**device_inputs)
        return features.pooler_output.cpu().numpy()  # pyright: ignore

    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        chunks = [
            _encode_chunk(mono[start : start + CLAP_ENCODE_MAX_BATCH], sample_rate)
            for start in range(0, len(mono), CLAP_ENCODE_MAX_BATCH)
        ]
        return np.concatenate(chunks, axis=0)

    return encode


def _resolve_same_checkpoint_dir(checkpoint: str) -> Path:
    """Resolve a local, R2, or HuggingFace SAME checkpoint directory.

    :param checkpoint: Checkpoint directory, R2 prefix, or HuggingFace repo id.
    :returns: Local directory containing SAME model files.
    """
    if r2_io.is_r2_uri(checkpoint):
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
    """Load SAME and return an encoder over prepared stereo 44.1 kHz audio.

    :param checkpoint: Local directory, R2 mirror, or HuggingFace repo id.
    :param device: Torch device, or ``None`` for automatic selection.
    :returns: Encoder producing ``(B, SAME_EMBEDDING_DIM, T_lat)`` latents.
    :raises ImportError: The optional ``same`` extra is unavailable.
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
    resolved_device = _resolve_torch_device(device)
    logger.info("loading_same_checkpoint", checkpoint=checkpoint, device=resolved_device)
    model_config = json.loads((checkpoint_dir / "model_config.json").read_text())
    model = create_model_from_config(model_config)
    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"))
    model = model.to(resolved_device).eval().requires_grad_(False)

    @torch.no_grad()
    def _encode_chunk(chunk: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)).to(resolved_device)
        latents: torch.Tensor = model.encode(wav)  # pyright: ignore
        return latents.float().cpu().numpy()

    def encode(stereo: np.ndarray) -> np.ndarray:
        chunks = [
            _encode_chunk(stereo[start : start + SAME_ENCODE_MAX_BATCH])
            for start in range(0, len(stereo), SAME_ENCODE_MAX_BATCH)
        ]
        return np.concatenate(chunks, axis=0)

    return encode


def _open_lance_dataset(uri: str) -> lance.LanceDataset:
    """Open a local or credentialed R2 Lance dataset.

    :param uri: Local path, ``r2://`` URI, or R2-backed ``s3://`` URI.
    :returns: Open Lance dataset.
    """
    import lance

    if r2_io.is_r2_uri(uri):
        uri = r2_io.to_s3_uri(uri)
    if uri.startswith("s3://"):
        r2_io.ensure_r2_env_loaded()
        return lance.dataset(uri, storage_options=r2_io.r2_storage_options())
    return lance.dataset(uri)


@hydra.main(
    version_base="1.3", config_path="pkg://synth_setter.configs", config_name="add_embeddings"
)
def _hydra_main(cfg: DictConfig) -> None:
    """Validate Hydra config and run registry-selected embedding augmentation.

    :param cfg: Hydra-composed endpoint config.
    """
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

    config = AddEmbeddingsConfig.from_hydra_cfg(cfg)
    _configure_lance_logging(debug=config.debug)
    logger.info("lance_logging_configured", native_level=os.environ["LANCE_LOG"])
    try:
        add_embeddings(config)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.error("add_embeddings_failed", uri=config.lance_uri, error=str(exc))
        sys.exit(1)


def main() -> None:
    """Run the Hydra CLI while allowing keyed overrides on the empty checkpoint map."""
    for index, override in enumerate(sys.argv[1:], start=1):
        if override.startswith("checkpoints."):
            sys.argv[index] = f"+{override}"
    _hydra_main()


if __name__ == "__main__":
    main()
