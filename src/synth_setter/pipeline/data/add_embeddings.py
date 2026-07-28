#!/usr/bin/env python
"""Append registry-selected audio-embedding columns to a finalized Lance dataset.

The registry keeps checkpoint loading, Arrow encoding, residency, optional dependencies, and
index policy together for each embedding. Co-resident encoders share one Lance UDF pass; large
SAME encoders run in separate load-write-release passes.

CLI: ``synth-setter-add-embeddings dataset_root_uri=ROOT embeddings=[clap,m2l]``.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import hydra
import numpy as np
import pyarrow as pa
import structlog
from einops import rearrange

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    M2L_FIELD,
    PARAM_ARRAY_FIELD,
    SAME_L_FIELD,
    SAME_S_FIELD,
    T5GEMMA_FIELD,
    TINYMU_FIELD,
)
from synth_setter.model_cache import embedding_model_dir, synth_setter_cache_dir
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.constants import DATASET_CARD_FILENAME, DATASET_COMPLETE_FILENAME
from synth_setter.pipeline.data.tinymu import (
    DEFAULT_TINYMU_CHECKPOINT,
    TINYMU_CHECKPOINT_REVISION,
    TINYMU_CHECKPOINT_SHA256,
    TINYMU_FRONTEND,
    TINYMU_SOURCE_COMMIT,
    TinyMUEncodeFn,
    load_tinymu_audio_encoder,
    tinymu_num_latent_frames,
)
from synth_setter.pipeline.file_uri import file_uri_to_path, is_file_uri
from synth_setter.workspace import operator_workspace

if TYPE_CHECKING:
    import lance
    from omegaconf import DictConfig

    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig
    from synth_setter.pipeline.schemas.lance_attempt import (
        EmbeddingProvenance,
        EmbeddingSplitProvenance,
        LanceDatasetCard,
    )

logger = structlog.get_logger(__name__)
operator_workspace()

DEFAULT_CLAP_CHECKPOINT: str = "laion/clap-htsat-unfused"
DEFAULT_M2L_CHECKPOINT: str = ""
DEFAULT_SAME_S_CHECKPOINT: str = "r2://intermediate-data/models/same-s"
DEFAULT_SAME_L_CHECKPOINT: str = "r2://intermediate-data/models/same-l"
DEFAULT_T5GEMMA_CHECKPOINT: str = "r2://intermediate-data/models/sa3-small-music"
_DEFAULT_SAME_CACHE_NAMES: dict[str, str] = {
    DEFAULT_SAME_L_CHECKPOINT: "same-l",
    DEFAULT_SAME_S_CHECKPOINT: "same-s",
}
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
type DatasetSplit = Literal["train", "val", "test"]
CANONICAL_SPLITS: tuple[DatasetSplit, ...] = ("train", "val", "test")
SAME_EMBEDDING_DIM: int = 256
SAME_SAMPLE_RATE: int = 44100
SAME_DOWNSAMPLING_RATIO: int = 4096
SAME_S_PAD_BLOCK_SAMPLES: int = 2 * SAME_DOWNSAMPLING_RATIO
SAME_LATENT_FRAMES: int = 44
SAME_ENCODE_MAX_BATCH: int = 16

type M2LEncodeFn = Callable[[np.ndarray], np.ndarray]
type ClapEncodeFn = Callable[[np.ndarray, int], np.ndarray]
type SameEncodeFn = Callable[[np.ndarray], np.ndarray]
type SameFrameCountFn = Callable[[int, int], int]
type ParamTextEncodeFn = Callable[[np.ndarray], np.ndarray]
type Encoder = M2LEncodeFn | ClapEncodeFn | SameEncodeFn | ParamTextEncodeFn | TinyMUEncodeFn
type LoadEncoderFn = Callable[[str, AddEmbeddingsConfig], Encoder]
type EncodeColumnFn = Callable[[np.ndarray, int, Encoder], pa.Array]


@dataclass(frozen=True)
class IndexSpec:
    """Declare the default vector-index policy for one embedding column.

    .. attribute :: metric

        Lance distance metric.

    .. attribute :: num_sub_vectors

        PQ sub-vector count.

    .. attribute :: pool

        Pooling applied before indexing, or ``none`` for an existing vector.

    .. attribute :: vector_column

        Companion vector column, or ``None`` to index the embedding column.
    """

    metric: str = DEFAULT_INDEX_METRIC
    num_sub_vectors: int = DEFAULT_NUM_SUB_VECTORS
    pool: Literal["none", "mean", "attention"] = "none"
    vector_column: str | None = None


@dataclass(frozen=True)
class EmbeddingSpec:
    """Declare one selectable embedding's complete write policy.

    .. attribute :: name

        Registry key and config token.

    .. attribute :: column

        Lance column written by the encoder.

    .. attribute :: default_checkpoint

        Checkpoint source used without a keyed config override.

    .. attribute :: co_resident

        Whether the encoder may share a UDF pass with other selected encoders.

    .. attribute :: index

        Vector-index policy, or ``None`` when indexing is disabled for the embedding.

    .. attribute :: load_encoder

        Checkpoint and device to encoder factory.

    .. attribute :: encode_column

        Source batch, sample rate, and encoder to Arrow array transform.

    .. attribute :: input_field

        Dataset column supplying this embedding's encoder input.
    """

    name: str
    column: str
    default_checkpoint: str
    co_resident: bool
    index: IndexSpec | None
    load_encoder: LoadEncoderFn
    encode_column: EncodeColumnFn
    input_field: str = AUDIO_FIELD


def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Average ``(B, C, T)`` audio into ``(B, T)`` float32 mono.

    :param audio: Audio with one or more channels.
    :returns: Float32 mono audio.
    """
    return audio.mean(axis=1, dtype=np.float32)


def _fixed_size_list(vectors: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    """Pack ``(B, dim)`` vectors as a Lance-indexable Arrow array.

    :param vectors: Float-compatible vectors.
    :param dim: Fixed vector width.
    :returns: Fixed-size-list float32 array.
    """
    flat = pa.array(np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1), pa.float32())
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
    return _fixed_size_list(vectors, CLAP_EMBEDDING_DIM)


def _same_resampled_samples(num_samples: int, sample_rate: int) -> int:
    """Return the ceiling sample count after resampling to SAME's 44.1 kHz input rate.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Resampled clip length in samples.
    :raises ValueError: Either input is non-positive.
    """
    if num_samples < 1 or sample_rate < 1:
        raise ValueError(f"need positive num_samples/sample_rate, got {num_samples}/{sample_rate}")
    return math.ceil(num_samples * SAME_SAMPLE_RATE / sample_rate)


def same_s_num_latent_frames(num_samples: int, sample_rate: int) -> int:
    """Return SAME-S's even frame count after resampling and two-hop padding.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Two frames per complete or partial 8192-sample block.
    """
    resampled = _same_resampled_samples(num_samples, sample_rate)
    return 2 * math.ceil(resampled / SAME_S_PAD_BLOCK_SAMPLES)


def same_l_num_latent_frames(num_samples: int, sample_rate: int) -> int:
    """Return SAME-L's frame count after resampling to its 4096-sample hop.

    :param num_samples: Positive source clip length in samples.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: One frame per complete or partial 4096-sample block.
    """
    resampled = _same_resampled_samples(num_samples, sample_rate)
    return math.ceil(resampled / SAME_DOWNSAMPLING_RATIO)


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


def _encode_same_column(
    audio: np.ndarray,
    sample_rate: int,
    encoder: Encoder,
    *,
    field: str,
    frame_count: SameFrameCountFn,
) -> pa.Array:
    """Encode one audio batch under the selected SAME model's frame contract.

    :param audio: ``(B, C, T)`` source audio.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: SAME encoder over prepared stereo audio.
    :param field: SAME target column.
    :param frame_count: Model-specific latent-frame calculation.
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
        frame_count(prepared.shape[-1], SAME_SAMPLE_RATE),
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
    return _encode_same_column(
        audio,
        sample_rate,
        encoder,
        field=SAME_S_FIELD,
        frame_count=same_s_num_latent_frames,
    )


def _encode_same_l_column(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
    """Encode a SAME-L Arrow column through the shared SAME contract.

    :param audio: Source audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: SAME-L encoder.
    :returns: Fixed-shape tensor array.
    """
    return _encode_same_column(
        audio,
        sample_rate,
        encoder,
        field=SAME_L_FIELD,
        frame_count=same_l_num_latent_frames,
    )


def _load_m2l_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Load music2latent through the registry's uniform factory signature.

    :param checkpoint: Unused registry placeholder.
    :param config: Run config supplying the device.
    :returns: m2l encoder.
    """
    del checkpoint
    return load_m2l_audio_encoder(config.device)


def _load_clap_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Load CLAP through the registry's uniform factory signature.

    :param checkpoint: HuggingFace CLAP model id.
    :param config: Run config supplying the device.
    :returns: CLAP encoder.
    """
    return load_clap_audio_encoder(checkpoint, config.device)


def _load_same_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Load SAME through the registry's uniform factory signature.

    :param checkpoint: SAME checkpoint source.
    :param config: Run config supplying the device.
    :returns: SAME encoder.
    """
    return load_same_audio_encoder(checkpoint, config.device)


def _load_tinymu_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Load TinyMU through the pinned external-source adapter.

    :param checkpoint: Exact pinned URI or a hash-identical local artifact.
    :param config: Run config supplying the source checkout and device.
    :returns: Frozen MATPAC encoder.
    """
    return load_tinymu_audio_encoder(
        checkpoint,
        source_dir=config.tinymu_source_dir,
        device=_resolve_torch_device(config.device),
    )


def _load_t5gemma_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Bind a param spec and text normalizer to an SA3 T5Gemma text encoder.

    :param checkpoint: SA3 checkpoint source.
    :param config: Run config supplying the device, param spec, and normalizer.
    :returns: Encoder over encoded param rows.
    :raises ValueError: The config selects no param spec.
    """
    from synth_setter.data.vst.param_spec_registry import resolve_param_spec
    from synth_setter.data.vst.param_text import resolve_param_text_normalizer
    from synth_setter.param_spec_name import ParamSpecName
    from synth_setter.pipeline.data.t5gemma import load_t5gemma_text_encoder

    if config.param_spec_name is None:
        raise ValueError(f"{T5GEMMA_FIELD} embeddings require param_spec_name")
    spec = resolve_param_spec(ParamSpecName(config.param_spec_name))
    normalize = resolve_param_text_normalizer(config.param_text_normalizer)
    encode_text = load_t5gemma_text_encoder(checkpoint, _resolve_torch_device(config.device))

    def encode(params: np.ndarray) -> np.ndarray:
        if params.shape[-1] != spec.encoded_width:
            raise ValueError(
                f"param rows are {params.shape[-1]} wide but param spec "
                f"{config.param_spec_name!r} has encoded width {spec.encoded_width}"
            )
        return encode_text(normalize(spec, params))

    return encode


def _encode_tinymu_column(audio: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
    """Encode one audio batch as a fixed-shape TinyMU MATPAC tensor column.

    :param audio: ``(B, C, T)`` source audio.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: Frozen MATPAC encoder over source audio.
    :returns: ``(B, 3840, T_tokens)`` fixed-shape tensor array.
    :raises ValueError: The encoder returns the wrong shape or non-finite values.
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    encode = cast("TinyMUEncodeFn", encoder)
    embeddings = _finite_embedding(TINYMU_FIELD, encode(audio, sample_rate))
    expected_shape = (
        len(audio),
        TINYMU_FRONTEND.embedding_dim,
        tinymu_num_latent_frames(audio.shape[-1], sample_rate),
    )
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"{TINYMU_FIELD} encoder produced shape {embeddings.shape}, expected {expected_shape}"
        )
    return tensor_array(embeddings, np.dtype("float32"), expected_shape[1:])


def _encode_t5gemma_column(params: np.ndarray, sample_rate: int, encoder: Encoder) -> pa.Array:
    """Encode one param batch as a fixed-shape text-embedding tensor column.

    :param params: ``(B, encoded_width)`` param rows.
    :param sample_rate: Unused source sample rate.
    :param encoder: Encoder over param rows.
    :returns: Fixed-shape tensor array.
    :raises ValueError: The encoder returns the wrong row count, rank, or non-finite values.
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    del sample_rate
    encode = cast("ParamTextEncodeFn", encoder)
    embeddings = _finite_embedding(T5GEMMA_FIELD, encode(params))
    if embeddings.ndim != 3 or len(embeddings) != len(params):
        raise ValueError(
            f"{T5GEMMA_FIELD} encoder produced shape {embeddings.shape}, expected "
            f"{len(params)} rows of (dim, seq) embeddings"
        )
    return tensor_array(embeddings, np.dtype("float32"), embeddings.shape[1:])


EMBEDDING_REGISTRY: dict[str, EmbeddingSpec] = {
    "clap": EmbeddingSpec(
        name="clap",
        column=CLAP_FIELD,
        default_checkpoint=DEFAULT_CLAP_CHECKPOINT,
        co_resident=True,
        index=IndexSpec(pool="none"),
        load_encoder=_load_clap_spec_encoder,
        encode_column=_encode_clap_column,
    ),
    "m2l": EmbeddingSpec(
        name="m2l",
        column=M2L_FIELD,
        default_checkpoint=DEFAULT_M2L_CHECKPOINT,
        co_resident=True,
        index=IndexSpec(pool="mean", vector_column=f"{M2L_FIELD}_vec"),
        load_encoder=_load_m2l_spec_encoder,
        encode_column=_encode_m2l_column,
    ),
    "same_s": EmbeddingSpec(
        name="same_s",
        column=SAME_S_FIELD,
        default_checkpoint=DEFAULT_SAME_S_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(pool="mean", vector_column=f"{SAME_S_FIELD}_vec"),
        load_encoder=_load_same_spec_encoder,
        encode_column=_encode_same_s_column,
    ),
    "same_l": EmbeddingSpec(
        name="same_l",
        column=SAME_L_FIELD,
        default_checkpoint=DEFAULT_SAME_L_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(pool="mean", vector_column=f"{SAME_L_FIELD}_vec"),
        load_encoder=_load_same_spec_encoder,
        encode_column=_encode_same_l_column,
    ),
    # Rows share one caption per param spec today, so an index over identical
    # vectors would be degenerate; revisit when a values-aware normalizer lands.
    "t5gemma": EmbeddingSpec(
        name="t5gemma",
        column=T5GEMMA_FIELD,
        default_checkpoint=DEFAULT_T5GEMMA_CHECKPOINT,
        co_resident=False,
        index=None,
        load_encoder=_load_t5gemma_spec_encoder,
        encode_column=_encode_t5gemma_column,
        input_field=PARAM_ARRAY_FIELD,
    ),
    "tinymu": EmbeddingSpec(
        name="tinymu",
        column=TINYMU_FIELD,
        default_checkpoint=DEFAULT_TINYMU_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(pool="mean", vector_column=f"{TINYMU_FIELD}_vec"),
        load_encoder=_load_tinymu_spec_encoder,
        encode_column=_encode_tinymu_column,
    ),
}


def _output_columns(spec: EmbeddingSpec) -> tuple[str, ...]:
    """Return every dataset column emitted by one embedding policy.

    :param spec: Embedding write and index policy.
    :returns: Sequence column followed by its optional vector companion.
    """
    if spec.index is None or spec.index.vector_column is None:
        return (spec.column,)
    return spec.column, spec.index.vector_column


def _guard_existing_columns(
    dataset: lance.LanceDataset, specs: Sequence[EmbeddingSpec]
) -> None:
    """Reject selected columns already present in the dataset.

    :param dataset: Open Lance dataset.
    :param specs: Selected embedding policies.
    :raises ValueError: Any selected target column already exists.
    """
    target_columns = {column for spec in specs for column in _output_columns(spec)}
    existing = target_columns & set(dataset.schema.names)
    if existing:
        raise ValueError(f"dataset already has embedding column(s): {sorted(existing)}")


def _validate_write_source(
    dataset: lance.LanceDataset, batch_size: int, input_fields: Sequence[str] = (AUDIO_FIELD,)
) -> int:
    """Validate source-column and row-count preconditions for one UDF commit.

    :param dataset: Open Lance dataset.
    :param batch_size: Requested rows per UDF call.
    :param input_fields: Source columns the selected policies read.
    :returns: Positive source row count.
    :raises ValueError: A source column is absent, the dataset is empty, or batch size is non-
        positive.
    """
    for field in input_fields:
        if field not in dataset.schema.names:
            raise ValueError(f"dataset has no {field!r} column to embed")
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
        encoders.append(spec.load_encoder(checkpoint, config))
    return encoders


def _pooled_vector_column(
    array: pa.Array, index: IndexSpec
) -> tuple[str, pa.FixedSizeListArray] | None:
    """Build a companion vector from an encoded sequence when requested.

    :param array: Encoded Arrow embedding column.
    :param index: Pooling and target-column policy.
    :returns: Companion name and mean-pooled vectors, or ``None`` for an existing vector.
    :raises NotImplementedError: Attention pooling is selected.
    :raises ValueError: Mean pooling lacks a companion name or receives a non-sequence.
    """
    if index.pool == "none":
        return None
    if index.pool == "attention":
        raise NotImplementedError("attention pooling is not implemented")
    if index.vector_column is None:
        raise ValueError("mean pooling requires a companion vector_column")
    values = array.to_numpy_ndarray()
    if values.ndim != 3:
        raise ValueError(f"mean pooling requires (B, D, T) embeddings, got {values.shape}")
    pooled = values.mean(axis=-1, dtype=np.float32)
    return index.vector_column, _fixed_size_list(pooled, pooled.shape[1])


def _decoded_sources(
    batch: pa.RecordBatch, input_fields: Sequence[str]
) -> dict[str, np.ndarray]:
    """Decode each required source column of one batch into a numpy array.

    :param batch: Source batch supplied by Lance.
    :param input_fields: Column names the selected policies read.
    :returns: Decoded arrays keyed by field name.
    """
    return {field: batch.column(field).to_numpy_ndarray() for field in input_fields}


def _encode_columns(
    sources: Mapping[str, np.ndarray],
    sample_rate: int,
    specs: Sequence[EmbeddingSpec],
    encoders: Sequence[Encoder],
    stage_ms: dict[str, float] | None = None,
) -> pa.RecordBatch:
    """Encode one decoded source batch through every policy in a UDF pass.

    :param sources: Decoded input columns keyed by field name.
    :param sample_rate: Dataset sample rate in Hz.
    :param specs: Policies sharing this pass.
    :param encoders: Encoders aligned with ``specs``.
    :param stage_ms: Optional destination for per-encoder wall times.
    :returns: Record batch containing each selected embedding column.
    """
    columns: dict[str, pa.Array] = {}
    for spec, encoder in zip(specs, encoders, strict=True):
        started_at = time.monotonic()
        encoded = spec.encode_column(sources[spec.input_field], sample_rate, encoder)
        columns[spec.column] = encoded
        if spec.index is not None:
            pooled = _pooled_vector_column(encoded, spec.index)
            if pooled is not None:
                vector_column, vectors = pooled
                columns[vector_column] = vectors
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
    import torch

    if not specs:
        raise ValueError("no embedding specs given; nothing to write")
    _guard_existing_columns(dataset, specs)
    input_fields = sorted({spec.input_field for spec in specs})
    total_rows = _validate_write_source(dataset, config.batch_size, input_fields)
    # Model construction must not consume the seed governing stochastic encoders.
    with torch.random.fork_rng():
        encoders = _load_encoders(specs, config)
    resume_cache = _resume_cache_for_specs(config.resume_cache, config.embeddings, specs)
    output_columns = [column for spec in specs for column in _output_columns(spec)]

    logger.info("inferring_embedding_schema", columns=output_columns)
    sample = next(dataset.to_batches(columns=input_fields, limit=1))
    # Schema probing must not perturb stochastic encoders' persisted outputs.
    with torch.random.fork_rng():
        sample_output = _encode_columns(
            _decoded_sources(sample, input_fields), sample_rate, specs, encoders
        )
    logger.info("inferred_embedding_schema", columns=output_columns)

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
        sources = _decoded_sources(batch, input_fields)
        output = _encode_columns(sources, sample_rate, specs, encoders, stage_ms)
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
        columns=output_columns,
        total_rows=total_rows,
        batch_size=config.batch_size,
        source_version=dataset.version,
    )
    dataset.add_columns(udf, read_columns=input_fields, batch_size=config.batch_size)
    _delete_resume_cache(resume_cache)
    logger.info(
        "wrote_embeddings",
        columns=output_columns,
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


def _add_embeddings_to_lance_uri(
    config: AddEmbeddingsConfig, uri: str
) -> tuple[lance.LanceDataset, dict[str, bool]]:
    """Append selected embeddings to one Lance split.

    :param config: Validated embedding, checkpoint, and write settings.
    :param uri: Local or remote Lance split URI.
    :returns: Updated dataset and index-built status by registry key.
    """
    from synth_setter.pipeline.data.lance_shard import read_shard_metadata

    specs = [EMBEDDING_REGISTRY[name] for name in config.embeddings]
    dataset = _open_lance_dataset(uri)
    sample_rate = int(read_shard_metadata(dataset.schema).sample_rate)
    _guard_existing_columns(dataset, specs)
    _validate_write_source(dataset, config.batch_size)
    output_columns = [column for spec in specs for column in _output_columns(spec)]

    logger.info(
        "adding_embeddings",
        uri=uri,
        columns=output_columns,
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

    index_results = {spec.name: False for spec in specs}
    if config.build_index:
        for spec in specs:
            if spec.index is not None:
                vector_column = spec.index.vector_column or spec.column
                index_results[spec.name] = build_index(
                    dataset, vector_column, index=spec.index, config=config
                )
    logger.info("added_embeddings", uri=uri, columns=output_columns)
    return dataset, index_results


def _dataset_root_child(root: str, name: str) -> str:
    """Join one dataset-root child without changing its URI scheme.

    :param root: Local path or dataset-root URI.
    :param name: Child name.
    :returns: Child path or URI.
    """
    from synth_setter.pipeline.spec_io import join_uri

    return join_uri(root, name)


def _local_uri_path(uri: str) -> Path | None:
    """Resolve local paths and ``file://`` URIs, leaving remote URIs unresolved.

    :param uri: Candidate path or URI.
    :returns: Local path, or ``None`` for R2/S3.
    """
    if r2_io.is_r2_uri(uri) or uri.startswith("s3://"):
        return None
    return file_uri_to_path(uri) if is_file_uri(uri) else Path(uri)


def _root_child_exists(uri: str) -> bool:
    """Return whether one local or remote dataset-root child exists.

    :param uri: Child path or URI.
    :returns: Whether a local path or remote directory exists.
    """
    local = _local_uri_path(uri)
    if local is not None:
        return local.exists()
    remote = r2_io.from_s3_uri(uri) if uri.startswith("s3://") else uri
    return r2_io.r2_directory_exists(remote)


def _read_dataset_card(root: str) -> LanceDatasetCard:
    """Read the strict dataset card from a finalized root.

    :param root: Local or remote dataset root.
    :returns: Parsed dataset card.
    """
    from synth_setter.pipeline.schemas.lance_attempt import LanceDatasetCard
    from synth_setter.pipeline.spec_io import localized_uri

    card_uri = _dataset_root_child(root, DATASET_CARD_FILENAME)
    with localized_uri(card_uri) as card_path:
        return LanceDatasetCard.model_validate_json(card_path.read_bytes())


def _write_dataset_card(root: str, card: LanceDatasetCard) -> None:
    """Atomically replace a local card or upload a complete remote card.

    :param root: Local or remote dataset root.
    :param card: Strict Pydantic card with ``model_dump_json``.
    """
    card_uri = _dataset_root_child(root, DATASET_CARD_FILENAME)
    local = _local_uri_path(card_uri)
    if local is not None:
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_name(f".{local.name}.tmp")
        temporary.write_text(card.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, local)
        return
    remote = r2_io.from_s3_uri(card_uri) if card_uri.startswith("s3://") else card_uri
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / DATASET_CARD_FILENAME
        temporary.write_text(card.model_dump_json(indent=2), encoding="utf-8")
        r2_io.upload_to_uri(temporary, remote)


@cache
def _producer_identity() -> tuple[str, str]:
    """Resolve the checkout commit and hash the MATPAC transform implementation.

    :returns: Producer Git SHA and transform-module SHA-256.
    :raises ValueError: The installed source cannot be tied to its owning checkout.
    """
    workspace = operator_workspace()
    source = Path(__file__).resolve()
    if not source.is_relative_to(workspace):
        raise ValueError(f"embedding producer source {source} is outside workspace {workspace}")
    try:
        git_sha = subprocess.check_output(  # noqa: S603 — fixed checkout identity probe
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ValueError(f"cannot resolve embedding producer Git SHA from {workspace}") from exc
    digest = hashlib.sha256()
    for module_path in (source, source.with_name("tinymu.py")):
        digest.update(module_path.name.encode())
        digest.update(b"\0")
        digest.update(module_path.read_bytes())
    return git_sha, digest.hexdigest()


def _embedding_provenance(
    name: str,
    config: AddEmbeddingsConfig,
    splits: tuple[EmbeddingSplitProvenance, ...],
) -> EmbeddingProvenance:
    """Build persisted output identity for one selected registry entry.

    :param name: Registry key.
    :param config: Output-defining augmentation settings.
    :param splits: Intended or committed split provenance entries.
    :returns: Strict embedding provenance.
    """
    from synth_setter.pipeline.schemas.lance_attempt import EmbeddingProvenance

    spec = EMBEDDING_REGISTRY[name]
    tinymu = name == "tinymu"
    producer_git_sha, producer_transform_sha256 = _producer_identity()
    param_sourced = spec.input_field == PARAM_ARRAY_FIELD
    indexed = spec.index is not None
    return EmbeddingProvenance(
        name=name,
        columns=_output_columns(spec),
        checkpoint=config.checkpoints.get(name, spec.default_checkpoint),
        producer_git_sha=producer_git_sha,
        producer_transform_sha256=producer_transform_sha256,
        source_commit=TINYMU_SOURCE_COMMIT if tinymu else None,
        checkpoint_revision=TINYMU_CHECKPOINT_REVISION if tinymu else None,
        checkpoint_sha256=TINYMU_CHECKPOINT_SHA256 if tinymu else None,
        param_spec_name=config.param_spec_name if param_sourced else None,
        param_text_normalizer=config.param_text_normalizer if param_sourced else None,
        index_requested=config.build_index if indexed else None,
        num_partitions=config.num_partitions if indexed else None,
        num_sub_vectors=config.num_sub_vectors if indexed else None,
        metric=config.metric if indexed else None,
        splits=splits,
    )


def _replace_embedding_split(
    provenance: EmbeddingProvenance, result: EmbeddingSplitProvenance
) -> EmbeddingProvenance:
    """Replace one canonical split result under an embedding identity.

    :param provenance: Existing embedding identity and split records.
    :param result: New state for one split.
    :returns: Revalidated provenance with exactly one record for the split.
    """
    from synth_setter.pipeline.schemas.lance_attempt import EmbeddingProvenance

    splits = tuple(item for item in provenance.splits if item.split != result.split) + (result,)
    return EmbeddingProvenance.model_validate({**provenance.model_dump(), "splits": splits})


def _replace_card_embeddings(
    card: LanceDatasetCard, embeddings: Mapping[str, EmbeddingProvenance]
) -> LanceDatasetCard:
    """Revalidate a card after replacing its embedding records.

    :param card: Existing v1 or v2 dataset card.
    :param embeddings: Provenance keyed by unique registry name.
    :returns: Schema-v2 card preserving finalize-owned fields.
    """
    from synth_setter.pipeline.schemas.lance_attempt import LanceDatasetCard

    return LanceDatasetCard.model_validate(
        {**card.model_dump(), "schema_version": 2, "embeddings": tuple(embeddings.values())}
    )


def _completion_marker_exists(root: str) -> bool:
    """Return whether the dataset root currently advertises readiness.

    :param root: Local or remote dataset root.
    :returns: Whether ``dataset.complete`` exists.
    """
    marker_uri = _dataset_root_child(root, DATASET_COMPLETE_FILENAME)
    local = _local_uri_path(marker_uri)
    if local is not None:
        return local.is_file()
    remote = r2_io.from_s3_uri(marker_uri) if marker_uri.startswith("s3://") else marker_uri
    return r2_io.object_size(remote) is not None


def _remove_completion_marker(root: str) -> None:
    """Remove readiness before the first split mutation.

    :param root: Local or remote dataset root.
    """
    marker_uri = _dataset_root_child(root, DATASET_COMPLETE_FILENAME)
    local = _local_uri_path(marker_uri)
    if local is not None:
        local.unlink(missing_ok=True)
        return
    remote = r2_io.from_s3_uri(marker_uri) if marker_uri.startswith("s3://") else marker_uri
    r2_io.delete_object(remote)


def _write_completion_marker(root: str) -> None:
    """Publish readiness after every intended embedding operation completes.

    :param root: Local or remote dataset root.
    """
    marker_uri = _dataset_root_child(root, DATASET_COMPLETE_FILENAME)
    local = _local_uri_path(marker_uri)
    if local is not None:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch()
        return
    remote = r2_io.from_s3_uri(marker_uri) if marker_uri.startswith("s3://") else marker_uri
    with tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / DATASET_COMPLETE_FILENAME
        marker.touch()
        r2_io.upload_to_uri(marker, remote)


def _index_exists(dataset: lance.LanceDataset, spec: EmbeddingSpec) -> bool:
    """Return whether the registry policy's target column has an index.

    :param dataset: Open split dataset.
    :param spec: Registry policy whose target is checked.
    :returns: Whether Lance reports an index over the target column.
    """
    if spec.index is None:
        return False
    target = spec.index.vector_column or spec.column
    indices = cast("list[dict[str, object]]", dataset.list_indices())
    return any(entry.get("fields") == [target] for entry in indices)


def _complete_root_embedding(
    config: AddEmbeddingsConfig,
    uri: str,
    name: str,
) -> tuple[lance.LanceDataset, bool]:
    """Write or resume one embedding policy on one split.

    :param config: Output and index settings.
    :param uri: Split URI.
    :param name: Registry key.
    :returns: Updated dataset and whether its declared index exists.
    :raises ValueError: Only a subset of the policy's columns is present.
    """
    spec = EMBEDDING_REGISTRY[name]
    dataset = _open_lance_dataset(uri)
    expected = set(_output_columns(spec))
    present = expected & set(dataset.schema.names)
    if present and present != expected:
        raise ValueError(f"split {uri} has partial {name} columns: {sorted(present)}")
    if not present:
        split_config = config.model_copy(update={"embeddings": (name,)})
        dataset, indexes = _add_embeddings_to_lance_uri(split_config, uri)
        return dataset, indexes[name]
    if not config.build_index or spec.index is None:
        return dataset, False
    if _index_exists(dataset, spec):
        return dataset, True
    target = spec.index.vector_column or spec.column
    return dataset, build_index(dataset, target, index=spec.index, config=config)


def _incomplete_embedding_work(
    embeddings: Mapping[str, EmbeddingProvenance],
) -> list[str]:
    """Name every embedding split that has not committed.

    :param embeddings: Persisted provenance keyed by registry name.
    :returns: ``name:split`` identifiers for incomplete work.
    """
    return [
        f"{entry.name}:{result.split}"
        for entry in embeddings.values()
        for result in entry.splits
        if not result.complete
    ]


def _add_embeddings_to_dataset_root(config: AddEmbeddingsConfig, root: str) -> None:
    """Augment every canonical split under a resumable readiness protocol.

    :param config: Validated root augmentation settings.
    :param root: Finalized dataset root.
    :raises FileNotFoundError: The root has no dataset card or no Lance splits.
    :raises ValueError: Finalization is incomplete or persisted state conflicts with the request.
    """
    from synth_setter.pipeline.schemas.lance_attempt import EmbeddingSplitProvenance

    if r2_io.is_r2_uri(root) or root.startswith("s3://"):
        r2_io.ensure_r2_env_loaded()
    card = _read_dataset_card(root)
    existing = {entry.name: entry for entry in card.embeddings}
    for name in config.embeddings:
        prior = existing.get(name)
        if prior is not None and prior != _embedding_provenance(name, config, prior.splits):
            raise ValueError(f"stored {name} provenance does not match this augmentation config")

    split_uris: list[tuple[DatasetSplit, str]] = [
        (split, uri)
        for split in CANONICAL_SPLITS
        if _root_child_exists(uri := _dataset_root_child(root, f"{split}.lance"))
    ]
    if not split_uris:
        raise FileNotFoundError(f"dataset root has no train/val/test Lance splits: {root}")

    marker_exists = _completion_marker_exists(root)
    resumable = any(not result.complete for entry in existing.values() for result in entry.splits)
    if not marker_exists and not resumable and card.schema_version == 1:
        raise ValueError(f"dataset root is not finalized; missing {DATASET_COMPLETE_FILENAME}: {root}")

    pending: list[tuple[DatasetSplit, str, str]] = []
    for split, uri in split_uris:
        dataset = _open_lance_dataset(uri)
        for name in config.embeddings:
            prior = existing.get(name)
            result = None if prior is None else next(
                (item for item in prior.splits if item.split == split), None
            )
            if result is not None and result.complete:
                expected = set(_output_columns(EMBEDDING_REGISTRY[name]))
                if not expected <= set(dataset.schema.names):
                    raise ValueError(f"stored {name} provenance disagrees with split {uri}")
                continue
            if result is None:
                expected = set(_output_columns(EMBEDDING_REGISTRY[name]))
                present = expected & set(dataset.schema.names)
                if present:
                    raise ValueError(f"split {uri} has untracked {name} columns: {sorted(present)}")
                result = EmbeddingSplitProvenance(
                    split=split,
                    dataset_version=dataset.version,
                    row_count=dataset.count_rows(),
                    index_built=False,
                    complete=False,
                )
                identity = prior or _embedding_provenance(name, config, ())
                existing[name] = _replace_embedding_split(identity, result)
            pending.append((split, uri, name))

    if not pending:
        incomplete = _incomplete_embedding_work(existing)
        if incomplete:
            raise ValueError(f"dataset root still has incomplete embedding work: {incomplete}")
        if not marker_exists:
            _write_completion_marker(root)
        return

    card = _replace_card_embeddings(card, existing)
    _write_dataset_card(root, card)
    if marker_exists:
        _remove_completion_marker(root)

    for split, uri, name in pending:
        prior = existing[name]
        pending_result = next(item for item in prior.splits if item.split == split)
        dataset, index_built = _complete_root_embedding(config, uri, name)
        if dataset.count_rows() != pending_result.row_count:
            raise ValueError(
                f"split {uri} row count changed from {pending_result.row_count} "
                f"to {dataset.count_rows()}"
            )
        complete = EmbeddingSplitProvenance(
            split=split,
            dataset_version=dataset.version,
            row_count=dataset.count_rows(),
            index_built=index_built,
            complete=True,
        )
        existing[name] = _replace_embedding_split(existing[name], complete)
        card = _replace_card_embeddings(card, existing)
        _write_dataset_card(root, card)

    incomplete = _incomplete_embedding_work(existing)
    if incomplete:
        raise ValueError(f"dataset root still has incomplete embedding work: {incomplete}")
    _write_completion_marker(root)


def add_embeddings(config: AddEmbeddingsConfig) -> None:
    """Append registry entries to one Lance split or every split in a dataset root.

    :param config: Validated dataset, embedding, checkpoint, and write settings.
    """
    if config.dataset_root_uri is not None:
        _add_embeddings_to_dataset_root(config, config.dataset_root_uri)
        return
    assert config.lance_uri is not None
    _add_embeddings_to_lance_uri(config, config.lance_uri)


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


def _resolve_clap_checkpoint(checkpoint: str) -> str:
    """Resolve a local or HuggingFace CLAP checkpoint directory.

    :param checkpoint: Local directory or HuggingFace model id.
    :returns: Local directory accepted by the Transformers loaders.
    """
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        return str(local)

    from huggingface_hub import snapshot_download

    if checkpoint == DEFAULT_CLAP_CHECKPOINT:
        cache_dir = embedding_model_dir("clap-htsat-unfused")
        return snapshot_download(checkpoint, local_dir=str(cache_dir))
    return snapshot_download(checkpoint)


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
    checkpoint_dir = _resolve_clap_checkpoint(checkpoint)
    model = ClapModel.from_pretrained(checkpoint_dir).to(resolved_device).eval()  # pyright: ignore
    processor = ClapProcessor.from_pretrained(checkpoint_dir)

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
        cache_name = _DEFAULT_SAME_CACHE_NAMES.get(checkpoint)
        if cache_name is None:
            cache_key = checkpoint.removeprefix("r2://").strip("/")
            cache_dir = synth_setter_cache_dir() / "models" / cache_key
        else:
            cache_dir = embedding_model_dir(cache_name)
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
    """
    import json

    import torch
    from safetensors.torch import load_file
    from stable_audio_3.factory import create_autoencoder_from_config

    checkpoint_dir = _resolve_same_checkpoint_dir(checkpoint)
    resolved_device = _resolve_torch_device(device)
    logger.info("loading_same_checkpoint", checkpoint=checkpoint, device=resolved_device)
    model_config = json.loads((checkpoint_dir / "model_config.json").read_text())
    model = create_autoencoder_from_config(model_config["model"], model_config["sample_rate"])
    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"), strict=True)
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
        target = config.dataset_root_uri or config.lance_uri
        logger.error("add_embeddings_failed", uri=target, error=str(exc))
        sys.exit(1)


def main() -> None:
    """Run the Hydra CLI while allowing keyed overrides on the empty checkpoint map."""
    for index, override in enumerate(sys.argv[1:], start=1):
        if override.startswith("checkpoints."):
            sys.argv[index] = f"+{override}"
    _hydra_main()


if __name__ == "__main__":
    main()
