#!/usr/bin/env python
"""Append registry-selected audio-embedding columns to a finalized Lance dataset.

The registry keeps checkpoint loading, Arrow encoding, residency, optional dependencies, and
index policy together for each embedding. Co-resident encoders share one Lance UDF pass; large
SAME encoders run in separate load-write-release passes.

CLI: ``synth-setter-add-embeddings lance_uri=DATASET embeddings=[clap,m2l,matpac_plus]``.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import hydra
import numpy as np
import pyarrow as pa
import structlog
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped

from synth_setter.clap import DEFAULT_CLAP_CHECKPOINT, resolve_clap_checkpoint
from synth_setter.conditioning import (
    PYFDN_SKETCH_CONTROLS,
    PYFDN_SKETCH_STRUCT_FIELD,
    SKETCH_STORAGE_FRAMES,
)
from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    DEFAULT_PESTO_CHECKPOINT,
    M2L_FIELD,
    MATPAC_PLUS_FIELD,
    MEANAUDIO_16K_FIELD,
    NUM_SKETCH_CONTROLS,
    PARAM_ARRAY_FIELD,
    SAME_L_FIELD,
    SAME_S_FIELD,
    SHIFT_FIELD,
    SKETCH_PITCH_BINS,
    SKETCH_STRUCT_FIELD,
    SKETCH_VEC_CHILD,
    SSONDO_FIELD,
    T5GEMMA_FIELD,
    PUPUJEPA_LARGE_FIELD,
    PUPUJEPA_TINY_FIELD,
    mel_n_frames_from_samples,
)
from synth_setter.model_cache import checkpoint_tree_sha256
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.matpac_plus import (
    DEFAULT_MATPAC_PLUS_CHECKPOINT,
    MATPAC_PLUS_FRONTEND,
    encode_matpac_plus_column,
    load_matpac_plus_audio_encoder,
    matpac_plus_artifact_digest,
)
from synth_setter.pipeline.data.meanaudio import (
    DEFAULT_MEANAUDIO_CHECKPOINT,
    MEANAUDIO_EMBEDDING_DIM,
    MEANAUDIO_INDEX_SUB_VECTORS,
    encode_meanaudio_column,
    load_meanaudio_audio_encoder,
    meanaudio_artifact_digest,
)
from synth_setter.pipeline.data.pupujepa import (
    PupuJepaEncodeFn,
    encode_pupujepa_column,
    encode_pupujepa_large_column,
    load_pupujepa_audio_encoder,
)
from synth_setter.pupujepa import (
    DEFAULT_PUPUJEPA_CHECKPOINT,
    PUPUJEPA_LARGE_EMBEDDING_DIM,
    PUPUJEPA_TINY_EMBEDDING_DIM,
    pupujepa_artifact_digest,
)
from synth_setter.pipeline.data.param_shift import (
    PARAM_SHIFT_INPUT_FIELDS,
    ROW_ID_FIELD,
    ParamShifter,
    encode_param_shift_column,
    load_param_shifter,
    param_shift_policy_values,
)
from synth_setter.pipeline.data.ssondo import (
    DEFAULT_SSONDO_CHECKPOINT,
    SSONDO_CHECKPOINT_SHA256,
    SSONDO_EMBEDDING_DIM,
    SSONDO_SOURCE_REVISION,
    SSONDOEncodeFn,
    load_ssondo_audio_encoder,
    resolve_ssondo_checkpoint,
)
from synth_setter.same import (
    DEFAULT_SAME_L_CHECKPOINT,
    DEFAULT_SAME_S_CHECKPOINT,
    SAME_EMBEDDING_DIM,
    SAME_SAMPLE_RATE,
    load_same_autoencoder,
    resolve_same_checkpoint,
    same_l_num_latent_frames,
    same_s_num_latent_frames,
)
from synth_setter.workspace import operator_workspace

if TYPE_CHECKING:
    import lance
    from omegaconf import DictConfig

    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

logger = structlog.get_logger(__name__)
operator_workspace()

DEFAULT_M2L_CHECKPOINT: str = ""
DEFAULT_T5GEMMA_CHECKPOINT: str = "r2://intermediate-data/models/sa3-small-music"
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
_EMBEDDING_NAME_METADATA = b"synth_setter.embedding.name"
_EMBEDDING_ARTIFACT_METADATA = b"synth_setter.embedding.artifact"
SAME_LATENT_FRAMES: int = 44
SAME_ENCODE_MAX_BATCH: int = 16
SKETCH_INDEX_SUB_VECTORS: int = 2
# PESTO's per-clip intermediates scale with batch size: a full 128-row Lance
# batch peaked at ~8.8 GiB RSS and drew earlyoom SIGTERMs in the field (#2707).
SKETCH_ENCODE_MAX_BATCH: int = 32
# Dotted path of the nested IVF companion inside the sketch struct (#2707).
# Whole-struct add_columns append works on storage 2.1 and 2.2 datasets;
# per-child schema evolution (unused here) is the 2.2-only operation.
SKETCH_VEC_COLUMN: str = f"{SKETCH_STRUCT_FIELD}.{SKETCH_VEC_CHILD}"
PYFDN_SKETCH_POLICY_VERSION = 2

type M2LEncodeFn = Callable[[np.ndarray], np.ndarray]
type ClapEncodeFn = Callable[[np.ndarray, int], np.ndarray]
# Structurally = ClapEncodeFn; the name documents the (B, C, T) input contract.
type SketchEncodeFn = Callable[[np.ndarray, int], np.ndarray]
type SameEncodeFn = Callable[[np.ndarray], np.ndarray]
type SameFrameCountFn = Callable[[int, int], int]
type ParamTextEncodeFn = Callable[[np.ndarray], np.ndarray]
type Encoder = (
    M2LEncodeFn
    | ClapEncodeFn
    | SameEncodeFn
    | SSONDOEncodeFn
    | PupuJepaEncodeFn
    | ParamTextEncodeFn
    | ParamShifter
)
type LoadEncoderFn = Callable[[str, AddEmbeddingsConfig], Encoder]
type EncodeColumnFn = Callable[[Mapping[str, np.ndarray], int, Encoder], pa.Array]
type ResolveArtifactIdentityFn = Callable[[str], str]


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

    .. attribute :: vector_dim

        Static vector width for config validation, or ``None`` when output-derived.
    """

    metric: str = DEFAULT_INDEX_METRIC
    num_sub_vectors: int = DEFAULT_NUM_SUB_VECTORS
    pool: Literal["none", "mean", "attention"] = "none"
    vector_column: str | None = None
    vector_dim: int | None = None


@dataclass(frozen=True)
class EmbeddingSpec:
    """Declare one selectable embedding's complete write policy.

    .. attribute :: name

        Registry key and config token.

    .. attribute :: column

        Lance column carrying the embedding and targeted by the index policy.

    .. attribute :: default_checkpoint

        Checkpoint source used without a keyed config override.

    .. attribute :: co_resident

        Whether the encoder may share a UDF pass with other selected encoders.

    .. attribute :: index

        Vector-index policy, or ``None`` when indexing is disabled for the embedding.

    .. attribute :: load_encoder

        Checkpoint and device to encoder factory.

    .. attribute :: encode_column

        Decoded source columns, sample rate, and encoder to one Arrow column.

    .. attribute :: resolve_artifact_identity

        Checkpoint source to immutable encoder-artifact identity resolver.

    .. attribute :: input_fields

        Dataset columns supplying this embedding's encoder input.

    .. attribute :: rerenders

        Whether the encoder re-renders audio, making the run's render config and seed
        part of its output identity.
    """

    name: str
    column: str
    default_checkpoint: str
    co_resident: bool
    index: IndexSpec | None
    load_encoder: LoadEncoderFn
    encode_column: EncodeColumnFn
    resolve_artifact_identity: ResolveArtifactIdentityFn
    input_fields: tuple[str, ...] = (AUDIO_FIELD,)
    rerenders: bool = False


EMBEDDING_POLICY_VERSION = 1


class _Digest(Protocol):
    """Structural hash state used by artifact identity framing."""

    def update(self, value: bytes, /) -> None:
        """Append bytes to the hash state.

        :param value: Bytes to append.
        """
        ...


def _update_framed_digest(digest: _Digest, value: bytes) -> None:
    """Append one length-delimited value to a digest.

    :param digest: Hash state to update.
    :param value: Bytes whose boundary must be preserved.
    """
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _versioned_artifact_identity(name: str, digest: str) -> str:
    """Bind checkpoint identity to synth-setter's preprocessing contract.

    :param name: Embedding registry key.
    :param digest: Immutable package or checkpoint digest.
    :returns: Versioned identity persisted in Lance field metadata.
    """
    return f"{name}:policy-v{EMBEDDING_POLICY_VERSION}:{digest}"


def _m2l_artifact_identity(checkpoint: str) -> str:
    """Resolve the package-owned music2latent artifact identity.

    :param checkpoint: Empty placeholder; caller overrides are unsupported.
    :returns: Versioned installed-package identity.
    :raises ValueError: A checkpoint override bypasses package-owned weights.
    """
    if checkpoint:
        raise ValueError("music2latent does not support checkpoint overrides")
    version = importlib.metadata.version("music2latent")
    return _versioned_artifact_identity("m2l", f"package:{version}")


def _clap_artifact_identity(checkpoint: str) -> str:
    """Resolve and hash one CLAP checkpoint tree.

    :param checkpoint: Local directory or HuggingFace model id.
    :returns: Versioned content identity.
    """
    checkpoint_dir = Path(_resolve_clap_checkpoint(checkpoint))
    return _versioned_artifact_identity("clap", checkpoint_tree_sha256(checkpoint_dir))


def _same_artifact_identity(checkpoint: str) -> str:
    """Resolve and hash one SAME checkpoint tree.

    :param checkpoint: Local, R2, or HuggingFace checkpoint source.
    :returns: Versioned content identity.
    """
    checkpoint_dir = resolve_same_checkpoint(checkpoint)
    return _versioned_artifact_identity("same", checkpoint_tree_sha256(checkpoint_dir))


def _t5gemma_artifact_identity(checkpoint: str) -> str:
    """Resolve and hash one T5Gemma checkpoint tree.

    :param checkpoint: Local, R2, or HuggingFace checkpoint source.
    :returns: Versioned content identity.
    """
    from synth_setter.pipeline.data.t5gemma import _resolve_t5gemma_checkpoint_dir

    checkpoint_dir = _resolve_t5gemma_checkpoint_dir(checkpoint)
    return _versioned_artifact_identity("t5gemma", checkpoint_tree_sha256(checkpoint_dir))


def _sketch_artifact_identity(checkpoint: str) -> str:
    """Identify the pesto-bundled sketch extraction artifact.

    :param checkpoint: Bundled PESTO checkpoint name shipping with the package.
    :returns: Versioned installed-package and checkpoint identity.
    """
    version = importlib.metadata.version("pesto-pitch")
    identity = (
        f"package:{version};checkpoint:{checkpoint};"
        f"storage:avgmax{SKETCH_STORAGE_FRAMES}"
    )
    return _versioned_artifact_identity("sketch", identity)


def _pyfdn_sketch_artifact_identity(checkpoint: str) -> str:
    """Identify the checkpoint-free pyFDN temporal-sketch extraction policy.

    :param checkpoint: Empty placeholder; the extractor has no learned weights.
    :returns: Versioned DSP, normalization, temporal-bin, and package identity.
    :raises ValueError: A checkpoint override was supplied.
    """
    if checkpoint:
        raise ValueError("pyfdn_sketch is checkpoint-free and rejects checkpoint overrides")
    packages = ",".join(
        f"{name}:{importlib.metadata.version(name)}" for name in ("numpy", "pyfdn", "scipy")
    )
    policy = (
        f"dsp:octave-edc-abel-huang-density-stft-flatness-v{PYFDN_SKETCH_POLICY_VERSION};"
        "normalization:signed-unit-edc-floor-60db-density-rational-flatness-linear;"
        "temporal:fractional-log-32-head-0.005-ratio-200-hann-1024-hop-128-frame-center;"
        f"packages:{packages}"
    )
    return _versioned_artifact_identity("pyfdn_sketch", policy)


def _ssondo_artifact_identity(checkpoint: str) -> str:
    """Verify and identify the pinned S-SONDO package/checkpoint pair.

    :param checkpoint: Pinned Hugging Face repo or SHA-identical local checkpoint.
    :returns: Versioned package and checkpoint identity.
    """
    resolve_ssondo_checkpoint(checkpoint)
    package_version = importlib.metadata.version("ssondo")
    digest = (
        f"package:{package_version};source:{SSONDO_SOURCE_REVISION};"
        f"checkpoint:sha256:{SSONDO_CHECKPOINT_SHA256}"
    )
    return _versioned_artifact_identity("ssondo", digest)


def _param_shift_artifact_identity(checkpoint: str) -> str:
    """Return the re-render embedder's identity stem.

    The output-determining settings are the render config and seed, which
    :func:`_resolve_artifact_identity` folds in through ``EmbeddingSpec.rerenders``.

    :param checkpoint: Empty placeholder; the embedder loads no checkpoint.
    :returns: Versioned identity stem.
    :raises ValueError: A checkpoint override was supplied for a checkpoint-free embedder.
    """
    if checkpoint:
        raise ValueError("param_shift renders through a render config, not a checkpoint")
    return _versioned_artifact_identity("param_shift", "renderer")


def _matpac_plus_artifact_identity(checkpoint: str) -> str:
    """Return the MATPAC++ package/checkpoint artifact identity.

    :param checkpoint: Pinned R2 URI or SHA-identical local checkpoint.
    :returns: Versioned package and checkpoint identity.
    """
    return _versioned_artifact_identity("matpac_plus", matpac_plus_artifact_digest(checkpoint))


def _meanaudio_artifact_identity(checkpoint: str) -> str:
    """Return the MeanAudio package/checkpoint artifact identity.

    :param checkpoint: Pinned Hugging Face repo or SHA-identical local checkpoint.
    :returns: Versioned package and checkpoint identity.
    """
    return _versioned_artifact_identity("meanaudio_16k", meanaudio_artifact_digest(checkpoint))


def _pupujepa_tiny_artifact_identity(checkpoint: str) -> str:
    """Return the pinned source and teacher-checkpoint identity.

    :param checkpoint: Canonical Hugging Face repo or local checkpoint directory.
    :returns: Versioned source and checkpoint identity.
    """
    return _versioned_artifact_identity(
        "pupujepa_tiny", pupujepa_artifact_digest(checkpoint, "tiny")
    )


def _pupujepa_large_artifact_identity(checkpoint: str) -> str:
    """Return the pinned source and Large teacher-checkpoint identity.

    :param checkpoint: Canonical Hugging Face repo or local checkpoint directory.
    :returns: Versioned source and checkpoint identity.
    """
    return _versioned_artifact_identity(
        "pupujepa_large", pupujepa_artifact_digest(checkpoint, "large")
    )


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


def _encode_m2l_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
) -> pa.Array:
    """Encode one audio batch as a fixed-shape m2l tensor column.

    :param sources: Decoded source columns carrying the ``(B, C, T)`` audio batch.
    :param sample_rate: Unused source sample rate.
    :param encoder: m2l encoder over the original channel layout.
    :returns: Fixed-shape tensor array.
    :raises ValueError: The encoder returns the wrong row count, rank, or non-finite values.
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    del sample_rate
    audio = sources[AUDIO_FIELD]
    encode = cast("M2LEncodeFn", encoder)
    latents = _finite_embedding(M2L_FIELD, encode(audio))
    if latents.ndim < 2 or len(latents) != len(audio):
        raise ValueError(
            f"{M2L_FIELD} encoder produced shape {latents.shape}, expected {len(audio)} rows "
            "with at least one embedding dimension"
        )
    return tensor_array(latents, np.dtype("float32"), latents.shape[1:])


def _encode_clap_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
) -> pa.Array:
    """Encode one audio batch as fixed-width CLAP vectors.

    :param sources: Decoded source columns carrying the ``(B, C, T)`` audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: CLAP encoder over mono audio.
    :returns: Fixed-size-list float32 array.
    :raises ValueError: The encoder returns the wrong shape or non-finite values.
    """
    audio = sources[AUDIO_FIELD]
    encode = cast("ClapEncodeFn", encoder)
    vectors = _finite_embedding(CLAP_FIELD, encode(_downmix_to_mono(audio), sample_rate))
    expected_shape = (len(audio), CLAP_EMBEDDING_DIM)
    if vectors.shape != expected_shape:
        raise ValueError(
            f"{CLAP_FIELD} encoder produced shape {vectors.shape}, expected {expected_shape}"
        )
    return _fixed_size_list(vectors, CLAP_EMBEDDING_DIM)


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
    sources: Mapping[str, np.ndarray],
    sample_rate: int,
    encoder: Encoder,
    *,
    field: str,
    frame_count: SameFrameCountFn,
) -> pa.Array:
    """Encode one audio batch under the selected SAME model's frame contract.

    :param sources: Decoded source columns carrying the ``(B, C, T)`` audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: SAME encoder over prepared stereo audio.
    :param field: SAME target column.
    :param frame_count: Model-specific latent-frame calculation.
    :returns: Fixed-shape tensor array.
    :raises ValueError: The encoder returns the wrong shape or non-finite values.
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    audio = sources[AUDIO_FIELD]
    prepared = same_encoder_input(audio, sample_rate)
    encode = cast("SameEncodeFn", encoder)
    latents = _finite_embedding(field, encode(prepared))
    expected_shape = (
        len(audio),
        SAME_EMBEDDING_DIM,
        frame_count(prepared.shape[-1], SAME_SAMPLE_RATE),
    )
    if latents.shape != expected_shape:
        raise ValueError(
            f"{field} encoder produced shape {latents.shape}, expected {expected_shape}"
        )
    return tensor_array(latents, np.dtype("float32"), expected_shape[1:])


def _encode_same_s_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
) -> pa.Array:
    """Encode a SAME-S Arrow column through the shared SAME contract.

    :param sources: Decoded source columns carrying the audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: SAME-S encoder.
    :returns: Fixed-shape tensor array.
    """
    return _encode_same_column(
        sources,
        sample_rate,
        encoder,
        field=SAME_S_FIELD,
        frame_count=same_s_num_latent_frames,
    )


def _encode_same_l_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
) -> pa.Array:
    """Encode a SAME-L Arrow column through the shared SAME contract.

    :param sources: Decoded source columns carrying the audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: SAME-L encoder.
    :returns: Fixed-shape tensor array.
    """
    return _encode_same_column(
        sources,
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


def _load_ssondo_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Load S-SONDO through its pinned checkpoint adapter.

    :param checkpoint: Pinned Hugging Face repo id or hash-identical local artifact.
    :param config: Run config supplying the device.
    :returns: S-SONDO encoder over source audio.
    """
    return load_ssondo_audio_encoder(checkpoint, _resolve_torch_device(config.device))


def _load_matpac_plus_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Load MATPAC++ through TinyMU's managed package dependency.

    :param checkpoint: Exact pinned URI or a hash-identical local artifact.
    :param config: Run config supplying the device.
    :returns: Frozen TinyMU encoder.
    """
    return load_matpac_plus_audio_encoder(
        checkpoint,
        device=_resolve_torch_device(config.device),
    )


def _load_meanaudio_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Load MeanAudio's encoder-only 16 kHz VAE.

    :param checkpoint: Pinned Hugging Face repo or SHA-identical local artifact.
    :param config: Run config supplying the device.
    :returns: Frozen posterior-mean audio encoder.
    """
    return load_meanaudio_audio_encoder(
        checkpoint,
        device=_resolve_torch_device(config.device),
    )


def _load_pupujepa_tiny_spec_encoder(
    checkpoint: str, config: AddEmbeddingsConfig
) -> Encoder:
    """Load PupuJEPA through the registry's uniform factory signature.

    :param checkpoint: Canonical Hugging Face repo or local checkpoint directory.
    :param config: Run config supplying the device.
    :returns: Frozen PupuJEPA Tiny encoder over source audio.
    """
    return load_pupujepa_audio_encoder(
        checkpoint,
        device=_resolve_torch_device(config.device),
        variant="tiny",
    )


def _load_pupujepa_large_spec_encoder(
    checkpoint: str, config: AddEmbeddingsConfig
) -> Encoder:
    """Load PupuJEPA Large through the registry factory signature.

    :param checkpoint: Canonical Hugging Face repo or local checkpoint directory.
    :param config: Run config supplying the device.
    :returns: Frozen PupuJEPA Large encoder over source audio.
    """
    return load_pupujepa_audio_encoder(
        checkpoint,
        device=_resolve_torch_device(config.device),
        variant="large",
    )


def _load_param_shift_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Build the re-render shifter through the registry's uniform factory signature.

    :param checkpoint: Unused registry placeholder.
    :param config: Run config supplying the composed render selection and seed.
    :returns: Renderer-bound param shifter.
    """
    del checkpoint
    return load_param_shifter(config)


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


def _require_stored_mono(audio: np.ndarray) -> None:
    """Reject audio the per-row pyFDN extractor cannot consume.

    :param audio: Decoded stored-audio batch.
    :raises ValueError: The batch is not ``(B, 1, T)`` mono.
    """
    if audio.ndim != 3 or audio.shape[1] != 1:
        raise ValueError(
            f"pyfdn_sketch requires stored mono (B, 1, T) audio, got {audio.shape}"
        )


class PyFDNSketchPoolEncoder:
    """Fan per-row pyFDN sketch extraction across a process pool, bit-exact vs serial.

    The context must be non-fork: lance is live in the backfill process and is not
    fork-safe. Workers unpickle only the lance-free worker module.
    """

    def __init__(self, num_workers: int) -> None:
        """Build the pool once at encoder-load time.

        :param num_workers: Worker processes; the extractor is bandwidth-bound, so returns are sub-
            linear past a few workers.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        self._num_workers = num_workers
        self._pool = ProcessPoolExecutor(
            max_workers=num_workers, mp_context=multiprocessing.get_context("spawn")
        )

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract one batch of temporal sketches through the pool.

        :param audio: Stored ``(B, 1, T)`` mono audio batch.
        :param sample_rate: Source sample rate in Hz.
        :returns: ``(B, controls, frames)`` float32 control stack.
        """
        from synth_setter.pipeline.data.pyfdn_sketch_worker import (
            extract_reverb_sketch_row,
        )

        _require_stored_mono(audio)
        extract = functools.partial(extract_reverb_sketch_row, sample_rate=float(sample_rate))
        chunksize = max(1, len(audio) // (self._num_workers * 4))
        rows = self._pool.map(extract, (row[0] for row in audio), chunksize=chunksize)
        return np.stack(list(rows)).astype(np.float32, copy=False)

    def close(self) -> None:
        """Release the worker processes."""
        self._pool.shutdown()


def _load_pyfdn_sketch_encoder(
    checkpoint: str, config: AddEmbeddingsConfig
) -> Encoder:
    """Bind the canonical checkpoint-free pyFDN sketch extractor.

    :param checkpoint: Empty registry placeholder.
    :param config: Uniform registry config supplying the worker count; no device is needed.
    :returns: Batch adapter over stored mono waveform audio.
    """
    _pyfdn_sketch_artifact_identity(checkpoint)
    if config.num_workers > 1:
        return PyFDNSketchPoolEncoder(config.num_workers)

    def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        from synth_setter.features import pyfdn_controls

        extract = cast(
            "Callable[[np.ndarray, float], np.ndarray]",
            pyfdn_controls.extract_reverb_sketch,
        )
        _require_stored_mono(audio)
        return np.stack(
            [extract(row[0], sample_rate) for row in audio]
        ).astype(np.float32, copy=False)

    return encode


def _encode_pyfdn_sketch_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
) -> pa.Array:
    """Encode stored waveform audio as the pyFDN temporal-sketch struct.

    :param sources: Decoded source columns carrying ``(B, 1, T)`` real audio.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: Canonical pyFDN sketch extractor batch adapter.
    :returns: Struct array containing the three temporal control families.
    :raises ValueError: Controls have the wrong shape, non-finite values, or leave ``[0, 1]``.
    """
    from synth_setter.pipeline.data.lance_shard import pyfdn_sketch_struct_array

    audio = sources[AUDIO_FIELD]
    encode = cast("SketchEncodeFn", encoder)
    controls = _finite_embedding(PYFDN_SKETCH_STRUCT_FIELD, encode(audio, sample_rate))
    expected_shape = (len(audio), PYFDN_SKETCH_CONTROLS, SKETCH_STORAGE_FRAMES)
    if controls.shape != expected_shape:
        raise ValueError(
            f"{PYFDN_SKETCH_STRUCT_FIELD} encoder produced shape {controls.shape}, "
            f"expected {expected_shape}"
        )
    if controls.min() < -1.0 or controls.max() > 1.0:
        raise ValueError(
            f"{PYFDN_SKETCH_STRUCT_FIELD} controls out of bounds; expected [-1, 1]"
        )
    return pyfdn_sketch_struct_array(controls)


def _encode_ssondo_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
) -> pa.Array:
    """Encode one audio batch as fixed-width S-SONDO vectors.

    :param sources: Decoded source columns carrying the ``(B, C, T)`` audio batch.
    :param sample_rate: Source sample rate in Hz.
    :param encoder: S-SONDO encoder over source audio.
    :returns: Fixed-size-list float32 array.
    :raises ValueError: The encoder returns the wrong shape or non-finite values.
    """
    audio = sources[AUDIO_FIELD]
    encode = cast("SSONDOEncodeFn", encoder)
    vectors = _finite_embedding(SSONDO_FIELD, encode(audio, sample_rate))
    expected_shape = (len(audio), SSONDO_EMBEDDING_DIM)
    if vectors.shape != expected_shape:
        raise ValueError(
            f"{SSONDO_FIELD} encoder produced shape {vectors.shape}, expected {expected_shape}"
        )
    return _fixed_size_list(vectors, SSONDO_EMBEDDING_DIM)


def _encode_t5gemma_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
) -> pa.Array:
    """Encode one param batch as a fixed-shape text-embedding tensor column.

    :param sources: Decoded source columns carrying ``(B, encoded_width)`` param rows.
    :param sample_rate: Unused source sample rate.
    :param encoder: Encoder over param rows.
    :returns: Fixed-shape tensor array.
    :raises ValueError: The encoder returns the wrong row count, rank, or non-finite values.
    """
    from synth_setter.pipeline.data.lance_shard import tensor_array

    del sample_rate
    params = sources[PARAM_ARRAY_FIELD]
    encode = cast("ParamTextEncodeFn", encoder)
    embeddings = _finite_embedding(T5GEMMA_FIELD, encode(params))
    if embeddings.ndim != 3 or len(embeddings) != len(params):
        raise ValueError(
            f"{T5GEMMA_FIELD} encoder produced shape {embeddings.shape}, expected "
            f"{len(params)} rows of (dim, seq) embeddings"
        )
    return tensor_array(embeddings, np.dtype("float32"), embeddings.shape[1:])


@jaxtyped(typechecker=beartype)
def _sketch_encode(
    audio: Float[np.ndarray, "batch channel time"], sample_rate: int, device: str = "cpu"
) -> Float[np.ndarray, "batch control frame"]:
    """Extract sketch controls for one audio batch in memory-capped chunks.

    Every track is per-clip independent, so chunking only moves values within
    float32 kernel jitter (~1e-6, already batch-size-dependent) while bounding
    extraction RSS at the default Lance batch size.

    :param audio: ``(B, C, T)`` audio batch.
    :param sample_rate: Source sample rate deciding the control frame grid.
    :param device: Torch device the extractor runs on.
    :returns: ``(B, NUM_SKETCH_CONTROLS, F)`` float32 controls.
    """
    import torch

    from synth_setter.features.sketch_controls import extract_sketch_controls_batch

    batch = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    chunks = [
        extract_sketch_controls_batch(
            batch[start : start + SKETCH_ENCODE_MAX_BATCH], sample_rate, device=device
        )
        .cpu()
        .numpy()
        for start in range(0, len(batch), SKETCH_ENCODE_MAX_BATCH)
    ]
    return np.concatenate(chunks, axis=0)


def _load_sketch_spec_encoder(checkpoint: str, config: AddEmbeddingsConfig) -> Encoder:
    """Bind the sketch-control extractor to the registry's uniform factory signature.

    Loads PESTO here so the batch transform stays free of model-file I/O and a missing or corrupt
    artifact fails before any row is processed.

    :param checkpoint: Bundled PESTO checkpoint name.
    :param config: Run config supplying the device.
    :returns: Encoder over the original audio batch.
    """
    from synth_setter.features.sketch_controls import load_pesto_model

    device = _resolve_torch_device(config.device)
    load_pesto_model(checkpoint, device=device)
    return functools.partial(_sketch_encode, device=device)


def _encode_sketch_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: Encoder
) -> pa.Array:
    """Encode one audio batch as the pooled nested sketch-control struct (#2707).

    :param sources: Decoded source columns carrying the ``(B, C, T)`` audio batch.
    :param sample_rate: Source sample rate deciding the control frame grid.
    :param encoder: Sketch extractor over the original audio batch.
    :returns: Struct array with canonically pooled loudness/centroid/pitch
        children and their frame-mean ``vec`` IVF companion.
    :raises ValueError: The encoder output is off the frame grid, non-finite,
        or outside the documented control bounds.
    """
    import torch

    from synth_setter.sketch import pool_sketch_controls
    from synth_setter.pipeline.data.lance_shard import sketch_struct_array

    audio = sources[AUDIO_FIELD]
    encode = cast("SketchEncodeFn", encoder)
    controls = _finite_embedding(SKETCH_STRUCT_FIELD, encode(audio, sample_rate))
    frames = mel_n_frames_from_samples(audio.shape[-1], sample_rate)
    expected = (len(audio), NUM_SKETCH_CONTROLS, frames)
    if controls.shape != expected:
        raise ValueError(
            f"{SKETCH_STRUCT_FIELD} encoder produced shape {controls.shape}, expected {expected}"
        )
    affine = controls[:, : NUM_SKETCH_CONTROLS - SKETCH_PITCH_BINS]
    pitch = controls[:, NUM_SKETCH_CONTROLS - SKETCH_PITCH_BINS :]
    if affine.min() < -1.0 or affine.max() > 1.0 or pitch.min() < 0.0 or pitch.max() > 1.0:
        raise ValueError(
            f"{SKETCH_STRUCT_FIELD} controls out of bounds: affine rows must lie in [-1, 1] "
            "and pitch rows in [0, 1]"
        )
    pooled = pool_sketch_controls(torch.from_numpy(controls)).numpy()
    return sketch_struct_array(pooled)


EMBEDDING_REGISTRY: dict[str, EmbeddingSpec] = {
    "clap": EmbeddingSpec(
        name="clap",
        column=CLAP_FIELD,
        default_checkpoint=DEFAULT_CLAP_CHECKPOINT,
        co_resident=True,
        index=IndexSpec(pool="none", vector_dim=CLAP_EMBEDDING_DIM),
        load_encoder=_load_clap_spec_encoder,
        encode_column=_encode_clap_column,
        resolve_artifact_identity=_clap_artifact_identity,
    ),
    "m2l": EmbeddingSpec(
        name="m2l",
        column=M2L_FIELD,
        default_checkpoint=DEFAULT_M2L_CHECKPOINT,
        co_resident=True,
        index=IndexSpec(pool="mean", vector_column=f"{M2L_FIELD}_vec"),
        load_encoder=_load_m2l_spec_encoder,
        encode_column=_encode_m2l_column,
        resolve_artifact_identity=_m2l_artifact_identity,
    ),
    "pupujepa_tiny": EmbeddingSpec(
        name="pupujepa_tiny",
        column=PUPUJEPA_TINY_FIELD,
        default_checkpoint=DEFAULT_PUPUJEPA_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(
            pool="mean",
            vector_column=f"{PUPUJEPA_TINY_FIELD}_vec",
            vector_dim=PUPUJEPA_TINY_EMBEDDING_DIM,
        ),
        load_encoder=_load_pupujepa_tiny_spec_encoder,
        encode_column=encode_pupujepa_column,
        resolve_artifact_identity=_pupujepa_tiny_artifact_identity,
    ),
    "pupujepa_large": EmbeddingSpec(
        name="pupujepa_large",
        column=PUPUJEPA_LARGE_FIELD,
        default_checkpoint=DEFAULT_PUPUJEPA_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(
            pool="mean",
            vector_column=f"{PUPUJEPA_LARGE_FIELD}_vec",
            vector_dim=PUPUJEPA_LARGE_EMBEDDING_DIM,
        ),
        load_encoder=_load_pupujepa_large_spec_encoder,
        encode_column=encode_pupujepa_large_column,
        resolve_artifact_identity=_pupujepa_large_artifact_identity,
    ),
    "pyfdn_sketch": EmbeddingSpec(
        name="pyfdn_sketch",
        column=PYFDN_SKETCH_STRUCT_FIELD,
        default_checkpoint="",
        co_resident=True,
        index=None,
        load_encoder=_load_pyfdn_sketch_encoder,
        encode_column=_encode_pyfdn_sketch_column,
        resolve_artifact_identity=_pyfdn_sketch_artifact_identity,
    ),
    "same_s": EmbeddingSpec(
        name="same_s",
        column=SAME_S_FIELD,
        default_checkpoint=DEFAULT_SAME_S_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(pool="mean", vector_column=f"{SAME_S_FIELD}_vec"),
        load_encoder=_load_same_spec_encoder,
        encode_column=_encode_same_s_column,
        resolve_artifact_identity=_same_artifact_identity,
    ),
    "same_l": EmbeddingSpec(
        name="same_l",
        column=SAME_L_FIELD,
        default_checkpoint=DEFAULT_SAME_L_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(pool="mean", vector_column=f"{SAME_L_FIELD}_vec"),
        load_encoder=_load_same_spec_encoder,
        encode_column=_encode_same_l_column,
        resolve_artifact_identity=_same_artifact_identity,
    ),
    # PQ sub-vectors must divide the control-vector width. The companion vec
    # is a struct child written by the encoder, so pooling is "none" (#2707).
    "sketch": EmbeddingSpec(
        name="sketch",
        column=SKETCH_STRUCT_FIELD,
        default_checkpoint=DEFAULT_PESTO_CHECKPOINT,
        co_resident=True,
        index=IndexSpec(
            pool="none",
            num_sub_vectors=SKETCH_INDEX_SUB_VECTORS,
            vector_column=SKETCH_VEC_COLUMN,
            vector_dim=NUM_SKETCH_CONTROLS,
        ),
        load_encoder=_load_sketch_spec_encoder,
        encode_column=_encode_sketch_column,
        resolve_artifact_identity=_sketch_artifact_identity,
    ),
    "ssondo": EmbeddingSpec(
        name="ssondo",
        column=SSONDO_FIELD,
        default_checkpoint=DEFAULT_SSONDO_CHECKPOINT,
        co_resident=True,
        index=IndexSpec(pool="none", vector_dim=SSONDO_EMBEDDING_DIM),
        load_encoder=_load_ssondo_spec_encoder,
        encode_column=_encode_ssondo_column,
        resolve_artifact_identity=_ssondo_artifact_identity,
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
        resolve_artifact_identity=_t5gemma_artifact_identity,
        input_fields=(PARAM_ARRAY_FIELD,),
    ),
    # Not an encoder: every row is re-rendered with one parameter redrawn, so the run's
    # render config replaces a checkpoint and the pass is solo (it holds a plugin host).
    "param_shift": EmbeddingSpec(
        name="param_shift",
        column=SHIFT_FIELD,
        default_checkpoint="",
        co_resident=False,
        index=None,
        load_encoder=_load_param_shift_encoder,
        encode_column=encode_param_shift_column,
        resolve_artifact_identity=_param_shift_artifact_identity,
        input_fields=PARAM_SHIFT_INPUT_FIELDS,
        rerenders=True,
    ),
    "matpac_plus": EmbeddingSpec(
        name="matpac_plus",
        column=MATPAC_PLUS_FIELD,
        default_checkpoint=DEFAULT_MATPAC_PLUS_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(
            pool="mean",
            vector_column=f"{MATPAC_PLUS_FIELD}_vec",
            vector_dim=MATPAC_PLUS_FRONTEND.embedding_dim,
        ),
        load_encoder=_load_matpac_plus_spec_encoder,
        encode_column=encode_matpac_plus_column,
        resolve_artifact_identity=_matpac_plus_artifact_identity,
    ),
    "meanaudio_16k": EmbeddingSpec(
        name="meanaudio_16k",
        column=MEANAUDIO_16K_FIELD,
        default_checkpoint=DEFAULT_MEANAUDIO_CHECKPOINT,
        co_resident=False,
        index=IndexSpec(
            pool="mean",
            num_sub_vectors=MEANAUDIO_INDEX_SUB_VECTORS,
            vector_column=f"{MEANAUDIO_16K_FIELD}_vec",
            vector_dim=MEANAUDIO_EMBEDDING_DIM,
        ),
        load_encoder=_load_meanaudio_spec_encoder,
        encode_column=cast("EncodeColumnFn", encode_meanaudio_column),
        resolve_artifact_identity=_meanaudio_artifact_identity,
    ),
}


def _output_columns(spec: EmbeddingSpec) -> tuple[str, ...]:
    """Return every top-level dataset column emitted by one embedding policy.

    :param spec: Embedding write and index policy.
    :returns: Sequence column followed by its optional vector companion; a dotted companion is a
        struct child the sequence column already carries.
    """
    if spec.index is None or spec.index.vector_column is None:
        return (spec.column,)
    if spec.index.vector_column.startswith(f"{spec.column}."):
        return (spec.column,)
    return spec.column, spec.index.vector_column


def _nested_schema_field(schema: pa.Schema, column: str) -> pa.Field | None:
    """Resolve a possibly dotted column path against a schema.

    :param schema: Dataset schema.
    :param column: Top-level name or dotted struct-child path.
    :returns: The resolved field, or ``None`` when any path segment is absent.
    """
    head, *rest = column.split(".")
    if schema.get_field_index(head) < 0:
        return None
    field = schema.field(head)
    for segment in rest:
        if not pa.types.is_struct(field.type):
            return None
        index = field.type.get_field_index(segment)
        if index < 0:
            return None
        field = field.type.field(index)
    return field


def _guard_existing_columns(dataset: lance.LanceDataset, specs: Sequence[EmbeddingSpec]) -> None:
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
        # ``_rowid`` is synthesized by Lance per scan, so it is never in the schema.
        if field == ROW_ID_FIELD:
            continue
        if field not in dataset.schema.names:
            raise ValueError(f"dataset has no {field!r} column to embed")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    total_rows = dataset.count_rows()
    if total_rows < 1:
        raise ValueError("dataset has no rows to embed")
    return total_rows


def _resume_identity_path(resume_cache: Path) -> Path:
    """Return the identity sidecar path for a Lance UDF cache.

    :param resume_cache: User-selected cache path.
    :returns: Sibling identity sidecar path.
    """
    return resume_cache.with_name(f"{resume_cache.name}.identity")


def _resume_source_identity(
    dataset: lance.LanceDataset,
    *,
    sample_rate: int,
    batch_size: int,
    input_fields: Sequence[str],
) -> str:
    """Identify the exact source and batching contract behind cached UDF outputs.

    :param dataset: Lance source read by the UDF.
    :param sample_rate: Dataset sample rate in Hz.
    :param batch_size: Rows passed to each UDF invocation.
    :param input_fields: Ordered source columns read by the UDF.
    :returns: Stable source-policy identity.
    """
    digest = hashlib.sha256()
    for value in (
        str(dataset.uri),
        str(dataset.version),
        str(sample_rate),
        str(batch_size),
        *input_fields,
    ):
        _update_framed_digest(digest, value.encode())
    _update_framed_digest(digest, dataset.schema.serialize().to_pybytes())
    fragments = sorted(dataset.get_fragments(), key=lambda fragment: fragment.fragment_id)
    for fragment in fragments:
        metadata = json.dumps(
            fragment.metadata.to_json(), sort_keys=True, separators=(",", ":")
        ).encode()
        _update_framed_digest(digest, metadata)
    return digest.hexdigest()


def _prepare_resume_cache(
    resume_cache: Path | None,
    identities: Mapping[str, str],
    source_identity: str,
) -> None:
    """Bind a Lance UDF cache to exact artifacts and source inputs.

    :param resume_cache: Cache path, or ``None`` for a cacheless run.
    :param identities: Artifact identities keyed by embedding name.
    :param source_identity: Exact Lance source and batching-policy identity.
    :raises ValueError: An existing cache belongs to different artifacts or inputs.
    """
    if resume_cache is None:
        return
    digest = hashlib.sha256()
    _update_framed_digest(digest, source_identity.encode())
    for name, identity in sorted(identities.items()):
        _update_framed_digest(digest, name.encode())
        _update_framed_digest(digest, identity.encode())
    expected = digest.hexdigest()
    identity_path = _resume_identity_path(resume_cache)
    if resume_cache.exists() and not identity_path.is_file():
        logger.warning(
            "resume_cache_identity_missing",
            resume_cache=str(resume_cache),
            identity_path=str(identity_path),
            action="discard",
        )
        resume_cache.unlink()
    if resume_cache.exists():
        actual = identity_path.read_text().strip()
        if actual != expected:
            raise ValueError(
                f"resume cache {resume_cache} resume identity in {identity_path} "
                "does not match requested source and policy"
            )
        return
    identity_path.write_text(expected)


def _delete_resume_cache(resume_cache: Path | None) -> None:
    """Best-effort delete a consumed UDF resume cache after commit.

    :param resume_cache: Cache path, or ``None`` for a cacheless run.
    """
    if resume_cache is None:
        return
    try:
        resume_cache.unlink(missing_ok=True)
        _resume_identity_path(resume_cache).unlink(missing_ok=True)
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


def _resolve_artifact_identity(spec: EmbeddingSpec, config: AddEmbeddingsConfig) -> str:
    """Resolve checkpoint and input-policy identity for one embedding.

    :param spec: Embedding policy to identify.
    :param config: Checkpoint and input-policy selection.
    :returns: Identity covering every output-affecting artifact and policy.
    """
    checkpoint = config.checkpoints.get(spec.name, spec.default_checkpoint)
    identity = spec.resolve_artifact_identity(checkpoint)
    policy_values: tuple[str, ...] = ()
    if PARAM_ARRAY_FIELD in spec.input_fields:
        policy_values += (config.param_spec_name or "", config.param_text_normalizer)
    if spec.rerenders:
        policy_values += tuple(param_shift_policy_values(config))
    if not policy_values:
        return identity
    digest = hashlib.sha256()
    for value in policy_values:
        _update_framed_digest(digest, value.encode())
    return f"{identity}:input-policy:{digest.hexdigest()}"


def _load_encoders(specs: Sequence[EmbeddingSpec], config: AddEmbeddingsConfig) -> list[Encoder]:
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


def _decoded_sources(batch: pa.RecordBatch, input_fields: Sequence[str]) -> dict[str, np.ndarray]:
    """Decode each required source column of one batch into a numpy array.

    :param batch: Source batch supplied by Lance.
    :param input_fields: Column names the selected policies read.
    :returns: Decoded arrays keyed by field name.
    """
    return {field: _decoded_column(batch.column(field)) for field in input_fields}


def _decoded_column(column: pa.Array) -> np.ndarray:
    """Decode one Arrow column, preserving tensor columns' per-row shape.

    :param column: Arrow column read from the source batch.
    :returns: ``(B, *inner_shape)`` for tensor columns, ``(B,)`` for flat ones.
    """
    if isinstance(column, pa.FixedShapeTensorArray):
        return column.to_numpy_ndarray()
    return column.to_numpy(zero_copy_only=False)


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
        encoded = spec.encode_column(sources, sample_rate, encoder)
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
    :raises RuntimeError: ``add_columns`` returns without committing every
        target column (a silent no-op write).
    """
    import lance
    import torch

    if not specs:
        raise ValueError("no embedding specs given; nothing to write")
    _guard_existing_columns(dataset, specs)
    input_fields = sorted({field for spec in specs for field in spec.input_fields})
    total_rows = _validate_write_source(dataset, config.batch_size, input_fields)
    # Model construction must not consume the seed governing stochastic encoders.
    with torch.random.fork_rng():
        encoders = _load_encoders(specs, config)
    identities = {spec.name: _resolve_artifact_identity(spec, config) for spec in specs}
    resume_cache = _resume_cache_for_specs(config.resume_cache, config.embeddings, specs)
    source_identity = _resume_source_identity(
        dataset,
        sample_rate=sample_rate,
        batch_size=config.batch_size,
        input_fields=input_fields,
    )
    _prepare_resume_cache(resume_cache, identities, source_identity)
    output_columns = [column for spec in specs for column in _output_columns(spec)]

    logger.info("inferring_embedding_schema", columns=output_columns)
    sample = next(dataset.to_batches(columns=input_fields, limit=1))
    # Schema probing must not perturb stochastic encoders' persisted outputs.
    with torch.random.fork_rng():
        sample_output = _encode_columns(
            _decoded_sources(sample, input_fields), sample_rate, specs, encoders
        )
    output_schema = _embedding_output_schema(
        sample_output.schema, specs, config, identities=identities
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
        output_schema=output_schema,
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
    # A zero-batch replay is valid only when the target columns are already committed.
    uncommitted = [
        column for column in output_columns if column not in dataset.schema.names
    ]
    if uncommitted:
        raise RuntimeError(
            f"add_columns returned without committing column(s) {uncommitted} "
            f"(rows_processed={rows_processed} of {total_rows}); refusing to "
            "treat the write as done"
        )
    _delete_resume_cache(resume_cache)
    logger.info(
        "wrote_embeddings",
        columns=output_columns,
        total_rows=total_rows,
        rows_processed=rows_processed,
        committed_version=dataset.version,
    )
    # Pool-backed encoders hold worker processes; release them with the pass.
    for encoder in encoders:
        closer = getattr(encoder, "close", None)
        if callable(closer):
            closer()
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
    vector_field = _nested_schema_field(dataset.schema, column)
    if vector_field is None:
        raise ValueError(f"dataset has no {column!r} vector column to index")
    vector_dim = vector_field.type.list_size
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
        max(1, round(rows**0.5)) if config.num_partitions is None else config.num_partitions
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


def _embedding_output_schema(
    schema: pa.Schema,
    specs: Sequence[EmbeddingSpec],
    config: AddEmbeddingsConfig,
    *,
    identities: Mapping[str, str] | None = None,
) -> pa.Schema:
    """Attach policy identity to every generated field.

    :param schema: Inferred encoder output schema.
    :param specs: Policies producing the output fields.
    :param config: Checkpoint overrides for this write.
    :param identities: Pre-resolved artifact identities, when available.
    :returns: Schema carrying resumable embedding identity metadata.
    """
    fields = []
    for spec in specs:
        identity = (
            _resolve_artifact_identity(spec, config)
            if identities is None
            else identities[spec.name]
        )
        metadata = {
            _EMBEDDING_NAME_METADATA: spec.name.encode(),
            _EMBEDDING_ARTIFACT_METADATA: identity.encode(),
        }
        fields.extend(
            schema.field(column).with_metadata(metadata) for column in _output_columns(spec)
        )
    return pa.schema(fields)


def _missing_embedding_specs(
    dataset: lance.LanceDataset,
    specs: Sequence[EmbeddingSpec],
    config: AddEmbeddingsConfig,
) -> list[EmbeddingSpec]:
    """Return absent policies while rejecting partial or incompatible commits.

    :param dataset: Open Lance dataset.
    :param specs: Selected embedding policies.
    :param config: Checkpoint overrides for this run.
    :returns: Policies whose complete output schema is absent.
    :raises ValueError: A policy is partial or has a different checkpoint identity.
    """
    names = set(dataset.schema.names)
    missing = []
    for spec in specs:
        expected = set(_output_columns(spec))
        present = expected & names
        if present and present != expected:
            raise ValueError(
                f"dataset has partial {spec.name} columns: {sorted(present)}; "
                f"expected {sorted(expected)}"
            )
        if not present:
            missing.append(spec)
            continue
        field_metadata = {
            column: dataset.schema.field(column).metadata or {} for column in expected
        }
        has_identity = any(
            _EMBEDDING_NAME_METADATA in metadata
            or _EMBEDDING_ARTIFACT_METADATA in metadata
            for metadata in field_metadata.values()
        )
        if not has_identity:
            logger.warning("legacy_embedding_identity_missing", embedding=spec.name)
            continue
        artifact_identity = _resolve_artifact_identity(spec, config).encode()
        for column, metadata in field_metadata.items():
            if (
                metadata.get(_EMBEDDING_NAME_METADATA) != spec.name.encode()
                or metadata.get(_EMBEDDING_ARTIFACT_METADATA) != artifact_identity
            ):
                raise ValueError(
                    f"dataset {column} checkpoint identity does not match requested "
                    f"{spec.name} policy"
                )
    return missing


def _index_config_matches(
    dataset: lance.LanceDataset,
    index_name: str,
    *,
    num_partitions: int,
    num_sub_vectors: int,
    metric: str,
) -> bool:
    """Compare persisted Lance index statistics with requested search policy.

    :param dataset: Open Lance dataset.
    :param index_name: Persisted index name.
    :param num_partitions: Requested IVF partition count.
    :param num_sub_vectors: Requested PQ sub-vector count.
    :param metric: Requested distance metric.
    :returns: Whether every persisted index segment has the requested policy.
    """
    os.environ.setdefault("LANCE_INCLUDE_VECTOR_CENTROIDS", "false")
    stats = dataset.index_statistics(index_name)
    segments = stats.get("indices")
    if not isinstance(segments, list) or not segments:
        return False
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        sub_index = segment.get("sub_index")
        if not isinstance(sub_index, dict):
            return False
        if (
            segment.get("metric_type") != metric
            or segment.get("num_partitions") != num_partitions
            or sub_index.get("num_sub_vectors") != num_sub_vectors
        ):
            return False
    return True


def _matching_index_exists(
    dataset: lance.LanceDataset,
    column: str,
    *,
    index: IndexSpec,
    config: AddEmbeddingsConfig,
) -> bool:
    """Return whether a column index exactly matches the requested policy.

    :param dataset: Open Lance dataset.
    :param column: Vector column selected by registry policy.
    :param index: Registry index defaults.
    :param config: Per-run index overrides.
    :returns: Whether a matching index targets the column.
    :raises ValueError: An index exists with incompatible search semantics.
    """
    rows = dataset.count_rows()
    num_partitions = (
        max(1, round(rows**0.5))
        if config.num_partitions is None
        else config.num_partitions
    )
    num_sub_vectors = config.num_sub_vectors or index.num_sub_vectors
    metric = config.metric
    indices = cast("list[dict[str, object]]", dataset.list_indices())
    column_indices = [entry for entry in indices if entry.get("fields") == [column]]
    for entry in column_indices:
        index_name = entry.get("name")
        if isinstance(index_name, str) and _index_config_matches(
            dataset,
            index_name,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            metric=metric,
        ):
            return True
    if column_indices:
        raise ValueError(f"dataset {column} index configuration does not match requested policy")
    return False


def add_embeddings(config: AddEmbeddingsConfig) -> None:
    """Append registry entries to one Lance dataset and resume missing index work.

    :param config: Validated dataset, embedding, checkpoint, and write settings.
    """
    from synth_setter.pipeline.data.lance_shard import read_shard_metadata

    specs = [EMBEDDING_REGISTRY[name] for name in config.embeddings]
    dataset = _open_lance_dataset(config.lance_uri)
    sample_rate = int(read_shard_metadata(dataset.schema).sample_rate)
    pending = _missing_embedding_specs(dataset, specs, config)
    if config.build_index:
        for spec in specs:
            if spec.index is None:
                continue
            vector_column = spec.index.vector_column or spec.column
            if _nested_schema_field(dataset.schema, vector_column) is not None:
                _matching_index_exists(dataset, vector_column, index=spec.index, config=config)
    if pending:
        _validate_write_source(dataset, config.batch_size)
    output_columns = [column for spec in specs for column in _output_columns(spec)]

    logger.info(
        "adding_embeddings",
        uri=config.lance_uri,
        columns=output_columns,
        sample_rate=sample_rate,
        rows=dataset.count_rows(),
        batch_size=config.batch_size,
    )
    co_resident = [spec for spec in pending if spec.co_resident]
    solo = [spec for spec in pending if not spec.co_resident]
    if co_resident:
        _write_columns(dataset, co_resident, sample_rate, config)
    for spec in solo:
        _write_columns(dataset, [spec], sample_rate, config)

    if config.build_index:
        for spec in specs:
            if spec.index is None:
                continue
            vector_column = spec.index.vector_column or spec.column
            if not _matching_index_exists(dataset, vector_column, index=spec.index, config=config):
                build_index(dataset, vector_column, index=spec.index, config=config)
    logger.info("added_embeddings", uri=config.lance_uri, columns=output_columns)


def _configure_lance_logging(*, debug: bool) -> None:
    """Set native Lance logging before its first import.

    :param debug: Whether to force debug-level native telemetry.
    """
    os.environ.setdefault("LANCE_INCLUDE_VECTOR_CENTROIDS", "false")
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


# Retain the private resolver alias for callers importing it.
_resolve_clap_checkpoint = resolve_clap_checkpoint


def load_clap_audio_encoder(
    checkpoint: str = DEFAULT_CLAP_CHECKPOINT,
    device: str | None = None,
) -> ClapEncodeFn:
    """Load CLAP and return an encoder over mono audio.

    :param checkpoint: Local directory, R2 prefix, or Hugging Face CLAP model id.
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
    target_sample_rate = processor.feature_extractor.sampling_rate

    @torch.no_grad()
    def _encode_chunk(chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        if sample_rate != target_sample_rate:
            wav = audio_fn.resample(wav, sample_rate, target_sample_rate)
        processor_kwargs = {
            "audio": list(wav.numpy()),
            "sampling_rate": target_sample_rate,
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


def load_same_audio_encoder(checkpoint: str, device: str | None = None) -> SameEncodeFn:
    """Load SAME and return an encoder over prepared stereo 44.1 kHz audio.

    :param checkpoint: Local directory, R2 mirror, or HuggingFace repo id.
    :param device: Torch device, or ``None`` for automatic selection.
    :returns: Encoder producing ``(B, SAME_EMBEDDING_DIM, T_lat)`` latents.
    """
    import torch

    checkpoint_dir = resolve_same_checkpoint(checkpoint)
    resolved_device = _resolve_torch_device(device)
    logger.info("loading_same_checkpoint", checkpoint=checkpoint, device=resolved_device)
    model = load_same_autoencoder(checkpoint_dir).to(resolved_device)

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

    try:
        config = AddEmbeddingsConfig.from_hydra_cfg(cfg)
        _configure_lance_logging(debug=config.debug)
        logger.info("lance_logging_configured", native_level=os.environ["LANCE_LOG"])
        add_embeddings(config)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        logger.error("add_embeddings_failed", uri=cfg.get("lance_uri"), error=str(exc))
        sys.exit(1)


def main() -> None:
    """Run the Hydra CLI while allowing keyed overrides on the empty checkpoint map."""
    for index, override in enumerate(sys.argv[1:], start=1):
        if override.startswith("checkpoints."):
            sys.argv[index] = f"+{override}"
    _hydra_main()


if __name__ == "__main__":
    main()
