"""Append LAION CLAP audio embeddings to Lance split files during finalize.

The functional core (:func:`clap_augmented_schema`, :func:`iter_clap_batches`)
maps a finalized Lance split's ``(schema, batches)`` to the same stream with an
extra ``clap`` fixed-shape-tensor column, reading each batch's ``audio`` rows,
downmixing them to mono, and running an *injected* ``encode`` callable — so it
is exercised without loading a CLAP checkpoint. :func:`load_clap_audio_encoder`
is the thin shell that builds the real encoder behind a lazy ``transformers``
import; its only test is a manual run (see the project test plan).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any, TypeAlias

import numpy as np
import pyarrow as pa
import structlog

from synth_setter.data.vst.shapes import AUDIO_FIELD, CLAP_FIELD
from synth_setter.pipeline.data.lance_shard import tensor_array, tensor_chunk_to_numpy

logger = structlog.get_logger(__name__)

# HuggingFace CLAP checkpoint whose audio tower projects to ``CLAP_EMBEDDING_DIM``.
DEFAULT_CLAP_CHECKPOINT = "laion/clap-htsat-unfused"
# CLAP's feature extractor rejects any other input rate, so audio is resampled to
# this before encoding.
CLAP_SAMPLE_RATE = 48000

# Maps a mono ``(B, T)`` batch and its sample rate to a ``(B, D)`` embedding batch.
EncodeFn: TypeAlias = Callable[[np.ndarray, int], np.ndarray]


def clap_augmented_schema(schema: pa.Schema, dim: int) -> pa.Schema:
    """Append a ``clap`` ``(dim,)`` float32 tensor column to a Lance shard schema.

    :param schema: Schema of a finalized Lance split (core dataset columns).
    :param dim: Embedding width; the appended column's per-row tensor shape.
    :returns: ``schema`` with the trailing ``clap`` column, metadata preserved.
    :rtype: pa.Schema
    """
    clap_type = pa.fixed_shape_tensor(pa.from_numpy_dtype(np.dtype("float32")), (dim,))
    return schema.append(pa.field(CLAP_FIELD, clap_type, nullable=False))


def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Average the channel axis of an ``(B, C, T)`` batch into ``(B, T)`` mono float32.

    ``dtype=np.float32`` upcasts the stored float16 audio in the reduction, so the
    encoder never sees half-precision.

    :param audio: Audio batch shaped ``(B, C, T)`` (on-disk float16).
    :returns: Mono batch shaped ``(B, T)`` as float32.
    :rtype: np.ndarray
    """
    return audio.mean(axis=1, dtype=np.float32)


def iter_clap_batches(
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    encode: EncodeFn,
    sample_rate: int,
    *,
    dim: int,
) -> Iterator[pa.RecordBatch]:
    """Yield each input batch with a ``clap`` embedding column appended.

    Reads every batch's ``audio`` rows, downmixes to mono, calls ``encode``, and
    appends the result as a ``clap`` column under :func:`clap_augmented_schema`.

    :param schema: Schema of the incoming batches (must carry ``audio``).
    :param batches: Record batches of one finalized Lance split, in row order.
    :param encode: Maps a mono ``(B, T)`` batch and ``sample_rate`` to a
        ``(B, dim)`` float32 embedding batch.
    :param sample_rate: Audio sample rate passed through to ``encode``.
    :param dim: Expected embedding width; ``encode`` output is asserted to match.
    :yields: Each batch with the trailing ``clap`` column.
    :ytype: pa.RecordBatch
    :raises ValueError: ``encode`` returns anything other than a ``(B, dim)`` batch.
    """
    out_schema = clap_augmented_schema(schema, dim)
    audio_inner = tuple(schema.field(AUDIO_FIELD).type.shape)
    audio_index = schema.get_field_index(AUDIO_FIELD)
    for batch in batches:
        audio = tensor_chunk_to_numpy(batch.column(audio_index), audio_inner)
        embeddings = np.asarray(encode(_downmix_to_mono(audio), sample_rate))
        if embeddings.ndim != 2 or embeddings.shape[1] != dim:
            raise ValueError(
                f"CLAP encoder produced shape {embeddings.shape}, expected (B, {dim})"
            )
        clap_column = tensor_array(embeddings, np.dtype("float32"), (dim,))
        yield pa.record_batch([*batch.columns, clap_column], schema=out_schema)


def load_clap_audio_encoder(
    checkpoint: str = DEFAULT_CLAP_CHECKPOINT,
    device: str | None = None,
) -> EncodeFn:
    """Load a CLAP checkpoint and return a mono-batch encode callable.

    The ``transformers`` and ``torch`` imports are deferred so this module stays
    importable (and its core testable with a fake encoder) without the heavy
    dependency or a model download.

    :param checkpoint: HuggingFace CLAP model id.
    :param device: Torch device string; defaults to cuda when available, else cpu.
    :returns: Encode callable mapping a mono ``(B, T)`` batch and sample rate to a
        ``(B, D)`` float32 embedding batch.
    :rtype: EncodeFn
    """
    import torch
    import torchaudio.functional as audio_fn
    from transformers import ClapModel, ClapProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading_clap_checkpoint", checkpoint=checkpoint, device=device)
    # transformers' processor/model call surface is dynamically typed; Any at this
    # external boundary keeps the type checker honest about what it cannot verify.
    model: Any = ClapModel.from_pretrained(checkpoint)
    model = model.to(device).eval()
    processor: Any = ClapProcessor.from_pretrained(checkpoint)

    @torch.no_grad()
    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(mono))
        if sample_rate != CLAP_SAMPLE_RATE:
            wav = audio_fn.resample(wav, sample_rate, CLAP_SAMPLE_RATE)
        inputs = processor(
            audio=list(wav.numpy()), sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # get_audio_features returns the audio-tower output whose pooler_output
        # holds the projected, L2-normalized joint-space embedding (B, D).
        features = model.get_audio_features(**inputs)
        return features.pooler_output.cpu().numpy()

    return encode
