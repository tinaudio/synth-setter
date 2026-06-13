"""Append LAION CLAP audio embeddings to Lance split files during finalize.

The functional core (:func:`clap_augmented_schema`, :func:`clap_augment_split`)
maps a finalized Lance split's ``(schema, batches)`` to the same stream with an
extra ``clap`` fixed-shape-tensor column, reading each batch's ``audio`` rows,
downmixing them to mono, and running an *injected* ``encode`` callable — so it
is exercised without loading a CLAP checkpoint. :func:`load_clap_audio_encoder`
is the thin shell that builds the real encoder behind a lazy ``transformers``
import; its only test is a manual run (see the PR's "Verification" checklist).
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
DEFAULT_CLAP_CHECKPOINT: str = "laion/clap-htsat-unfused"
# CLAP's feature extractor rejects any other input rate, so audio is resampled to
# this before encoding.
CLAP_SAMPLE_RATE: int = 48000

# On-disk dtype of the ``clap`` column. float32 keeps the projected embedding at
# full precision (unlike the float16 audio it is derived from).
_CLAP_DTYPE = np.dtype("float32")

# Maps a mono ``(B, T)`` batch and its sample rate to a ``(B, D)`` embedding batch.
EncodeFn: TypeAlias = Callable[[np.ndarray, int], np.ndarray]


def clap_augmented_schema(schema: pa.Schema, dim: int) -> pa.Schema:
    """Append a ``clap`` ``(dim,)`` float32 tensor column to a Lance shard schema.

    :param schema: Schema of a finalized Lance split (core dataset columns).
    :param dim: Embedding width; the appended column's per-row tensor shape.
    :returns: ``schema`` with the trailing ``clap`` column, metadata preserved.
    """
    clap_type = pa.fixed_shape_tensor(pa.from_numpy_dtype(_CLAP_DTYPE), (dim,))
    return schema.append(pa.field(CLAP_FIELD, clap_type, nullable=False))


def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Average the channel axis of an ``(B, C, T)`` batch into ``(B, T)`` mono float32.

    ``dtype=np.float32`` upcasts the stored float16 audio in the reduction, so the
    encoder never sees half-precision.

    :param audio: Audio batch shaped ``(B, C, T)`` with ``C >= 1`` (on-disk float16).
    :returns: Mono batch shaped ``(B, T)`` as float32.
    """
    return audio.mean(axis=1, dtype=np.float32)


def _iter_clap_record_batches(
    in_schema: pa.Schema,
    out_schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    encode: EncodeFn,
    sample_rate: int,
    *,
    dim: int,
) -> Iterator[pa.RecordBatch]:
    """Yield each input batch with a ``clap`` column appended under ``out_schema``.

    :param in_schema: Schema of the incoming batches (must carry ``audio``).
    :param out_schema: ``in_schema`` extended with the ``clap`` column.
    :param batches: Record batches of one finalized Lance split, in row order.
    :param encode: Maps a mono ``(B, T)`` batch and ``sample_rate`` to a
        ``(B, dim)`` float32 embedding batch.
    :param sample_rate: Audio sample rate passed through to ``encode``.
    :param dim: Expected embedding width; ``encode`` output is asserted to match.
    :yields: Each batch with the trailing ``clap`` column.
    :ytype: pa.RecordBatch
    :raises ValueError: ``encode`` returns anything other than a ``(num_rows, dim)``
        batch, where ``num_rows`` is the input batch's row count.
    """
    audio_inner = tuple(in_schema.field(AUDIO_FIELD).type.shape)
    audio_index = in_schema.get_field_index(AUDIO_FIELD)
    for batch in batches:
        audio = tensor_chunk_to_numpy(batch.column(audio_index), audio_inner)
        embeddings = np.asarray(encode(_downmix_to_mono(audio), sample_rate))
        # Pin both axes: a wrong row count would otherwise surface only as a cryptic
        # Arrow "arrays must have the same length" error at record_batch assembly.
        if embeddings.shape != (batch.num_rows, dim):
            raise ValueError(
                f"CLAP encoder produced shape {embeddings.shape}, "
                f"expected ({batch.num_rows}, {dim})"
            )
        clap_column = tensor_array(embeddings, _CLAP_DTYPE, (dim,))
        yield pa.record_batch([*batch.columns, clap_column], schema=out_schema)


def clap_augment_split(
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    encode: EncodeFn,
    sample_rate: int,
    *,
    dim: int,
) -> tuple[pa.Schema, Iterator[pa.RecordBatch]]:
    """Augment one Lance split's ``(schema, batches)`` with a ``clap`` embedding column.

    Builds the extended schema once and returns it alongside a generator that
    appends each row's embedding, so the schema is available to the writer before
    the batches are consumed.

    :param schema: Schema of the incoming batches (must carry ``audio``).
    :param batches: Record batches of one finalized Lance split, in row order.
    :param encode: Maps a mono ``(B, T)`` batch and ``sample_rate`` to a
        ``(B, dim)`` float32 embedding batch.
    :param sample_rate: Audio sample rate passed through to ``encode``.
    :param dim: Embedding width of the appended column.
    :returns: The extended schema and the ``clap``-augmented batch iterator.
    """
    out_schema = clap_augmented_schema(schema, dim)
    augmented = _iter_clap_record_batches(
        schema, out_schema, batches, encode, sample_rate, dim=dim
    )
    return out_schema, augmented


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
    """
    import torch
    import torchaudio.functional as audio_fn
    from transformers import ClapModel, ClapProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading_clap_checkpoint", checkpoint=checkpoint, device=device)
    # ``Any``: ``@can_return_tuple`` makes ``get_audio_features`` return a union that
    # would defeat the ``.pooler_output`` access below under concrete annotations.
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
        # pooler_output holds the projected, L2-normalized embedding (B, D).
        features = model.get_audio_features(**inputs)
        return features.pooler_output.cpu().numpy()

    return encode
