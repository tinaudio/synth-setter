"""Offline PupuJEPA Tiny adapter over the shared Torch waveform encoder."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import pyarrow as pa
import structlog

from synth_setter.data.vst.shapes import AUDIO_FIELD, PUPUJEPA_TINY_FIELD
from synth_setter.pupujepa import (
    DEFAULT_PUPUJEPA_TINY_CHECKPOINT,
    PUPUJEPA_CHECKPOINT_REVISION,
    PUPUJEPA_EMBEDDING_DIM,
    PUPUJEPA_SAMPLE_RATE,
    PUPUJEPA_UPSTREAM_COMMIT,
    pupujepa_num_time_patches,
)
from synth_setter.utils.logging_utils import resolve_git_sha

logger = structlog.get_logger(__name__)

PUPUJEPA_ENCODE_MAX_BATCH = 16

type PupuJepaEncodeFn = Callable[[np.ndarray, int], np.ndarray]


def pupujepa_encoder_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate and downmix ``(B, C, T)`` audio without leaving Torch inference.

    :param audio: One- or two-channel waveform batch.
    :param sample_rate: Positive source sample rate in Hz.
    :returns: Contiguous float32 mono waveforms at the source rate.
    :raises ValueError: Shape, rate, duration, or values violate the encoder contract.
    """
    if audio.ndim != 3:
        raise ValueError(f"expected a (B, C, T) batch for PupuJEPA, got shape {audio.shape}")
    if len(audio) < 1:
        raise ValueError("PupuJEPA expects a non-empty batch")
    if audio.shape[1] not in (1, 2):
        raise ValueError(f"PupuJEPA expects 1 or 2 channels, got shape {audio.shape}")
    if sample_rate < 1:
        raise ValueError(f"PupuJEPA needs a positive sample_rate, got {sample_rate}")
    pupujepa_num_time_patches(audio.shape[-1], sample_rate)
    if not np.isfinite(audio).all():
        raise ValueError("PupuJEPA input audio contains non-finite values")
    return np.ascontiguousarray(audio.mean(axis=1, dtype=np.float32))


def encode_pupujepa_column(
    sources: Mapping[str, np.ndarray],
    sample_rate: int,
    encoder: object,
) -> pa.Array:
    """Encode one audio batch as a fixed-shape PupuJEPA Tiny tensor column.

    :param sources: Decoded source columns carrying ``(B, C, T)`` waveforms.
    :param sample_rate: Dataset sample rate in Hz.
    :param encoder: Loaded shared PupuJEPA waveform encoder adapter.
    :returns: Float32 tensor array shaped ``(1536, time_patches)`` per row.
    :raises ValueError: Encoder rank, orientation, shape, or values violate the contract.
    """
    audio = sources[AUDIO_FIELD]
    encode = cast("PupuJepaEncodeFn", encoder)
    embeddings = np.asarray(encode(audio, sample_rate), dtype=np.float32)
    expected_shape = (
        len(audio),
        PUPUJEPA_EMBEDDING_DIM,
        pupujepa_num_time_patches(audio.shape[-1], sample_rate),
    )
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"{PUPUJEPA_TINY_FIELD} encoder produced shape {embeddings.shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{PUPUJEPA_TINY_FIELD} encoder produced non-finite values")

    from synth_setter.pipeline.data.lance_shard import tensor_array

    return tensor_array(
        np.ascontiguousarray(embeddings),
        np.dtype("float32"),
        expected_shape[1:],
    )


def load_pupujepa_audio_encoder(
    checkpoint: str = DEFAULT_PUPUJEPA_TINY_CHECKPOINT,
    *,
    device: str = "cpu",
) -> PupuJepaEncodeFn:
    """Load the frozen teacher and return the bounded offline NumPy adapter.

    :param checkpoint: Canonical pinned Hugging Face repo or local checkpoint directory.
    :param device: Explicit Torch inference device.
    :returns: Encoder from ``(B, C, T)`` audio to ``(B, 1536, time_patches)``.
    """
    import torch

    from synth_setter.models.components.pupujepa_encoder import PupuJepaAudioEncoder

    model = PupuJepaAudioEncoder.from_pretrained(
        sample_rate=PUPUJEPA_SAMPLE_RATE,
        checkpoint=checkpoint,
        revision=PUPUJEPA_CHECKPOINT_REVISION,
    ).to(device)
    logger.info(
        "loaded_pupujepa_tiny_checkpoint",
        checkpoint=checkpoint,
        checkpoint_revision=PUPUJEPA_CHECKPOINT_REVISION,
        device=device,
        source_commit=PUPUJEPA_UPSTREAM_COMMIT,
        synth_setter_git_sha=resolve_git_sha(),
    )

    def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode source-rate audio in bounded inference chunks.

        :param audio: One- or two-channel waveform batch.
        :param sample_rate: Positive source sample rate in Hz.
        :returns: Contiguous float32 teacher sequences.
        """
        mono = pupujepa_encoder_input(audio, sample_rate)
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(mono), PUPUJEPA_ENCODE_MAX_BATCH):
                waveform = torch.from_numpy(
                    mono[start : start + PUPUJEPA_ENCODE_MAX_BATCH]
                ).to(device)
                sequence = model(waveform, sample_rate=sample_rate)
                chunks.append(sequence.float().cpu().numpy())
        return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)

    return encode
