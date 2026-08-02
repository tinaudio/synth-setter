"""Offline PupuJEPA adapters over the shared Torch waveform encoder."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import pyarrow as pa
import structlog

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    PUPUJEPA_LARGE_FIELD,
    PUPUJEPA_TINY_FIELD,
)
from synth_setter.pupujepa import (
    DEFAULT_PUPUJEPA_CHECKPOINT,
    PUPUJEPA_CHECKPOINT_REVISION,
    PUPUJEPA_CHECKPOINT_SPECS,
    PUPUJEPA_LARGE_CONFIG,
    PUPUJEPA_SAMPLE_RATE,
    PUPUJEPA_TINY_CONFIG,
    PupuJepaConfig,
    PupuJepaVariant,
    PUPUJEPA_UPSTREAM_COMMIT,
    pupujepa_num_time_patches,
)
from synth_setter.utils.logging_utils import resolve_git_sha

logger = structlog.get_logger(__name__)

PUPUJEPA_ENCODE_MAX_BATCH = PUPUJEPA_CHECKPOINT_SPECS["tiny"].encode_max_batch
PUPUJEPA_LARGE_ENCODE_MAX_BATCH = PUPUJEPA_CHECKPOINT_SPECS["large"].encode_max_batch

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
    if audio.min() < -1.0 or audio.max() > 1.0:
        raise ValueError("PupuJEPA input audio channels must lie within [-1, 1]")
    mono = audio.mean(axis=1, dtype=np.float32)
    if mono.min() < -1.0 or mono.max() > 1.0:
        raise ValueError("PupuJEPA input audio must lie within [-1, 1] after downmixing")
    return np.ascontiguousarray(mono)


def _encode_pupujepa_column(
    sources: Mapping[str, np.ndarray],
    sample_rate: int,
    encoder: object,
    *,
    field: str,
    config: PupuJepaConfig,
) -> pa.Array:
    """Encode one audio batch as a fixed-shape PupuJEPA tensor column.

    :param sources: Decoded source columns carrying ``(B, C, T)`` waveforms.
    :param sample_rate: Dataset sample rate in Hz.
    :param encoder: Loaded shared PupuJEPA waveform encoder adapter.
    :param field: Lance field receiving the selected variant.
    :param config: Selected teacher geometry.
    :returns: Float32 tensor array shaped ``(output_dim, time_patches)`` per row.
    :raises ValueError: Encoder rank, orientation, shape, or values violate the contract.
    """
    audio = sources[AUDIO_FIELD]
    encode = cast("PupuJepaEncodeFn", encoder)
    embeddings = np.asarray(encode(audio, sample_rate), dtype=np.float32)
    expected_shape = (
        len(audio),
        config.output_dim,
        pupujepa_num_time_patches(audio.shape[-1], sample_rate, config),
    )
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"{field} encoder produced shape {embeddings.shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{field} encoder produced non-finite values")

    from synth_setter.pipeline.data.lance_shard import tensor_array

    return tensor_array(
        np.ascontiguousarray(embeddings),
        np.dtype("float32"),
        expected_shape[1:],
    )


def encode_pupujepa_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: object
) -> pa.Array:
    """Encode the PupuJEPA Tiny sequence column.

    :param sources: Decoded source columns carrying ``(B, C, T)`` waveforms.
    :param sample_rate: Dataset sample rate in Hz.
    :param encoder: Loaded Tiny encoder adapter.
    :returns: Float32 tensor rows shaped ``(1536, time_patches)``.
    """
    return _encode_pupujepa_column(
        sources,
        sample_rate,
        encoder,
        field=PUPUJEPA_TINY_FIELD,
        config=PUPUJEPA_TINY_CONFIG,
    )


def encode_pupujepa_large_column(
    sources: Mapping[str, np.ndarray], sample_rate: int, encoder: object
) -> pa.Array:
    """Encode the PupuJEPA Large sequence column.

    :param sources: Decoded source columns carrying ``(B, C, T)`` waveforms.
    :param sample_rate: Dataset sample rate in Hz.
    :param encoder: Loaded Large encoder adapter.
    :returns: Float32 tensor rows shaped ``(8192, time_patches)``.
    """
    return _encode_pupujepa_column(
        sources,
        sample_rate,
        encoder,
        field=PUPUJEPA_LARGE_FIELD,
        config=PUPUJEPA_LARGE_CONFIG,
    )


def load_pupujepa_audio_encoder(
    checkpoint: str = DEFAULT_PUPUJEPA_CHECKPOINT,
    *,
    device: str = "cpu",
    variant: PupuJepaVariant = "tiny",
) -> PupuJepaEncodeFn:
    """Load the frozen teacher and return the bounded offline NumPy adapter.

    :param checkpoint: Canonical pinned Hugging Face repo or local checkpoint directory.
    :param device: Explicit Torch inference device.
    :param variant: Released teacher size to load.
    :returns: Encoder from ``(B, C, T)`` waveforms to
        ``(B, config.output_dim, time_patches)`` sequences.
    """
    import torch

    from synth_setter.models.components.pupujepa_encoder import PupuJepaAudioEncoder

    model = PupuJepaAudioEncoder.from_pretrained(
        sample_rate=PUPUJEPA_SAMPLE_RATE,
        checkpoint=checkpoint,
        revision=PUPUJEPA_CHECKPOINT_REVISION,
        variant=variant,
    ).to(device)
    logger.info(
        "loaded_pupujepa_checkpoint",
        checkpoint=checkpoint,
        checkpoint_revision=PUPUJEPA_CHECKPOINT_REVISION,
        device=device,
        source_commit=PUPUJEPA_UPSTREAM_COMMIT,
        variant=variant,
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
        max_batch = PUPUJEPA_CHECKPOINT_SPECS[variant].encode_max_batch
        with torch.inference_mode():
            for start in range(0, len(mono), max_batch):
                waveform = torch.from_numpy(mono[start : start + max_batch]).to(device)
                sequence = model(waveform, sample_rate=sample_rate)
                chunks.append(sequence.float().cpu().numpy())
        return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)

    return encode
