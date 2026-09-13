"""GPU-capable constant-Q features for Lance conditioning columns."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from synth_setter.cqt import (
    CQT_BINS_PER_OCTAVE,
    CQT_EMBEDDING_DIM,
    CQT_MODE,
    CQT_NUM_OCTAVES,
    CQT_PACKAGE_COMMIT,
    CQT_POLICY_DIGEST,
    cqt_num_frames,
)

type CQTEncodeFn = Callable[[np.ndarray, int], np.ndarray]


def cqt_artifact_digest(checkpoint: str) -> str:
    """Return the immutable source and feature-policy identity.

    :param checkpoint: Empty placeholder; CQT has no learned checkpoint.
    :returns: Source commit and preprocessing policy identity.
    :raises ValueError: A checkpoint override was supplied.
    """
    if checkpoint:
        raise ValueError("cqt is checkpoint-free and rejects checkpoint overrides")
    return CQT_POLICY_DIGEST


def load_cqt_audio_encoder(device: str) -> CQTEncodeFn:
    """Build an adapter returning canonical CQT features as NumPy arrays.

    :param device: Torch device used for transform construction and extraction.
    :returns: Encoder from ``(B, C, T)`` waveforms to float32 CQT features.
    """
    import torch

    from synth_setter.models.components.cqt_encoder import CqtAudioEncoder

    torch_device = torch.device(device)
    encoders: dict[int, CqtAudioEncoder] = {}

    def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode one waveform batch.

        :param audio: Waveforms shaped ``(batch, channels, samples)``.
        :param sample_rate: Waveform sample rate in Hz.
        :returns: CQT log-magnitude features on the canonical time grid.
        :raises ValueError: Audio does not satisfy the rank, channel, or length contract.
        """
        if audio.ndim != 3:
            raise ValueError(f"expected audio shaped (B, C, T), got {audio.shape}")
        encoder = encoders.get(sample_rate)
        if encoder is None:
            encoder = CqtAudioEncoder(sample_rate=sample_rate)
            encoders[sample_rate] = encoder
        waveform = torch.as_tensor(np.ascontiguousarray(audio), device=torch_device)
        return encoder(waveform).cpu().numpy().astype(np.float32, copy=False)

    return encode


__all__ = [
    "CQT_BINS_PER_OCTAVE",
    "CQT_EMBEDDING_DIM",
    "CQT_MODE",
    "CQT_NUM_OCTAVES",
    "CQT_PACKAGE_COMMIT",
    "CQT_POLICY_DIGEST",
    "CQTEncodeFn",
    "cqt_artifact_digest",
    "cqt_num_frames",
    "load_cqt_audio_encoder",
]
