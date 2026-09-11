"""GPU-capable constant-Q features for Lance conditioning columns."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from synth_setter.data.vst.shapes import mel_n_frames_from_samples

CQT_PACKAGE_COMMIT = "2404597d0ccef74a93cad05d0619cb42728f987b"
CQT_NUM_OCTAVES = 8
CQT_BINS_PER_OCTAVE = 32
CQT_EMBEDDING_DIM = CQT_NUM_OCTAVES * CQT_BINS_PER_OCTAVE
CQT_MODE = "matrix"
CQT_POLICY_DIGEST = (
    f"source:{CQT_PACKAGE_COMMIT};transform:nsgt-{CQT_MODE};"
    f"octaves:{CQT_NUM_OCTAVES};bins-per-octave:{CQT_BINS_PER_OCTAVE};"
    "channels:mean;scale:log1p-magnitude;time-grid:100hz-centered"
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


def cqt_num_frames(num_samples: int, sample_rate: int) -> int:
    """Return the CQT storage-frame count on the canonical 100 Hz grid.

    :param num_samples: Number of waveform samples per row.
    :param sample_rate: Waveform sample rate in Hz.
    :returns: Fixed storage-frame count.
    """
    return mel_n_frames_from_samples(num_samples, sample_rate)


def load_cqt_audio_encoder(device: str) -> CQTEncodeFn:
    """Build a lazy upstream CQT encoder on one Torch device.

    Transform kernels are cached by sample rate and clip length because upstream
    fixes both values at construction time.

    :param device: Torch device used for transform construction and extraction.
    :returns: Encoder from ``(B, C, T)`` waveform batches to ``(B, 256, frames)``
        float32 log-magnitude features.
    """
    import torch
    import torch.nn.functional as functional
    from cqt_nsgt_pytorch import CQT_nsgt

    torch_device = torch.device(device)
    transforms: dict[tuple[int, int], CQT_nsgt] = {}

    def encode(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode one waveform batch.

        :param audio: Waveforms shaped ``(batch, channels, samples)``.
        :param sample_rate: Waveform sample rate in Hz.
        :returns: CQT log-magnitude features on the canonical time grid.
        :raises TypeError: Audio cannot be converted to a tensor.
        :raises ValueError: Audio does not satisfy the rank, channel, or length contract.
        """
        if audio.ndim != 3:
            raise ValueError(f"expected audio shaped (B, C, T), got {audio.shape}")
        if audio.shape[1] < 1 or audio.shape[2] < 1:
            raise ValueError(f"expected audio with channels and samples, got {audio.shape}")

        num_samples = audio.shape[2]
        key = sample_rate, num_samples
        transform = transforms.get(key)
        if transform is None:
            transform = CQT_nsgt(
                CQT_NUM_OCTAVES,
                CQT_BINS_PER_OCTAVE,
                mode=CQT_MODE,
                fs=sample_rate,
                audio_len=num_samples,
                device=str(torch_device),
                dtype=torch.float32,
                verbose=False,
            )
            transforms[key] = transform

        with torch.inference_mode():
            waveform = torch.as_tensor(
                np.ascontiguousarray(audio, dtype=np.float32), device=torch_device
            )
            mono = waveform.mean(dim=1, keepdim=True)
            coefficients = transform.fwd(mono)
            if not isinstance(coefficients, torch.Tensor):
                raise TypeError(f"CQT {CQT_MODE} mode returned {type(coefficients).__name__}")
            log_magnitude = torch.log1p(coefficients.abs()).squeeze(1)
            features = functional.adaptive_avg_pool1d(
                log_magnitude,
                cqt_num_frames(num_samples, sample_rate),
            )
        return features.cpu().numpy().astype(np.float32, copy=False)

    return encode
