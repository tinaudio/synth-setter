"""Estimate normalization statistics from raw waveform batches."""

from collections.abc import Iterable

import numpy as np
import torch

from synth_setter.models.components.spec_encoder import LogMelFrontend
from synth_setter.pipeline.data.stats import finalize, update

NORMALIZATION_SAMPLE_LIMIT = 10_000


def estimate_log_mel_statistics(
    audio_batches: Iterable[torch.Tensor],
    frontend: LogMelFrontend,
    sample_limit: int = NORMALIZATION_SAMPLE_LIMIT,
    *,
    mask_degenerate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-position population statistics over raw log-mel rows.

    :param audio_batches: Waveform tensors shaped ``(batch, samples)``.
    :param frontend: Frontend whose ``forward_raw`` returns unnormalized log-mel grids.
    :param sample_limit: Maximum number of waveforms to consume.
    :param mask_degenerate: Whether constant positions use ``std=1`` with a warning.
    :returns: Float32 ``(mean, std)`` arrays matching one frontend output row.
    :raises ValueError: If the limit is invalid or fewer than two rows are available.
    """
    if sample_limit <= 0:
        raise ValueError(f"sample_limit must be positive, got {sample_limit}")

    state = (0, 0, 0)
    reference = next(frontend.buffers(), torch.empty(0))
    device, dtype = reference.device, reference.dtype
    batches = iter(audio_batches)
    with torch.no_grad():
        while state[0] < sample_limit:
            try:
                batch = next(batches)
            except StopIteration:
                break
            if batch.ndim != 2:
                raise ValueError(f"Expected waveform batches shaped (batch, samples), got {batch.shape}")
            remaining = sample_limit - state[0]
            raw = frontend.forward_raw(batch[:remaining].to(device=device, dtype=dtype))
            rows = raw.detach().to(device="cpu", dtype=torch.float64).numpy()
            if not np.isfinite(rows).all():
                raise ValueError("Raw log-mel features must contain only finite values")
            for row in rows:
                state = update(state, row)

    if state[0] < 2:
        raise ValueError(
            f"Need at least 2 waveform rows to estimate normalization statistics; got {state[0]}"
        )
    mean, std = finalize(state, mask_degenerate=mask_degenerate)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Estimated normalization statistics must contain only finite values")
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
