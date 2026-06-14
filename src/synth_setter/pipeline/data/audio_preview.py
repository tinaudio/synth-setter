"""Encode rendered audio rows to MP3 for the Lance dataset's ``audio_mp3`` column.

The bytes become a per-row ``audio_mp3`` column for Lance viewers — a lossy
convenience artifact, never a training input. ``pedalboard`` and ``librosa``
import lazily so this module stays cheap on paths that never encode a preview.
"""

from __future__ import annotations

import io

import numpy as np

# MPEG layer-III sample rates pedalboard's MP3 encoder accepts without
# resampling; any other rate is resampled to MP3_PREVIEW_SAMPLE_RATE.
_MP3_SUPPORTED_SAMPLE_RATES: frozenset[int] = frozenset({32000, 44100, 48000})
MP3_PREVIEW_SAMPLE_RATE: int = 44100


def encode_mp3_preview(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode one ``(channels, samples)`` audio row to MP3 bytes.

    :param audio: Waveform of shape ``(channels, samples)``; any float dtype
        (the Lance writer passes pedalboard's native ``float32``).
    :param sample_rate: Source sample rate in Hz. Rates outside the MP3-native
        set are resampled to :data:`MP3_PREVIEW_SAMPLE_RATE`.
    :returns: The encoded MPEG layer-III payload.
    :raises ValueError: ``audio`` is not 2-D ``(channels, samples)``.
    """
    if audio.ndim != 2:
        raise ValueError(f"audio must be (channels, samples), got shape {audio.shape}")

    from pedalboard.io import AudioFile

    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    out_rate = sample_rate
    if sample_rate not in _MP3_SUPPORTED_SAMPLE_RATES:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=MP3_PREVIEW_SAMPLE_RATE, axis=-1
        )
        out_rate = MP3_PREVIEW_SAMPLE_RATE

    num_channels = waveform.shape[0]
    buffer = io.BytesIO()
    with AudioFile(
        buffer, "w", samplerate=out_rate, num_channels=num_channels, format="mp3"
    ) as out:
        out.write(waveform)
    return buffer.getvalue()
