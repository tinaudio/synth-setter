"""Encode rendered audio rows to MP3 for the Lance dataset's preview column.

The finalize stage attaches the returned bytes as a per-row ``audio_mp3``
column so open-source Lance viewers can play an audio preview. The MP3 is a
lossy convenience artifact, never a training input — the lossless ``audio``
tensor column remains the source of truth.

Kept free of the project's heavy import surface: ``pedalboard`` and
``librosa`` import lazily inside :func:`encode_mp3_preview` so importing this
module (or its caller, ``lance_shard``) stays cheap for the validator and
launcher paths that never encode a preview.
"""

from __future__ import annotations

import io

import numpy as np

# MPEG layer-III sample rates pedalboard's MP3 encoder accepts. Any other rate
# (e.g. the 100 Hz smoke fixtures) is resampled to MP3_PREVIEW_SAMPLE_RATE.
_MP3_SUPPORTED_SAMPLE_RATES: frozenset[int] = frozenset({32000, 44100, 48000})
MP3_PREVIEW_SAMPLE_RATE: int = 44100


def encode_mp3_preview(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode one ``(channels, samples)`` audio row to MP3 bytes.

    :param audio: A single render's waveform, shape ``(channels, samples)``;
        any float dtype (the on-disk column is ``float16``).
    :param sample_rate: Source sample rate in Hz. Rates MP3 cannot represent
        (i.e. not 32000/44100/48000) are resampled to
        :data:`MP3_PREVIEW_SAMPLE_RATE`.
    :returns: The encoded MP3 payload.
    :raises ValueError: ``audio`` is not 2-D ``(channels, samples)``.
    """
    from pedalboard.io import AudioFile

    if audio.ndim != 2:
        raise ValueError(f"audio must be (channels, samples), got shape {audio.shape}")

    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    if sample_rate not in _MP3_SUPPORTED_SAMPLE_RATES:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=MP3_PREVIEW_SAMPLE_RATE, axis=-1
        )
        sample_rate = MP3_PREVIEW_SAMPLE_RATE

    num_channels = waveform.shape[0]
    buffer = io.BytesIO()
    with AudioFile(
        buffer, "w", samplerate=sample_rate, num_channels=num_channels, format="mp3"
    ) as out:
        out.write(waveform)
    return buffer.getvalue()
