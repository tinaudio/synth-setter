"""MP3 encoding and deterministic identifiers for persisted audio rows."""

from __future__ import annotations

import io
import uuid

import numpy as np
from pedalboard.io import AudioFile

DEFAULT_MP3_BITRATE_KBPS = 128
SUPPORTED_MP3_SAMPLE_RATES: frozenset[int] = frozenset(
    {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000}
)

_AUDIO_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "synth-setter.com")


def validate_mp3_sample_rate(sample_rate: int) -> None:
    """Reject playback rates that the MP3 container cannot represent.

    :param sample_rate: Playback rate in Hz.
    :raises ValueError: ``sample_rate`` is unsupported by the MP3 encoder.
    """
    if sample_rate not in SUPPORTED_MP3_SAMPLE_RATES:
        supported = ", ".join(str(rate) for rate in sorted(SUPPORTED_MP3_SAMPLE_RATES))
        raise ValueError(
            f"MP3 previews require sample_rate to be one of {supported}; got {sample_rate}"
        )


def encode_audio_to_mp3(audio: np.ndarray, sample_rate: int, bitrate_kbps: int) -> bytes:
    """Encode one ``(channels, time)`` audio tensor to a CBR MP3 byte string.

    :param audio: Float audio shaped ``(channels, time_samples)`` with non-empty axes.
    :param sample_rate: Playback rate in Hz.
    :param bitrate_kbps: Constant bitrate in kbps.
    :returns: Complete MP3 bitstream.
    :raises ValueError: The audio shape or playback rate is unsupported.
    """
    if audio.ndim != 2 or 0 in audio.shape:
        raise ValueError(
            f"audio must be 2-D (channels, time) with non-empty axes, got shape {audio.shape}"
        )
    validate_mp3_sample_rate(sample_rate)
    buffer = io.BytesIO()
    with AudioFile(
        buffer,
        "w",
        samplerate=sample_rate,
        num_channels=audio.shape[0],
        format="mp3",
        quality=str(bitrate_kbps),
    ) as output:
        output.write(np.ascontiguousarray(audio, dtype=np.float32))
    return buffer.getvalue()


def audio_uuid(audio: np.ndarray) -> str:
    """Return the deterministic UUIDv5 fingerprint of C-ordered audio bytes.

    :param audio: Audio tensor whose element bytes form the UUID name.
    :returns: Canonical UUIDv5 string under the ``synth-setter.com`` namespace.
    """
    return str(uuid.uuid5(_AUDIO_UUID_NAMESPACE, audio.tobytes(order="C").hex()))
