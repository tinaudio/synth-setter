"""Behavior tests for the MP3 audio-preview encoder.

Each test drives the real pedalboard encoder and decodes the produced bytes
back through ``pedalboard.io.AudioFile`` — a codec bug surfaces as a decode
failure or a wrong channel/rate, never as a silently-accepted blob.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from pedalboard.io import AudioFile

from synth_setter.pipeline.data.audio_preview import (
    MP3_PREVIEW_SAMPLE_RATE,
    encode_mp3_preview,
)


def _decode(mp3_bytes: bytes) -> tuple[int, int]:
    """Decode MP3 bytes and return ``(samplerate, num_channels)``.

    :param mp3_bytes: Encoded MP3 payload.
    :returns: The decoded stream's sample rate and channel count.
    """
    with AudioFile(io.BytesIO(mp3_bytes)) as f:
        return int(f.samplerate), int(f.num_channels)


@pytest.mark.parametrize("rate", [32000, 44100, 48000])
def test_encode_mp3_preview_supported_rate_keeps_that_rate(rate: int) -> None:
    """Each MP3-native rate is encoded as-is, not resampled away.

    :param rate: A sample rate the MP3 encoder accepts directly.
    """
    audio = (np.random.default_rng(0).random((2, rate)) * 2 - 1).astype(np.float32)

    samplerate, channels = _decode(encode_mp3_preview(audio, rate))

    assert samplerate == rate
    assert channels == 2


def test_encode_mp3_preview_unsupported_rate_resamples_to_preview_rate() -> None:
    """A rate MP3 cannot represent is resampled to the standard preview rate."""
    audio = (np.random.default_rng(3).random((2, 100)) * 2 - 1).astype(np.float16)

    payload = encode_mp3_preview(audio, 100)
    samplerate, channels = _decode(payload)

    assert samplerate == MP3_PREVIEW_SAMPLE_RATE
    assert channels == 2
    # A degenerate near-empty encode would still carry a valid header; require
    # real frame bytes so a dropped-signal regression fails here.
    assert len(payload) > 100


def test_encode_mp3_preview_non_2d_input_raises_value_error() -> None:
    """A 1-D waveform (missing the channel axis) is rejected before encoding."""
    with pytest.raises(ValueError, match="channels, samples"):
        encode_mp3_preview(np.zeros(44100, dtype=np.float16), 44100)


def test_encode_mp3_preview_mono_input_stays_mono() -> None:
    """A single-channel row encodes to a mono MP3."""
    audio = (np.random.default_rng(1).random((1, 44100)) * 2 - 1).astype(np.float32)

    _, channels = _decode(encode_mp3_preview(audio, 44100))

    assert channels == 1


def test_encode_mp3_preview_float16_input_produces_decodable_mp3() -> None:
    """Float16 audio (the on-disk dtype) encodes without a dtype error."""
    audio = (np.random.default_rng(2).random((2, 44100)) * 2 - 1).astype(np.float16)

    samplerate, channels = _decode(encode_mp3_preview(audio, 44100))

    assert (samplerate, channels) == (44100, 2)


def test_encode_mp3_preview_returns_mp3_frame_header() -> None:
    """The payload begins with an MPEG audio frame sync so viewers detect it as MP3."""
    audio = np.zeros((2, 44100), dtype=np.float16)

    payload = encode_mp3_preview(audio, 44100)

    # 0xFFE_ is the 11-bit MPEG frame sync; the third byte's high nibble varies.
    assert payload[0] == 0xFF
    assert payload[1] & 0xE0 == 0xE0
