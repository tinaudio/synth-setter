"""Shape and mel-front-end primitives shared by the writers and the validator.

Hosts the per-row array names, on-disk dtypes, mel-spectrogram constants, and
dataset-shape calculators. Kept as a thin sibling module so that the shard
validator and the writers can import these primitives without pulling in the
rest of ``generate_vst_dataset.py``'s import surface (pedalboard, the
VST renderer).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import librosa
import numpy as np

# Re-exported for the writers and validator; canonical home is ``conditioning``.
from synth_setter.conditioning import (
    NUM_SKETCH_CONTROLS as NUM_SKETCH_CONTROLS,
    SKETCH_CENTROID_CHILD as SKETCH_CENTROID_CHILD,
    SKETCH_CENTROID_ROW as SKETCH_CENTROID_ROW,
    SKETCH_CTRL_FIELD as SKETCH_CTRL_FIELD,
    SKETCH_LOUDNESS_CHILD as SKETCH_LOUDNESS_CHILD,
    SKETCH_LOUDNESS_ROW as SKETCH_LOUDNESS_ROW,
    SKETCH_PITCH_BINS as SKETCH_PITCH_BINS,
    SKETCH_PITCH_CHILD as SKETCH_PITCH_CHILD,
    SKETCH_PITCH_SLICE as SKETCH_PITCH_SLICE,
    SKETCH_STRUCT_FIELD as SKETCH_STRUCT_FIELD,
    SKETCH_VEC_CHILD as SKETCH_VEC_CHILD,
)

if TYPE_CHECKING:
    # Type-only on purpose: a runtime import would risk a cycle (spec.py lazily
    # imports the param-spec registry from data.vst).
    from synth_setter.pipeline.schemas.spec import RenderConfig

AUDIO_FIELD: str = "audio"
AUDIO_MP3_FIELD: str = "audio_mp3"
AUDIO_UUID_FIELD: str = "audio_uuid"
DEBUG_FIELD: str = "debug"
MEL_SPEC_FIELD: str = "mel_spec"
PARAM_ARRAY_FIELD: str = "param_array"
DATASET_FIELD_NAMES: tuple[str, ...] = (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)
PREVIEW_FIELD_NAMES: tuple[str, ...] = (AUDIO_MP3_FIELD, AUDIO_UUID_FIELD)

AUDIO_MP3_FIELD_METADATA: dict[bytes, bytes] = {b"mime_type": b"audio/mpeg"}

# Optional audio-embedding columns appended post-hoc by the add_embeddings CLI;
# not in DATASET_FIELD_NAMES because the writers never emit them.
M2L_FIELD: str = "m2l"
CLAP_FIELD: str = "clap"
SAME_S_FIELD: str = "same_s"
SAME_L_FIELD: str = "same_l"
SSONDO_FIELD: str = "ssondo"
T5GEMMA_FIELD: str = "t5gemma"
TINYMU_FIELD: str = "tinymu"
MATPAC_PLUS_FIELD: str = "matpac_plus"
MEANAUDIO_16K_FIELD: str = "meanaudio_16k"
PUPUJEPA_TINY_FIELD: str = "pupujepa_tiny"
PUPUJEPA_LARGE_FIELD: str = "pupujepa_large"
# Emits the 128-semitone x 3-bin activation width that ``SKETCH_PITCH_BINS`` pins.
DEFAULT_PESTO_CHECKPOINT: str = "mir-1k_g7"

# Single-parameter sensitivity struct appended by the ``param_shift`` embedder. One nested
# column keeps the shift's seven facets together and readable as ``shift.param``,
# ``shift.audio``, ... rather than seven suffixed siblings of the dataset's own columns.
SHIFT_FIELD: str = "shift"
SHIFT_PARAM_SUBFIELD: str = "param"
SHIFT_AMOUNT_SUBFIELD: str = "amount"
SHIFT_AUDIO_SUBFIELD: str = "audio"
SHIFT_RMS_SUBFIELD: str = "rms"
SHIFT_SOT_SUBFIELD: str = "sot"
SHIFT_WMFCC_SUBFIELD: str = "wmfcc"
SHIFT_MSS_SUBFIELD: str = "mss"
SHIFT_SUBFIELD_NAMES: tuple[str, ...] = (
    SHIFT_PARAM_SUBFIELD,
    SHIFT_AMOUNT_SUBFIELD,
    SHIFT_AUDIO_SUBFIELD,
    SHIFT_RMS_SUBFIELD,
    SHIFT_SOT_SUBFIELD,
    SHIFT_WMFCC_SUBFIELD,
    SHIFT_MSS_SUBFIELD,
)

# Backward-compatible storage defaults. ``RenderConfig`` overrides signal
# storage; parameter arrays retain the default dtype.
DATASET_FIELD_DTYPES: dict[str, np.dtype] = {
    AUDIO_FIELD: np.dtype("float16"),
    MEL_SPEC_FIELD: np.dtype("float32"),
    PARAM_ARRAY_FIELD: np.dtype("float32"),
}

MEL_FRAMES_PER_SECOND = 100
MEL_N_MELS = 128
MEL_N_FFT_FRACTION_OF_SAMPLE_RATE = 0.025
MEL_WINDOW = "hamming"


def mel_hop_length(sample_rate: float) -> int:
    """Librosa hop length: ``sample_rate / MEL_FRAMES_PER_SECOND``.

    :param sample_rate: Audio sample rate in Hz. Must be at least
        ``MEL_FRAMES_PER_SECOND`` — lower rates round down to a hop of 0,
        which is not a valid librosa ``hop_length``.
    :returns: Hop length in samples, rounded down to an integer.
    :rtype: int
    :raises ValueError: If ``sample_rate`` would produce a hop length of 0.
    """
    hop = int(sample_rate / MEL_FRAMES_PER_SECOND)
    if hop <= 0:
        raise ValueError(
            f"sample_rate={sample_rate} produces hop length {hop}; "
            f"sample_rate must be at least MEL_FRAMES_PER_SECOND={MEL_FRAMES_PER_SECOND}."
        )
    return hop


def mel_n_fft(sample_rate: float) -> int:
    """Librosa FFT window length: ``MEL_N_FFT_FRACTION_OF_SAMPLE_RATE * sample_rate``.

    :param sample_rate: Audio sample rate in Hz. Must be large enough that
        ``int(MEL_N_FFT_FRACTION_OF_SAMPLE_RATE * sample_rate) >= 1`` —
        smaller rates round down to ``n_fft=0``, which is not a valid librosa
        FFT window length.
    :returns: FFT window length in samples, rounded down to an integer.
    :rtype: int
    :raises ValueError: If ``sample_rate`` would produce ``n_fft`` of 0.
    """
    n_fft = int(MEL_N_FFT_FRACTION_OF_SAMPLE_RATE * sample_rate)
    if n_fft <= 0:
        raise ValueError(
            f"sample_rate={sample_rate} produces n_fft {n_fft}; "
            f"sample_rate must be at least "
            f"1/MEL_N_FFT_FRACTION_OF_SAMPLE_RATE={1 / MEL_N_FFT_FRACTION_OF_SAMPLE_RATE}."
        )
    return n_fft


def mel_n_frames(sample_rate: float, signal_duration_seconds: float) -> int:
    """Return the number of mel-time frames librosa produces (``center=True`` default).

    Mirrors librosa's ``1 + audio_length // hop_length`` calculation.

    :param sample_rate: Audio sample rate in Hz.
    :param signal_duration_seconds: Duration of the rendered audio in seconds.
    :returns: Number of time frames in the produced mel spectrogram.
    :rtype: int
    """
    audio_length = int(sample_rate * signal_duration_seconds)
    return mel_n_frames_from_samples(audio_length, sample_rate)


def mel_n_frames_from_samples(num_samples: int, sample_rate: float) -> int:
    """Return the mel-grid frame count for a waveform length in samples.

    :param num_samples: Waveform length in samples.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``1 + num_samples // hop_length`` frames.
    :rtype: int
    """
    return 1 + num_samples // mel_hop_length(sample_rate)


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Per-channel mel-spectrogram in dB; STFT params come from module-level constants.

    Canonical training front-end: every consumer that must match stored
    ``mel_spec`` values calls this rather than reimplementing the librosa call.

    :param audio: Channel-leading waveform shaped ``(channels, samples)``.
    :param sample_rate: Audio sample rate in Hz.
    :returns: ``(channels, MEL_N_MELS, frames)`` decibel-scaled mel spectrogram.
    """
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=MEL_N_MELS,
        n_fft=mel_n_fft(sample_rate),
        hop_length=mel_hop_length(sample_rate),
        window=MEL_WINDOW,
        center=True,
    )
    return librosa.power_to_db(spec, ref=np.max)


def audio_dataset_shape(
    num_samples: int,
    channels: int,
    sample_rate: float,
    signal_duration_seconds: float,
) -> tuple[int, int, int]:
    """Audio dataset shape ``(N, C, time_samples)``.

    :param num_samples: Number of rows (shard batch size).
    :param channels: Audio channels (typically 1 or 2).
    :param sample_rate: Audio sample rate in Hz.
    :param signal_duration_seconds: Duration of each rendered sample in seconds.
    :returns: Three-tuple ``(num_samples, channels, time_samples)``.
    :rtype: tuple[int, int, int]
    """
    return (num_samples, channels, int(sample_rate * signal_duration_seconds))


def mel_dataset_shape(
    num_samples: int,
    channels: int,
    sample_rate: float,
    signal_duration_seconds: float,
) -> tuple[int, int, int, int]:
    """Mel-spectrogram dataset shape ``(N, C, n_mels, n_frames)``.

    :param num_samples: Number of rows (shard batch size).
    :param channels: Audio channels (typically 1 or 2).
    :param sample_rate: Audio sample rate in Hz.
    :param signal_duration_seconds: Duration of each rendered sample in seconds.
    :returns: Four-tuple ``(num_samples, channels, n_mels, n_frames)``.
    :rtype: tuple[int, int, int, int]
    """
    return (
        num_samples,
        channels,
        MEL_N_MELS,
        mel_n_frames(sample_rate, signal_duration_seconds),
    )


def param_array_dataset_shape(num_samples: int, num_params: int) -> tuple[int, int]:
    """Param-array dataset shape ``(N, num_params)``.

    :param num_samples: Number of rows (shard batch size).
    :param num_params: Width of the per-row parameter vector.
    :returns: Two-tuple ``(num_samples, num_params)``.
    :rtype: tuple[int, int]
    """
    return (num_samples, num_params)


def dataset_field_dtypes(render: RenderConfig) -> Mapping[str, np.dtype]:
    """Return the configured physical dtype for each writer-emitted field.

    :param render: Per-shard renderer config supplying signal storage dtypes.
    :returns: Mapping keyed by ``DATASET_FIELD_NAMES``.
    """
    return {
        AUDIO_FIELD: np.dtype(render.audio_dtype),
        MEL_SPEC_FIELD: np.dtype(render.mel_spec_dtype),
        PARAM_ARRAY_FIELD: DATASET_FIELD_DTYPES[PARAM_ARRAY_FIELD],
    }


def dataset_field_shapes(render: RenderConfig, num_params: int) -> dict[str, tuple[int, ...]]:
    """Return the full per-field shapes (leading row axis included) the writers emit per shard.

    Single source of the field→shape contract — keyed by
    ``DATASET_FIELD_NAMES`` with ``N = render.samples_per_shard``.

    :param render: Per-shard renderer config supplying row count, channels,
        sample rate, and duration.
    :param num_params: Width of the per-row parameter vector.
    :returns: Mapping with one full ``(N, ...)`` shape tuple per dataset field.
    :rtype: dict[str, tuple[int, ...]]
    """
    return {
        AUDIO_FIELD: audio_dataset_shape(
            render.samples_per_shard,
            render.channels,
            render.sample_rate,
            render.signal_duration_seconds,
        ),
        MEL_SPEC_FIELD: mel_dataset_shape(
            render.samples_per_shard,
            render.channels,
            render.sample_rate,
            render.signal_duration_seconds,
        ),
        PARAM_ARRAY_FIELD: param_array_dataset_shape(render.samples_per_shard, num_params),
    }
