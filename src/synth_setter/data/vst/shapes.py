"""Shape and mel-front-end primitives shared by the writers and the validator.

Hosts the per-row array names, on-disk dtypes, mel-spectrogram constants, and
dataset-shape calculators. Kept as a thin sibling module so that the shard
validator and the writers can import these primitives without pulling in the
rest of ``generate_vst_dataset.py``'s import surface (h5py, pedalboard, the
VST renderer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Type-only on purpose: a runtime import would risk a cycle (spec.py lazily
    # imports the param-spec registry from data.vst).
    from synth_setter.pipeline.schemas.spec import RenderConfig

AUDIO_FIELD: str = "audio"
MEL_SPEC_FIELD: str = "mel_spec"
PARAM_ARRAY_FIELD: str = "param_array"
CLAP_FIELD: str = "clap"
DATASET_FIELD_NAMES: tuple[str, ...] = (AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD)

# Row width of the optional ``clap`` Lance column; must track DEFAULT_CLAP_CHECKPOINT's
# projection. Kept out of DATASET_FIELD_NAMES: opt-in, not a per-shard core field.
CLAP_EMBEDDING_DIM: int = 512

# Per-field on-disk dtype, matching what the HDF5 / wds writers emit. Audio is
# stored as ``float16`` for compressed storage efficiency; mel and params stay
# ``float32``. Consumers upcast as needed; this map is the single source of
# truth the validator enforces and the resharder honors when constructing
# VirtualLayouts.
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
    return 1 + audio_length // mel_hop_length(sample_rate)


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
