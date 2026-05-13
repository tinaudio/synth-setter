"""Shape and mel-front-end primitives shared by writer and validator.

Hosts the per-row array names (``DATASET_FIELD_NAMES``), the mel-spectrogram
constants the writer's ``make_spectrogram`` uses, and the audio / mel-spec /
param-array dataset-shape calculators. Kept as a thin sibling module so that
the (planned) shard validator and the (planned) wds writer can import these
primitives without pulling in the rest of ``generate_vst_dataset.py``'s
import surface (h5py, pedalboard, the VST renderer).
"""

from __future__ import annotations

DATASET_FIELD_NAMES: tuple[str, ...] = ("audio", "mel_spec", "param_array")

MEL_FRAMES_PER_SECOND = 100
MEL_N_MELS = 128
MEL_N_FFT_FRACTION_OF_SAMPLE_RATE = 0.025
MEL_WINDOW = "hamming"


def mel_hop_length(sample_rate: float) -> int:
    """Librosa hop length: ``sample_rate / MEL_FRAMES_PER_SECOND``.

    :param sample_rate: Audio sample rate in Hz.
    :returns: Hop length in samples, rounded down to an integer.
    :rtype: int
    """
    return int(sample_rate / MEL_FRAMES_PER_SECOND)


def mel_n_fft(sample_rate: float) -> int:
    """Librosa FFT window length: ``MEL_N_FFT_FRACTION_OF_SAMPLE_RATE * sample_rate``.

    :param sample_rate: Audio sample rate in Hz.
    :returns: FFT window length in samples, rounded down to an integer.
    :rtype: int
    """
    return int(MEL_N_FFT_FRACTION_OF_SAMPLE_RATE * sample_rate)


def mel_n_frames(sample_rate: float, signal_duration_seconds: float) -> int:
    """Number of mel-time frames librosa produces (``center=True`` default).

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
