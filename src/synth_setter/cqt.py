"""Canonical constant-Q transform policy shared by online and cached conditioning."""

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


def cqt_num_frames(num_samples: int, sample_rate: int) -> int:
    """Return the CQT storage-frame count on the canonical 100 Hz grid.

    :param num_samples: Number of waveform samples per row.
    :param sample_rate: Waveform sample rate in Hz.
    :returns: Fixed conditioning-frame count.
    """
    return mel_n_frames_from_samples(num_samples, sample_rate)
