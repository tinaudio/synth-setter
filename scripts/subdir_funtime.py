import os
import click
import numpy as np
import librosa
from pedalboard.io import AudioFile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEL_PARAMS = [
    (10, 5, 32),
    (25, 10, 64),
    (100, 50, 128),
]

def compute_mel_specs(y: np.ndarray, sample_rate: float = 44100.0):
    """Given an audio signal 'y' of shape (channels, samples), compute three mel-spectrograms using
    different window/hop/mel parameters.

    Return them as a list of dB-scaled numpy arrays.
    """
    # If the input audio has more than 1 channel, we can average down to mono
    # or otherwise choose a specific channel. Here we simply sum across channels:
    if y.ndim == 2 and y.shape[0] > 1:
        y = np.mean(y, axis=0)

    mel_specs = []
    for window_size, hop_size, n_mels in MEL_PARAMS:
        win = int(window_size * sample_rate / 1000.0)
        hop = int(hop_size * sample_rate / 1000.0)

        spec = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=win,
            hop_length=hop,
            window="hann",
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        mel_specs.append(spec_db)

    return mel_specs

def compute_mss(target: np.ndarray, pred: np.ndarray) -> float:
    """Compute the mean spectral (mel) distance (MSS) between 'target' and 'pred' audio, both of
    which must be numpy arrays of shape (channels, samples)."""
    logger.info("Computing MSS...")
    target_specs = compute_mel_specs(target)
    pred_specs = compute_mel_specs(pred)

    dist = 0.0
    for target_spec, pred_spec in zip(target_specs, pred_specs):
        dist += np.mean(np.abs(target_spec - pred_spec))

    dist /= len(target_specs)
    return dist

@click.command()
@click.option(
    "--root_dir",
    required=True,
    help="Path to the folder containing sample_1, sample_2, etc."
)
@click.option(
    "--n_subdirs",
    default=10,
    show_default=True,
    help="Number of subdirectories to process."
)
@click.option(
    "--output_txt",
    default="mss_results.txt",
    show_default=True,
    help="Output file name for storing sorted results."
)
def main(root_dir, n_subdirs, output_txt):
    """Script that computes the MSS metric for up to N subdirectories (each containing 'pred.wav'
    and 'target.wav') and writes a text file containing the subdir names sorted by ascending MSS
    value."""
    # Grab all potential subdirectories in root_dir
    all_subdirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    # If you only want sample_<number> style subdirs, you could filter here. E.g.:
    # all_subdirs = [d for d in all_subdirs if d.startswith("sample_")]

    # Slice to the first n_subdirs if fewer than total are desired
    selected_subdirs = all_subdirs[:n_subdirs]

    results = []

    for subdir in selected_subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        pred_path = os.path.join(subdir_path, "pred.wav")
        target_path = os.path.join(subdir_path, "target.wav")

        # Check existence
        if not (os.path.isfile(pred_path) and os.path.isfile(target_path)):
            logger.warning(f"Skipping {subdir}, missing pred.wav or target.wav.")
            continue

        # Load the audio with pedalboard.io.AudioFile
        with AudioFile(target_path) as f:
            target_audio = f.read(f.frames)
            sample_rate = f.samplerate  # We'll assume pred.wav has the same sample rate

        with AudioFile(pred_path) as f:
            pred_audio = f.read(f.frames)
            # sample_rate for pred assumed consistent, or you can check here

        # Compute MSS
        mss_value = compute_mss(target_audio, pred_audio)
        results.append((subdir, mss_value))

    # Sort by MSS, ascending
    results_sorted = sorted(results, key=lambda x: x[1])

    # Write to text file
    with open(output_txt, "w", encoding="utf-8") as f_out:
        for subdir_name, mss_val in results_sorted:
            f_out.write(f"{subdir_name}\t{mss_val:.6f}\n")

    logger.info(f"Done. Results written to {output_txt}")

if __name__ == "__main__":
    main()
