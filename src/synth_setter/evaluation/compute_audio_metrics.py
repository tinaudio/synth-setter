"""Run paper evaluations against rendered audio.

Expects audio in the following folder structure:

audio/
    sample_0/
        target.wav
        pred.wav
        ...
    sample_1/
        ...
    ...

We compute the following metrics:

1. MSS: log-Mel multi-scale spectrogram (10ms, 25ms, 100ms) windows and
    (5ms, 10ms, 50ms) hop lengths, (32, 64, 128) mels, hann window, L1 distance.
2. JTFS: joint time-frequency scattering transform, L1 distance.
3. wMFCC: dynamic time-warping cost between MFCCs (50ms window, 10ms hop), 128 mels, L1 distance
4. f0 features: intermediate features from some sort of pitch NN (check speech
    literature for an option here?). cosine sim.
5. amp env: compute RMS amp envelopes (50ms window, 25ms hop). take cosine similarity
    (i.e. normalized dot prod).
"""

import math
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import librosa
import numpy as np
import pandas as pd
import pesto
import torch
from dtw import dtw
from kymatio.numpy import Scattering1D
from loguru import logger
from pedalboard.io import AudioFile

from synth_setter.evaluation.shuffle_pred_audio import (
    params_are_uniform,
    shuffle_pred_audio,
)


def subdir_matches_pattern(sample_dir: Path) -> bool:
    """Return ``True`` if ``sample_dir`` contains ``pred.wav`` and ``target.wav``.

    :param sample_dir: Directory to inspect.
    :returns: True when both audio files are present.
    """
    return (sample_dir / "target.wav").exists() and (sample_dir / "pred.wav").exists()


def find_possible_subdirs(audio_dir: Path) -> list[Path]:
    """Return subdirs of ``audio_dir`` that contain both ``pred.wav`` and ``target.wav``.

    :param audio_dir: Root directory whose immediate children are candidate sample dirs.
    :returns: Matching subdirs (order is filesystem-dependent).
    """
    all_subdirectories = [d for d in audio_dir.glob("*") if d.is_dir()]
    matching_dirs = [d for d in all_subdirectories if subdir_matches_pattern(d)]
    return matching_dirs


MEL_PARAMS = [
    (10, 5, 32),
    (25, 10, 64),
    (100, 50, 128),
]


def compute_mel_specs(y: np.ndarray, sample_rate: float = 44100.0) -> list[np.ndarray]:
    mel_specs = []
    for window_size, hop_size, n_mels in MEL_PARAMS:
        window_size = int(window_size * sample_rate / 1000.0)
        hop_size = int(hop_size * sample_rate / 1000.0)

        spec = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=window_size,
            hop_length=hop_size,
            window="hann",
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        mel_specs.append(spec_db)

    return mel_specs


def compute_mss(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing MSS...")
    target_specs = compute_mel_specs(target)
    pred_specs = compute_mel_specs(pred)

    dist = 0.0
    for target_spec, pred_spec in zip(target_specs, pred_specs):
        dist += np.mean(np.abs(target_spec - pred_spec))

    dist = dist / len(target_specs)
    return dist


scatter = None


def compute_jtfs(y: np.ndarray, J: int = 10, Q: int = 12) -> np.ndarray:
    global scatter
    if scatter is None:
        scatter = Scattering1D(J=J, Q=Q, shape=y.shape[-1])

    return scatter(y)


def compute_jtfs_distance(target: np.ndarray, pred: np.ndarray, J: int = 10, Q: int = 12) -> float:
    logger.info("Computing JTFS...")

    target_jtfs = compute_jtfs(target, J, Q)
    pred_jtfs = compute_jtfs(pred, J, Q)

    dist = np.mean(np.abs(target_jtfs - pred_jtfs))
    return dist


def compute_mfcc(target: np.ndarray, sample_rate: float = 44100.0) -> np.ndarray:
    window_length = int(0.05 * sample_rate)
    hop_length = int(0.01 * sample_rate)

    mfcc = librosa.feature.mfcc(
        y=target,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=window_length,
        hop_length=hop_length,
        n_mels=128,
    )

    return mfcc


def _l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean L1 distance between arrays ``a`` and ``b``.

    :param a: First array.
    :param b: Second array, same shape as ``a``.
    :returns: Scalar mean absolute difference.
    """
    return np.mean(np.abs(a - b))


def compute_wmfcc(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing wMFCC...")

    target_mfcc = compute_mfcc(target)
    pred_mfcc = compute_mfcc(pred)

    target_mfcc = target_mfcc.reshape(-1, target_mfcc.shape[-1])
    pred_mfcc = pred_mfcc.reshape(-1, pred_mfcc.shape[-1])

    dist = dtw(target_mfcc.T, pred_mfcc.T, dist_method=_l1_distance, distance_only=True)
    return dist.normalizedDistance


pesto_model = None


@torch.no_grad()
def get_pesto_activations(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0
) -> np.ndarray:
    global pesto_model
    if pesto_model is None:
        pesto_model = pesto.load_model("mir-1k_g7", step_size=20.0)

    tp = np.stack((target, pred), axis=0)
    x = torch.from_numpy(tp)
    x = x.mean(1)
    preds, confidence, _, _ = pesto_model(x, sample_rate)

    target_f0, pred_f0 = preds.chunk(2, 0)
    target_confidence, pred_confidence = confidence.chunk(2, 0)

    mask = (target_confidence > 0.85) & (pred_confidence > 0.85)
    return target_f0[mask].numpy(), pred_f0[mask].numpy()


def compute_f0(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing f0...")
    target_f0, pred_f0 = get_pesto_activations(target, pred)
    return np.mean(np.abs(target_f0 - pred_f0))


def get_stft(y: np.ndarray, sample_rate: float = 44100.0) -> np.ndarray:
    win_length = int(0.05 * sample_rate)
    hop_length = int(0.02 * sample_rate)
    stft = librosa.stft(
        y.mean(axis=0),
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    ).T
    stft_mag = np.abs(stft)
    return stft_mag


def batched_wasserstein_distance_np(
    hist1: np.ndarray,
    hist2: np.ndarray,
) -> np.ndarray:
    bin_width = 1 / hist1.shape[-1]
    cdf1 = np.cumsum(hist1, axis=-1)
    cdf2 = np.cumsum(hist2, axis=-1)
    distance = np.sum(np.abs(cdf1 - cdf2), axis=-1) * bin_width
    return distance


def compute_sot(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing SOT...")
    target_stft = get_stft(target)
    pred_stft = get_stft(pred)

    target_stft = target_stft / np.clip(target_stft.sum(axis=-1, keepdims=True), 1e-6, None)
    pred_stft = pred_stft / np.clip(pred_stft.sum(axis=-1, keepdims=True), 1e-6, None)

    dists = batched_wasserstein_distance_np(target_stft, pred_stft)
    return dists.mean()


def compute_rms(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    logger.info("Computing amp env...")
    win_length = int(0.05 * sample_rate)
    hop_length = int(0.025 * sample_rate)

    target_rms = librosa.feature.rms(
        y=target.mean(axis=0), frame_length=win_length, hop_length=hop_length
    )
    pred_rms = librosa.feature.rms(
        y=pred.mean(axis=0), frame_length=win_length, hop_length=hop_length
    )

    target_norm = np.linalg.vector_norm(target_rms, axis=-1, ord=2)
    pred_norm = np.linalg.vector_norm(pred_rms, axis=-1, ord=2)

    # Silent (or near-silent) pred would make ``pred_norm * target_norm`` underflow
    # and the cosine become NaN (``0/0``) or unbounded. Short-circuit to ``0`` so the
    # worst rating is returned and silence cannot be gamed into a higher score.
    denom = target_norm * pred_norm
    if float(denom) < 1e-12:
        logger.warning(
            "compute_rms: denominator underflow "
            "(target_norm={t:.3e}, pred_norm={p:.3e}); returning 0",
            t=float(target_norm),
            p=float(pred_norm),
        )
        return 0.0
    cosine_sim = np.dot(target_rms[0], pred_rms[0]) / denom

    return cosine_sim.mean()


def compute_metrics_on_dir(audio_dir: Path) -> dict[str, float]:
    with AudioFile(str(audio_dir / "target.wav")) as target_file:
        target = target_file.read(target_file.frames)
    with AudioFile(str(audio_dir / "pred.wav")) as pred_file:
        pred = pred_file.read(pred_file.frames)

    mss = compute_mss(target, pred)
    wmfcc = compute_wmfcc(target, pred)
    sot = compute_sot(target, pred)
    rms = compute_rms(target, pred)

    return dict(mss=mss, wmfcc=wmfcc, sot=sot, rms=rms)


def compute_metrics(audio_dirs: list[Path], output_dir: Path) -> Path:
    idxs = []
    rows = []
    for sample_dir in audio_dirs:
        metrics = compute_metrics_on_dir(sample_dir)
        rows.append(metrics)
        idxs.append(sample_dir.name.rsplit("_", 1)[1])

    pid = multiprocessing.current_process().pid

    df = pd.DataFrame(rows, index=idxs)
    metric_file = output_dir / f"metrics-{pid}.csv"
    df.to_csv(metric_file)

    return metric_file


def _aggregate_metrics(audio_dirs: list[Path], work_dir: Path, num_workers: int) -> pd.DataFrame:
    """Run the parallel per-sample metrics pass and return the concatenated DataFrame.

    Intermediate per-worker CSVs are written to ``work_dir`` and left there so the
    caller can write ``metrics.csv`` from them if desired.

    :param audio_dirs: Sample dirs to score (each must contain ``target.wav`` + ``pred.wav``).
    :param work_dir: Directory for per-worker intermediate ``metrics-<pid>.csv`` files.
    :param num_workers: ProcessPoolExecutor worker count.
    :returns: Concatenated per-sample metrics DataFrame.
    """
    sublist_length = math.ceil(len(audio_dirs) / num_workers) if audio_dirs else 1
    sublists = [
        audio_dirs[i * sublist_length : (i + 1) * sublist_length] for i in range(num_workers)
    ]
    metric_dfs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_metrics, sublist, work_dir) for sublist in sublists]
        for future in as_completed(futures):
            metric_file = future.result()
            metric_df = pd.read_csv(metric_file)
            metric_df.set_index(metric_df.columns[0], inplace=True)
            metric_dfs.append(metric_df)
    if not metric_dfs:
        return pd.DataFrame()
    return pd.concat(metric_dfs)


@click.command()
@click.argument("audio_dir", type=str)
@click.argument("output_dir", type=str, default="metrics")
@click.option("--num_workers", "-w", type=int, default=8)
@click.option(
    "--shuffle_seed",
    type=int,
    default=0,
    help="Seed for the render-order probe permutation. Non-zero implies shuffle is intended.",
)
def main(audio_dir: str, output_dir: str, num_workers: int, shuffle_seed: int) -> None:
    """Score rendered audio under ``audio_dir`` and write aggregated metrics to ``output_dir``.

    Runs the parallel per-sample pass writing ``metrics.csv`` and
    ``aggregated_metrics.csv``. When all sample dirs share identical
    ``params.csv`` (render-order probe, #489), a second pass with permuted
    ``pred.wav`` symlinks writes ``aggregated_metrics_shuffled.csv``.

    :param audio_dir: Root containing per-sample subdirectories
        (each must have ``pred.wav`` and ``target.wav``).
    :param output_dir: Destination for CSV outputs.
    :param num_workers: Number of parallel worker processes.
    :param shuffle_seed: Permutation seed for the render-order probe; non-zero
        implies the probe is intended and raises if params are not uniform.
    :raises ValueError: when ``shuffle_seed`` is non-zero but ``params.csv``
        files are not uniform across sample dirs.
    """
    audio_dir = Path(audio_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    audio_dirs = find_possible_subdirs(audio_dir)

    df = _aggregate_metrics(audio_dirs, output_dir, num_workers)
    df.to_csv(output_dir / "metrics.csv")

    columnwise_means = df.mean(axis=0)
    columnwise_stds = df.std(axis=0)
    logger.info("metric means:\n{m}", m=columnwise_means.to_string())
    logger.info("metric stds:\n{s}", s=columnwise_stds.to_string())

    pd.DataFrame({"mean": columnwise_means, "std": columnwise_stds}).to_csv(
        output_dir / "aggregated_metrics.csv"
    )

    # Render-order probe (#489): filter to sample_* dirs to match shuffle_pred_audio._sample_dirs
    # (find_possible_subdirs uses glob("*"), _sample_dirs uses glob("sample_*")).
    probe_dirs = [d for d in audio_dirs if d.name.startswith("sample_")]
    uniform = params_are_uniform(probe_dirs)
    if not uniform and shuffle_seed != 0:
        raise ValueError(
            f"shuffle_seed={shuffle_seed} was set but params.csv files are not uniform across "
            "sample dirs — the render-order probe requires identical params. Either fix the "
            "dataset or omit --shuffle_seed to silently skip the probe."
        )
    if uniform and len(probe_dirs) >= 2:
        shuffled_view = output_dir / "shuffled_audio"
        permutation = shuffle_pred_audio(audio_dir, shuffled_view, shuffle_seed)
        if len(permutation) >= 2:
            logger.info(
                "Render-order probe: scoring permuted pred audio (seed={s})", s=shuffle_seed
            )
            shuffled_dirs = find_possible_subdirs(shuffled_view)
            shuffled_tmp = output_dir / "_shuffle_tmp"
            shuffled_tmp.mkdir(exist_ok=True)
            try:
                shuffled_df = _aggregate_metrics(shuffled_dirs, shuffled_tmp, num_workers)
                pd.DataFrame(
                    {"mean": shuffled_df.mean(axis=0), "std": shuffled_df.std(axis=0)}
                ).to_csv(output_dir / "aggregated_metrics_shuffled.csv")
            finally:
                shutil.rmtree(shuffled_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
