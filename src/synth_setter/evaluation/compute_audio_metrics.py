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
6. SOT: spectral optimal transport — per-frame Wasserstein-1 distance between
    sum-normalised STFT magnitudes (50ms window, 20ms hop), averaged over frames.
7. MLDR: multi-scale loudness dynamic range (DiffVox, arXiv:2504.14735 eq. 14-15) —
    L1 distance of the log ratio between short- and long-window energy envelopes at
    (50ms, 1s) and (100ms, 2s) integration times.
8. Stereo only: MLDR after an energy-preserving mid/side transform.
9. pyFDN only: octave-band RT60 natural-log RMSE.
10. pyFDN only: octave-band energy-decay-curve RMSE in dB.
11. pyFDN only (Götz et al., arXiv:2510.23158): octave-band T30 mean absolute
    percentage error and C50 mean absolute error in dB per sample, plus per-band
    Pearson correlation of both parameters across the dataset.
12. ``--fad``: Fréchet Audio Distance between the target and predicted sets on
    CLAP embeddings (dataset-level, one row in the aggregate).
"""

import math
import multiprocessing
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

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
from pyFDN import MatchEnergyDecay, Response, estimate_rt_bands
from scipy.signal import lfilter

from synth_setter.evaluation import acoustic_parameters

# Column headers load_aggregated_metrics requires of the aggregated-metrics CSVs;
# the write sites below still spell them literally.
AGGREGATED_METRICS_STATS: tuple[str, ...] = ("mean", "std")
type ReverbMetricBackend = Literal["pyfdn"]
type AudioEncodeFn = Callable[[np.ndarray, int], np.ndarray]

# Per-sample columns holding raw octave-band parameters; they feed the dataset-level
# Pearson rows instead of the mean/std aggregate.
_ACOUSTIC_PARAMETER_PREFIX = "acoustic_param/"
_ACOUSTIC_PARAMETER_NAMES: tuple[str, ...] = ("t30", "c50")


def _acoustic_parameter_column(name: str, centre_hz: int, side: str) -> str:
    """Return the per-sample column holding one side's octave-band parameter.

    :param name: Parameter name, ``"t30"`` or ``"c50"``.
    :param centre_hz: Octave-band centre frequency in Hz.
    :param side: ``"target"`` or ``"pred"``.
    :returns: Column name under :data:`_ACOUSTIC_PARAMETER_PREFIX`.
    """
    return f"{_ACOUSTIC_PARAMETER_PREFIX}{name}/{centre_hz}hz/{side}"


ACOUSTIC_PARAMETER_COLUMNS: tuple[str, ...] = tuple(
    _acoustic_parameter_column(name, centre, side)
    for name in _ACOUSTIC_PARAMETER_NAMES
    for centre in acoustic_parameters.BAND_CENTRES_HZ
    for side in ("target", "pred")
)


def subdir_matches_pattern(sample_dir: Path) -> bool:
    """Return ``True`` if ``sample_dir`` contains ``pred.wav`` and ``target.wav``.

    :param sample_dir: Directory to inspect.
    :returns: ``True`` when both audio files are present.
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
    """Compute log-Mel spectrograms for each entry in ``MEL_PARAMS``.

    :param y: Audio waveform, shape ``(C, T)``; multi-channel input is accepted by the
        underlying mel transform.
    :param sample_rate: Sample rate in Hz.
    :returns: One dB-scaled mel spectrogram per ``MEL_PARAMS`` entry.
    """
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


def compute_mss(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    """Return mean multi-scale spectrogram distance between ``target`` and ``pred``.

    :param target: Target audio, shape ``(C, T)``.
    :param pred: Predicted audio, same shape as ``target``.
    :param sample_rate: Sample rate in Hz; governs the mel window and hop lengths.
    :returns: Mean absolute spectrogram difference averaged across mel scales.
    """
    logger.info("Computing MSS...")
    target_specs = compute_mel_specs(target, sample_rate)
    pred_specs = compute_mel_specs(pred, sample_rate)

    dist = 0.0
    for target_spec, pred_spec in zip(target_specs, pred_specs):
        dist += np.mean(np.abs(target_spec - pred_spec))

    dist = dist / len(target_specs)
    return dist


scatter = None


def compute_jtfs(y: np.ndarray, J: int = 10, Q: int = 12) -> np.ndarray:
    """Apply the joint time-frequency scattering transform to ``y``.

    Caches the ``Scattering1D`` object module-wide on the first call; the same instance
    is reused for all subsequent calls regardless of shape changes.

    :param y: Audio waveform array.
    :param J: Log-scale resolution (number of octaves).
    :param Q: Quality factor (wavelets per octave).
    :returns: Scattering coefficients array.
    """
    global scatter
    if scatter is None:
        scatter = Scattering1D(J=J, Q=Q, shape=y.shape[-1])

    return scatter(y)


def compute_jtfs_distance(target: np.ndarray, pred: np.ndarray, J: int = 10, Q: int = 12) -> float:
    """Return mean L1 JTFS distance between ``target`` and ``pred``.

    :param target: Target audio, shape ``(C, T)``.
    :param pred: Predicted audio, same shape as ``target``.
    :param J: Log-scale resolution forwarded to :func:`compute_jtfs`.
    :param Q: Quality factor forwarded to :func:`compute_jtfs`.
    :returns: Mean absolute difference of scattering coefficients.
    """
    logger.info("Computing JTFS...")

    target_jtfs = compute_jtfs(target, J, Q)
    pred_jtfs = compute_jtfs(pred, J, Q)

    dist = np.mean(np.abs(target_jtfs - pred_jtfs))
    return dist


def compute_mfcc(target: np.ndarray, sample_rate: float = 44100.0) -> np.ndarray:
    """Return MFCC features for ``target`` via librosa.

    :param target: Audio waveform; shape ``(T,)`` or ``(C, T)`` — passed through to librosa
        as-is (multi-channel produces ``(C, 20, frames)``).
    :param sample_rate: Sample rate in Hz; governs window and hop lengths.
    :returns: MFCC array; shape ``(20, frames)`` for 1-D input, ``(C, 20, frames)`` for 2-D.
    """
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
    """Return mean absolute element-wise difference between ``a`` and ``b``.

    :param a: First array.
    :param b: Second array; must be the same shape as ``a``.
    :returns: Scalar mean absolute difference.
    """
    return np.mean(np.abs(a - b))


def compute_wmfcc(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    """Return DTW-normalised MFCC distance between ``target`` and ``pred``.

    :param target: Target audio, shape ``(C, T)``.
    :param pred: Predicted audio, same shape as ``target``.
    :param sample_rate: Sample rate in Hz; governs the MFCC window and hop lengths.
    :returns: DTW-normalised L1 distance between MFCC sequences.
    """
    logger.info("Computing wMFCC...")

    target_mfcc = compute_mfcc(target, sample_rate)
    pred_mfcc = compute_mfcc(pred, sample_rate)

    target_mfcc = target_mfcc.reshape(-1, target_mfcc.shape[-1])
    pred_mfcc = pred_mfcc.reshape(-1, pred_mfcc.shape[-1])

    dist = dtw(target_mfcc.T, pred_mfcc.T, dist_method=_l1_distance, distance_only=True)
    return dist.normalizedDistance


pesto_model = None


@torch.no_grad()
def get_pesto_activations(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return PESTO F0 activations for ``target`` and ``pred``, both shape ``(C, T)``.

    Filters to frames where both signals exceed the 0.85 confidence threshold.

    :param target: Target audio.
    :param pred: Predicted audio.
    :param sample_rate: Sample rate in Hz.
    :returns: Tuple ``(target_f0, pred_f0)`` — 1-D arrays of Hz values at confident frames.
    """
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
    """Return mean absolute F0 error at high-confidence PESTO frames.

    :param target: Target audio, shape ``(C, T)``.
    :param pred: Predicted audio, same shape as ``target``.
    :returns: Mean Hz error at frames where both signals exceed the 0.85 confidence threshold.
    """
    logger.info("Computing f0...")
    target_f0, pred_f0 = get_pesto_activations(target, pred)
    return np.mean(np.abs(target_f0 - pred_f0))


def get_stft(y: np.ndarray, sample_rate: float = 44100.0) -> np.ndarray:
    """Return magnitude STFT of ``y``; output shape ``(frames, n_fft // 2 + 1)``.

    :param y: Audio waveform, shape ``(C, T)``; channels are averaged before transform.
    :param sample_rate: Sample rate in Hz; governs window and hop lengths.
    :returns: Magnitude spectrogram, shape ``(frames, n_fft // 2 + 1)``.
    """
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
    """Return the Wasserstein-1 distance between row-normalised histograms.

    :param hist1: Normalised histogram batch, shape ``(frames, bins)``.
    :param hist2: Second batch, same shape as ``hist1``.
    :returns: Per-frame distance, shape ``(frames,)``.
    """
    bin_width = 1 / hist1.shape[-1]
    cdf1 = np.cumsum(hist1, axis=-1)
    cdf2 = np.cumsum(hist2, axis=-1)
    distance = np.sum(np.abs(cdf1 - cdf2), axis=-1) * bin_width
    return distance


def compute_sot(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    """Return mean Sliced Optimal Transport distance between spectrograms.

    :param target: Target audio, shape ``(C, T)``.
    :param pred: Predicted audio, same shape as ``target``.
    :param sample_rate: Sample rate in Hz; governs the STFT window and hop lengths.
    :returns: Mean Wasserstein distance across frequency bins.
    """
    logger.info("Computing SOT...")
    target_stft = get_stft(target, sample_rate)
    pred_stft = get_stft(pred, sample_rate)

    target_stft = target_stft / np.clip(target_stft.sum(axis=-1, keepdims=True), 1e-6, None)
    pred_stft = pred_stft / np.clip(pred_stft.sum(axis=-1, keepdims=True), 1e-6, None)

    dists = batched_wasserstein_distance_np(target_stft, pred_stft)
    return dists.mean()


def _mono_impulse_response(audio: np.ndarray, name: str) -> np.ndarray:
    """Return one finite mono impulse response as a time-major vector.

    :param audio: Channel-first audio expected to have shape ``(1, samples)``.
    :param name: Signal name used in validation errors.
    :returns: The signal's one-dimensional sample vector.
    :raises ValueError: The signal is not finite mono audio.
    """
    array = np.asarray(audio)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError(f"{name} impulse response must be mono with shape (1, samples)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} impulse response must contain only finite values")
    return array[0]


def _paired_mono_impulse_responses(
    target: np.ndarray, pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return a target/prediction mono impulse-response pair.

    :param target: Target channel-first impulse response.
    :param pred: Predicted channel-first impulse response.
    :returns: Time-major target and prediction vectors with equal lengths.
    :raises ValueError: Either signal is invalid or their lengths differ.
    """
    target_ir = _mono_impulse_response(target, "target")
    pred_ir = _mono_impulse_response(pred, "predicted")
    if target_ir.shape != pred_ir.shape:
        raise ValueError("target and predicted impulse responses must have the same sample count")
    return target_ir, pred_ir


def compute_octave_rt60_log_rmse(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0
) -> float:
    """Return log-RMSE between valid paired octave-band RT60 estimates.

    pyFDN returns zero when a band's decay cannot be fitted. Such bands and any
    non-finite estimates are excluded jointly so logarithms cannot contaminate logs.

    :param target: Target mono impulse response, shape ``(1, samples)``.
    :param pred: Predicted mono impulse response, same shape as ``target``.
    :param sample_rate: Sample rate in Hz.
    :returns: Root mean squared natural-log RT60 ratio across valid octave bands.
    :raises ValueError: Shapes, centre frequencies, or fitted bands are invalid.
    """
    target_ir, pred_ir = _paired_mono_impulse_responses(target, pred)
    target_rt, target_centres = estimate_rt_bands(target_ir, sample_rate)
    pred_rt, pred_centres = estimate_rt_bands(pred_ir, sample_rate)
    if not np.array_equal(target_centres, pred_centres):
        raise ValueError("target and predicted octave-band centre frequencies differ")

    valid = np.isfinite(target_rt) & np.isfinite(pred_rt) & (target_rt > 0) & (pred_rt > 0)
    if not valid.any():
        raise ValueError("no valid paired octave-band RT60 estimates")
    log_error = np.log(pred_rt[valid]) - np.log(target_rt[valid])
    return float(np.sqrt(np.mean(log_error**2)))


@torch.no_grad()
def compute_octave_edc_rmse_db(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0
) -> float:
    """Return pyFDN's octave-band energy-decay-curve RMSE in dB.

    :param target: Target mono impulse response, shape ``(1, samples)``.
    :param pred: Predicted mono impulse response, same shape as ``target``.
    :param sample_rate: Sample rate in Hz.
    :returns: RMS dB difference over target-valid octave-band decay frames.
    :raises ValueError: Shapes are invalid or the resulting score is non-finite.
    """
    target_ir, pred_ir = _paired_mono_impulse_responses(target, pred)
    response = Response(
        h=torch.as_tensor(pred_ir[:, np.newaxis, np.newaxis]),
        fs=float(sample_rate),
    )
    value = float(MatchEnergyDecay(target_ir)(response).item())
    if not np.isfinite(value):
        raise ValueError("octave-band energy-decay RMSE must be finite")
    return value


def compute_acoustic_parameter_metrics(
    target: np.ndarray, pred: np.ndarray, sample_rate: float
) -> dict[str, float]:
    """Return Götz et al. T30/C50 errors plus the raw per-band parameters of both sides.

    :param target: Target mono impulse response, shape ``(1, samples)``.
    :param pred: Predicted mono impulse response, same shape as ``target``.
    :param sample_rate: Sample rate in Hz.
    :returns: ``t30_mape``, ``c50_mae_db`` and one ``acoustic_param/...`` entry per
        parameter, octave band and side; unfittable T30 bands read ``NaN``. Invalid
        shapes and a T30 that fits on no band on both sides raise ``ValueError``.
    """
    target_ir, pred_ir = _paired_mono_impulse_responses(target, pred)
    target_t30, _ = acoustic_parameters.octave_band_t30(target_ir, sample_rate)
    pred_t30, _ = acoustic_parameters.octave_band_t30(pred_ir, sample_rate)
    target_c50, _ = acoustic_parameters.octave_band_c50(target_ir, sample_rate)
    pred_c50, _ = acoustic_parameters.octave_band_c50(pred_ir, sample_rate)
    metrics = {
        "t30_mape": acoustic_parameters.t30_mape(target_t30, pred_t30),
        "c50_mae_db": acoustic_parameters.c50_mae_db(target_c50, pred_c50),
    }
    per_band = {"t30": (target_t30, pred_t30), "c50": (target_c50, pred_c50)}
    for name, (target_values, pred_values) in per_band.items():
        bands = zip(acoustic_parameters.BAND_CENTRES_HZ, target_values, pred_values, strict=True)
        for centre, target_value, pred_value in bands:
            metrics[_acoustic_parameter_column(name, centre, "target")] = float(target_value)
            metrics[_acoustic_parameter_column(name, centre, "pred")] = float(pred_value)
    return metrics


def compute_acoustic_parameter_pcc(per_sample: pd.DataFrame) -> dict[str, float]:
    """Return per-band Pearson correlation between target and predicted parameters.

    :param per_sample: Per-sample metrics frame carrying ``acoustic_param/`` columns.
    :returns: ``{"<param>_pcc_<centre>hz": r}`` for every band present in the frame.
    """
    pcc: dict[str, float] = {}
    for name in _ACOUSTIC_PARAMETER_NAMES:
        for centre in acoustic_parameters.BAND_CENTRES_HZ:
            target_column = _acoustic_parameter_column(name, centre, "target")
            if target_column not in per_sample.columns:
                continue
            pred_column = _acoustic_parameter_column(name, centre, "pred")
            pcc[f"{name}_pcc_{centre}hz"] = acoustic_parameters.pearson_correlation(
                per_sample[target_column].to_numpy(dtype=float),
                per_sample[pred_column].to_numpy(dtype=float),
            )
    return pcc


def _load_fad_encoder() -> AudioEncodeFn:
    """Load the repo's CLAP audio encoder for Fréchet Audio Distance.

    :returns: Encoder mapping ``(B, T)`` mono audio at a sample rate to ``(B, D)``.
    """
    from synth_setter.pipeline.data.add_embeddings import load_clap_audio_encoder

    return load_clap_audio_encoder()


def compute_fad(audio_dirs: list[Path], encode: AudioEncodeFn) -> float:
    """Return the Fréchet Audio Distance between the target and predicted sets.

    :param audio_dirs: Sample dirs each containing ``target.wav`` and ``pred.wav``.
    :param encode: Audio encoder mapping ``(B, T)`` mono audio to ``(B, D)`` embeddings.
    :returns: Fréchet distance between Gaussian fits of the two embedding sets.
    """
    embeddings: dict[str, list[np.ndarray]] = {"target": [], "pred": []}
    for sample_dir in audio_dirs:
        for side, rows in embeddings.items():
            with AudioFile(str(sample_dir / f"{side}.wav")) as audio_file:
                mono = audio_file.read(audio_file.frames).mean(axis=0, keepdims=True)
                rows.append(encode(mono, int(audio_file.samplerate)))
    return acoustic_parameters.frechet_distance(
        np.concatenate(embeddings["target"]), np.concatenate(embeddings["pred"])
    )


def compute_rms(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    """Return the cosine similarity of the RMS amplitude envelopes of ``target`` and ``pred``.

    :param target: Target audio, shape ``(C, T)``.
    :param pred: Predicted audio, same shape as ``target``.
    :param sample_rate: Sample rate in Hz; governs window and hop lengths.
    :returns: Cosine similarity in ``[-1, 1]``, or ``0.0`` when either envelope is silent.
    """
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


# (short, long) integration times in ms — DiffVox's ``s_taus``/``l_taus`` for the MLDR loss.
LDR_SCALES_MS: tuple[tuple[float, float], ...] = ((50.0, 1000.0), (100.0, 2000.0))
# Energy floor before the log, matching the reference implementation's ``clamp_min``.
_LDR_ENERGY_FLOOR = 1e-8
# Numerator used by torchcomp's ``ms2coef``.
_TORCHCOMP_MS_TO_COEF = 2200.0


def _one_pole_average(energy: np.ndarray, time_ms: float, sample_rate: float) -> np.ndarray:
    """Return torchcomp's running-average envelope of ``energy`` along the last axis.

    :param energy: Non-negative signal, any leading shape.
    :param time_ms: Integration time in ms, converted with torchcomp's ``ms2coef``.
    :param sample_rate: Sample rate in Hz.
    :returns: Smoothed signal, same shape as ``energy``.
    """
    coef = 1.0 - math.exp(-_TORCHCOMP_MS_TO_COEF / (time_ms * sample_rate))
    return lfilter([coef], [1.0, coef - 1.0], energy, axis=-1)


def _loudness_dynamic_range(
    energy: np.ndarray, short_ms: float, long_ms: float, sample_rate: float
) -> np.ndarray:
    """Return the per-sample log ratio of the short to the delayed long energy envelope.

    The long envelope is advanced by half the integration-time gap (circularly, as the
    reference implementation's ``roll`` does) so both envelopes centre on the same instant.
    The roll is per row: DiffVox rolls the flattened tensor, which bleeds each channel's
    tail into the next channel and departs from the paper's per-signal definition.

    :param energy: Floored squared signal, shape ``(rows, T)``.
    :param short_ms: Short integration time in ms.
    :param long_ms: Long integration time in ms.
    :param sample_rate: Sample rate in Hz.
    :returns: Log loudness dynamic range, shape ``(rows, T)``.
    """
    half_gap_seconds = (long_ms - short_ms) / 1000.0 / 2.0
    align_shift = int(sample_rate * half_gap_seconds)
    short_env = _one_pole_average(energy, short_ms, sample_rate)
    long_env = np.roll(_one_pole_average(energy, long_ms, sample_rate), -align_shift, axis=-1)
    return np.log(short_env) - np.log(long_env)


def compute_mldr(target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0) -> float:
    """Return the multi-scale loudness dynamic range distance between ``target`` and ``pred``.

    Sums, over ``LDR_SCALES_MS``, the mean absolute difference of the two signals' log
    short-to-long energy-envelope ratios (DiffVox eq. 15). Gain-invariant by construction.

    :param target: Target audio, shape ``(C, T)``; channels are scored as independent rows.
    :param pred: Predicted audio, same shape as ``target``.
    :param sample_rate: Sample rate in Hz; governs the envelope time constants.
    :returns: Non-negative distance in natural-log units.
    :raises ValueError: ``target`` and ``pred`` are not two-dimensional arrays of one shape.
    """
    logger.info("Computing MLDR...")
    if target.ndim != 2 or target.shape != pred.shape:
        raise ValueError(
            f"target and pred must share one (C, T) shape; got {target.shape} and {pred.shape}"
        )
    target_energy = np.maximum(np.square(target, dtype=np.float64), _LDR_ENERGY_FLOOR)
    pred_energy = np.maximum(np.square(pred, dtype=np.float64), _LDR_ENERGY_FLOOR)

    dist = 0.0
    for short_ms, long_ms in LDR_SCALES_MS:
        target_ldr = _loudness_dynamic_range(target_energy, short_ms, long_ms, sample_rate)
        pred_ldr = _loudness_dynamic_range(pred_energy, short_ms, long_ms, sample_rate)
        dist += float(np.mean(np.abs(target_ldr - pred_ldr)))
    return dist


def compute_mldr_mid_side(
    target: np.ndarray, pred: np.ndarray, sample_rate: float = 44100.0
) -> float:
    """Return MLDR after an energy-preserving stereo mid/side transform.

    :param target: Target stereo audio with shape ``(2, T)``.
    :param pred: Predicted stereo audio with the same shape as ``target``.
    :param sample_rate: Sample rate in Hz; governs the MLDR envelope time constants.
    :returns: Non-negative mid/side distance in natural-log units.
    :raises ValueError: Inputs are not finite, matching, nonempty stereo arrays.
    """
    if (
        target.ndim != 2
        or target.shape[0] != 2
        or target.shape[-1] == 0
        or target.shape != pred.shape
    ):
        raise ValueError(
            "target and pred must have matching nonempty stereo (2, T) shapes; "
            f"got {target.shape} and {pred.shape}"
        )
    scale = math.sqrt(2.0)
    target_float = np.asarray(target, dtype=np.float64)
    pred_float = np.asarray(pred, dtype=np.float64)
    if not np.isfinite(target_float).all() or not np.isfinite(pred_float).all():
        raise ValueError("target and pred must contain only finite values")
    target_mid_side = np.stack(
        ((target_float[0] + target_float[1]) / scale, (target_float[0] - target_float[1]) / scale)
    )
    pred_mid_side = np.stack(
        ((pred_float[0] + pred_float[1]) / scale, (pred_float[0] - pred_float[1]) / scale)
    )
    return compute_mldr(target_mid_side, pred_mid_side, sample_rate)


def compute_metrics_on_dir(
    audio_dir: Path, renderer_backend: ReverbMetricBackend | None = None
) -> dict[str, float]:
    """Load one target/prediction pair and return applicable audio metrics.

    :param audio_dir: Directory containing ``target.wav`` and ``pred.wav``.
    :param renderer_backend: ``"pyfdn"`` to add impulse-response metrics.
    :returns: Metric scores, including ``mldr_mid_side`` only for stereo pairs.
    :raises ValueError: The target and predicted WAV files have different sample rates.
    """
    with AudioFile(str(audio_dir / "target.wav")) as target_file:
        target = target_file.read(target_file.frames)
        target_sample_rate = float(target_file.samplerate)
    with AudioFile(str(audio_dir / "pred.wav")) as pred_file:
        pred = pred_file.read(pred_file.frames)
        pred_sample_rate = float(pred_file.samplerate)

    if target_sample_rate != pred_sample_rate:
        raise ValueError("target and predicted audio must have the same sample rate")
    metrics = {
        "mss": compute_mss(target, pred, target_sample_rate),
        "wmfcc": compute_wmfcc(target, pred, target_sample_rate),
        "sot": compute_sot(target, pred, target_sample_rate),
        "rms": compute_rms(target, pred, target_sample_rate),
        "mldr": compute_mldr(target, pred, target_sample_rate),
    }
    if target.shape[0] == 2 and pred.shape[0] == 2:
        metrics["mldr_mid_side"] = compute_mldr_mid_side(target, pred, target_sample_rate)
    if renderer_backend == "pyfdn":
        metrics.update(
            {
                "octave_rt60_log_rmse": compute_octave_rt60_log_rmse(
                    target, pred, target_sample_rate
                ),
                "octave_edc_rmse_db": compute_octave_edc_rmse_db(target, pred, target_sample_rate),
            }
        )
        metrics.update(compute_acoustic_parameter_metrics(target, pred, target_sample_rate))
    return metrics


def compute_metrics(
    audio_dirs: list[Path],
    output_dir: Path,
    renderer_backend: ReverbMetricBackend | None = None,
) -> Path:
    """Score each dir in ``audio_dirs`` and write a per-sample CSV to ``output_dir``.

    :param audio_dirs: Sample dirs to score (each must contain ``target.wav`` + ``pred.wav``).
    :param output_dir: Directory for the per-worker ``metrics-<pid>.csv`` output file.
    :param renderer_backend: ``"pyfdn"`` to add impulse-response metrics.
    :returns: Path to the written CSV file.
    """
    idxs = []
    rows = []
    for sample_dir in audio_dirs:
        metrics = compute_metrics_on_dir(sample_dir, renderer_backend)
        rows.append(metrics)
        idxs.append(sample_dir.name.rsplit("_", 1)[-1])

    pid = multiprocessing.current_process().pid

    df = pd.DataFrame(rows, index=idxs)
    metric_file = output_dir / f"metrics-{pid}.csv"
    df.to_csv(metric_file)

    return metric_file


def _aggregate_metrics(
    audio_dirs: list[Path],
    work_dir: Path,
    num_workers: int,
    renderer_backend: ReverbMetricBackend | None = None,
) -> pd.DataFrame:
    """Run the parallel per-sample metrics pass and return the concatenated DataFrame.

    Intermediate per-worker CSVs are written to ``work_dir`` and left there alongside
    the aggregated output.

    :param audio_dirs: Sample dirs to score (each must contain ``target.wav`` + ``pred.wav``).
    :param work_dir: Directory for per-worker intermediate ``metrics-<pid>.csv`` files.
    :param num_workers: ProcessPoolExecutor worker count; capped to ``len(audio_dirs)`` to
        avoid spawning idle processes.
    :param renderer_backend: ``"pyfdn"`` to add impulse-response metrics.
    :returns: Concatenated per-sample metrics DataFrame.
    """
    effective_workers = min(num_workers, len(audio_dirs)) if audio_dirs else 1
    sublist_length = math.ceil(len(audio_dirs) / effective_workers) if audio_dirs else 1
    sublists = [
        s
        for s in (
            audio_dirs[i * sublist_length : (i + 1) * sublist_length]
            for i in range(effective_workers)
        )
        if s
    ]
    metric_dfs = []
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(compute_metrics, sublist, work_dir, renderer_backend)
            for sublist in sublists
        ]
        for future in as_completed(futures):
            metric_file = future.result()
            metric_df = pd.read_csv(metric_file)
            metric_df.set_index(metric_df.columns[0], inplace=True)
            metric_dfs.append(metric_df)
    if not metric_dfs:
        return pd.DataFrame()
    return pd.concat(metric_dfs)


def _is_tool_owned_audio_view(path: Path) -> bool:
    """Return whether ``path`` matches the generated symlink-view layout.

    :param path: Candidate directory containing sample subdirectories.
    :returns: ``True`` for a nonempty tree of sample directories containing only
        ``pred.wav`` and ``target.wav`` symlinks.
    """
    if not path.is_dir() or path.is_symlink():
        return False
    sample_dirs = list(path.iterdir())
    if not sample_dirs:
        return False
    for sample_dir in sample_dirs:
        if not sample_dir.name.startswith("sample_") or not sample_dir.is_dir():
            return False
        entries = list(sample_dir.iterdir())
        if {entry.name for entry in entries} != {"pred.wav", "target.wav"}:
            return False
        if not all(entry.is_symlink() for entry in entries):
            return False
    return True


def _remove_deprecated_metric_outputs(output_dir: Path) -> None:
    """Remove unsupported generated artifacts while preserving unowned content.

    :param output_dir: Metrics directory that may contain unsupported artifacts.
    """
    audio_view = output_dir / "shuffled_audio"
    if not _is_tool_owned_audio_view(audio_view):
        return

    for name in ("aggregated_metrics_shuffled.csv", "shuffle_permutation.csv"):
        path = output_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
    shutil.rmtree(audio_view)


def load_aggregated_metrics(csv_path: Path) -> dict[str, float]:
    """Flatten an aggregated-metrics CSV into ``{"<metric>_<stat>": value}``.

    Reads the layout this module writes: metric names as rows and
    :data:`AGGREGATED_METRICS_STATS` as columns. Keys are returned unprefixed;
    namespacing them is the caller's policy.

    :param csv_path: Aggregated-metrics CSV to read.
    :returns: One entry per ``(metric, stat)`` cell.
    :raises FileNotFoundError: when the producing subprocess returned 0 without writing
        the CSV; surfaced so the silent-success failure mode is loud.
    :raises ValueError: when the CSV is missing a required stat column.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path.name} missing at {csv_path} — the compute_audio_metrics "
            "subprocess returned 0 but did not write the aggregated CSV."
        )
    df = pd.read_csv(csv_path, index_col=0)
    missing = [stat for stat in AGGREGATED_METRICS_STATS if stat not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} missing required stat columns {missing}; got {list(df.columns)}."
        )
    flattened: dict[str, float] = {}
    for metric in df.index:
        for stat in AGGREGATED_METRICS_STATS:
            flattened[f"{metric}_{stat}"] = float(df.at[metric, stat])
    return flattened


@click.command()
@click.argument("audio_dir", type=str)
@click.argument("output_dir", type=str, default="metrics")
@click.option("--num_workers", "-w", type=click.IntRange(min=1), default=8)
@click.option("--renderer-backend", type=click.Choice(["pyfdn"]), default=None)
@click.option("--fad", is_flag=True, help="Add CLAP Fréchet Audio Distance (loads CLAP).")
def main(
    audio_dir: str,
    output_dir: str,
    num_workers: int,
    renderer_backend: ReverbMetricBackend | None,
    fad: bool,
) -> None:
    """Score rendered audio under ``audio_dir`` and write metrics to ``output_dir``.

    Runs the parallel per-sample pass writing ``metrics.csv`` and
    ``aggregated_metrics.csv``. Dataset-level rows (per-band Pearson correlation,
    ``fad_clap``) carry the statistic as ``mean`` and ``NaN`` as ``std``.

    :param audio_dir: Root containing per-sample subdirectories
        (each must have ``pred.wav`` and ``target.wav``).
    :param output_dir: Destination for CSV outputs.
    :param num_workers: Number of parallel worker processes.
    :param renderer_backend: ``"pyfdn"`` to add impulse-response metrics.
    :param fad: Add the CLAP Fréchet Audio Distance between the target and predicted sets.
    :raises ValueError: when no valid sample dirs are found or the input and output
        directories overlap.
    """
    audio_dir_path = Path(audio_dir)
    output_dir_path = Path(output_dir)
    resolved_audio_dir = audio_dir_path.resolve()
    resolved_output_dir = output_dir_path.resolve()
    paths_overlap = (
        resolved_output_dir == resolved_audio_dir
        or resolved_output_dir in resolved_audio_dir.parents
        or resolved_audio_dir in resolved_output_dir.parents
    )
    if paths_overlap:
        raise ValueError(
            "output_dir must not equal, contain, or be contained by audio_dir "
            "to preserve source artifacts."
        )
    os.makedirs(output_dir_path, exist_ok=True)

    audio_dirs = find_possible_subdirs(audio_dir_path)
    if not audio_dirs:
        raise ValueError(
            f"No valid sample dirs with pred.wav and target.wav found under {audio_dir_path}."
        )

    _remove_deprecated_metric_outputs(output_dir_path)
    df = _aggregate_metrics(audio_dirs, output_dir_path, num_workers, renderer_backend)
    df.to_csv(output_dir_path / "metrics.csv")

    is_parameter_column = df.columns.str.startswith(_ACOUSTIC_PARAMETER_PREFIX)
    scalar_metrics = df.loc[:, ~is_parameter_column]
    columnwise_means = scalar_metrics.mean(axis=0)
    columnwise_stds = scalar_metrics.std(axis=0)
    dataset_level = compute_acoustic_parameter_pcc(df.loc[:, is_parameter_column])
    if fad:
        dataset_level["fad_clap"] = compute_fad(audio_dirs, _load_fad_encoder())
    for name, value in dataset_level.items():
        columnwise_means[name] = value
        columnwise_stds[name] = float("nan")
    logger.info("metric means:\n{m}", m=columnwise_means.to_string())
    logger.info("metric stds:\n{s}", s=columnwise_stds.to_string())

    pd.DataFrame({"mean": columnwise_means, "std": columnwise_stds}).to_csv(
        output_dir_path / "aggregated_metrics.csv"
    )


if __name__ == "__main__":
    main()
