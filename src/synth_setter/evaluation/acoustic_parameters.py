"""Room-acoustic evaluation metrics of Götz et al. (arXiv:2510.23158).

Octave-band T30 and C50 are compared between target and predicted impulse responses across the
seven bands centred at 125 Hz … 8 kHz; Pearson correlation and Fréchet distance are the dataset-
level statistics the paper reports on top of them.
"""

from __future__ import annotations

import numpy as np
from flareverb.analysis import compute_clarity_parameters
from pyFDN import estimate_rt_bands, octave_band_filterbank, octave_bands
from scipy.linalg import sqrtm
from scipy.signal import sosfilt

_COVARIANCE_JITTER = 1e-6
BAND_CENTRES_HZ: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000, 8000)
# pyFDN addresses the bands as octave offsets from 1 kHz: 2**-3 kHz … 2**3 kHz.
_BAND_START_OCTAVE = -3.0
_BAND_COUNT = len(BAND_CENTRES_HZ)


def octave_band_t30(ir: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return T30 per octave band via pyFDN's 30 dB Schroeder fit.

    :param ir: Mono impulse response, shape ``(samples,)``.
    :param sample_rate: Sample rate in Hz.
    :returns: ``(t30_seconds, centre_hz)``; a band pyFDN cannot fit reads ``NaN``.
    """
    t30, centres = estimate_rt_bands(
        ir, sample_rate, start=_BAND_START_OCTAVE, n=_BAND_COUNT, decay_db=30.0
    )
    return np.where(t30 > 0, t30, np.nan), centres


def octave_band_c50(ir: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return C50 in dB per octave band from flareverb's clarity on band-passed responses.

    :param ir: Mono impulse response, shape ``(samples,)``.
    :param sample_rate: Sample rate in Hz.
    :returns: ``(c50_db, centre_hz)``.
    """
    bands, centres = octave_bands(start=_BAND_START_OCTAVE, n=_BAND_COUNT, fs=sample_rate)
    c50 = np.empty(len(centres))
    for index, sos in enumerate(octave_band_filterbank(bands, sample_rate)):
        band_ir = np.asarray(sosfilt(sos, np.asarray(ir, dtype=float)))
        c50[index] = compute_clarity_parameters(band_ir[np.newaxis, :], sample_rate)[0]
    return c50, centres


def t30_mape(target_t30: np.ndarray, pred_t30: np.ndarray) -> float:
    """Return the mean absolute percentage T30 error over jointly fittable bands.

    :param target_t30: Target T30 per octave band in seconds; unfittable bands ``NaN``.
    :param pred_t30: Predicted T30 per octave band, same shape as ``target_t30``.
    :returns: ``100 * mean(|pred - target| / target)`` across fitted octave bands.
    :raises ValueError: No octave band is fitted on both sides.
    """
    valid = np.isfinite(target_t30) & np.isfinite(pred_t30)
    if not valid.any():
        raise ValueError("no valid paired octave-band T30 estimates")
    relative_error = np.abs(pred_t30[valid] - target_t30[valid]) / target_t30[valid]
    return float(100.0 * np.mean(relative_error))


def c50_mae_db(target_c50: np.ndarray, pred_c50: np.ndarray) -> float:
    """Return the mean absolute C50 error in dB across octave bands.

    :param target_c50: Target C50 per octave band in dB.
    :param pred_c50: Predicted C50 per octave band, same shape as ``target_c50``.
    :returns: Mean over bands of ``|C50_pred - C50_target|``.
    """
    return float(np.mean(np.abs(pred_c50 - target_c50)))


def pearson_correlation(target: np.ndarray, pred: np.ndarray) -> float:
    """Return Pearson's r over pairs where both values are finite.

    :param target: Ground-truth parameter values, shape ``(samples,)``.
    :param pred: Predicted parameter values, same shape as ``target``.
    :returns: Correlation coefficient, or ``NaN`` with fewer than two usable pairs.
    """
    valid = np.isfinite(target) & np.isfinite(pred)
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(target[valid], pred[valid])[0, 1])


def frechet_distance(target: np.ndarray, pred: np.ndarray) -> float:
    """Return the Fréchet distance between Gaussian fits of two embedding sets.

    :param target: Target embeddings, shape ``(n_target, dim)``.
    :param pred: Predicted embeddings, shape ``(n_pred, dim)``.
    :returns: ``‖μ_t − μ_p‖² + Tr(Σ_t + Σ_p − 2·(Σ_t Σ_p)^½)``.
    :raises ValueError: Either set has fewer than two embeddings.
    """
    if target.shape[0] < 2 or pred.shape[0] < 2:
        raise ValueError("Fréchet distance needs at least two embeddings per set")
    mean_diff = target.mean(axis=0) - pred.mean(axis=0)
    target_cov = np.atleast_2d(np.cov(target, rowvar=False))
    pred_cov = np.atleast_2d(np.cov(pred, rowvar=False))
    cov_sqrt = _real_matrix_sqrt(target_cov @ pred_cov)
    if not np.isfinite(cov_sqrt).all():
        # Rank-deficient covariances (few samples) can defeat sqrtm; a diagonal
        # jitter is the standard FID/FAD fallback.
        offset = _COVARIANCE_JITTER * np.eye(target_cov.shape[0])
        cov_sqrt = _real_matrix_sqrt((target_cov + offset) @ (pred_cov + offset))
    distance = mean_diff @ mean_diff + np.trace(target_cov + pred_cov - 2.0 * cov_sqrt)
    # Rounding can push a zero distance a hair negative; a distance is never negative.
    return max(0.0, float(distance))


def _real_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return the principal square root of ``matrix`` with any spurious imaginary part dropped.

    :param matrix: Square matrix, a product of two covariance matrices.
    :returns: Real-valued square root; non-finite entries signal a failed decomposition.
    """
    # Singular covariances yield tiny imaginary parts that carry no information.
    return np.real(np.asarray(sqrtm(matrix)))
