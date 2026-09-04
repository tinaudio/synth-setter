"""Extract decay and modal-excitation descriptors from pyFDN responses.

Typical usage::

    rt60, octave_centres = extract_octave_rt60_seconds(impulse_response, sample_rate)
    edc_db, _ = extract_octave_edc_db(impulse_response, sample_rate)
    excitation = extract_modal_excitation_quantiles_db(residue_magnitudes)
"""

import numpy as np
from pyFDN import edc, estimate_rt_bands, octave_band_filterbank, octave_bands
from scipy.signal import sosfilt

MODAL_EXCITATION_FLOOR_DB = -120.0
MODAL_EXCITATION_QUANTILE_PROBABILITIES = np.array(
    [0.05, 0.25, 0.75, 0.90, 0.95, 0.99], dtype=np.float64
)
MODAL_EXCITATION_QUANTILE_PROBABILITIES.setflags(write=False)


def _validated_impulse_response(ir: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return a finite, non-empty mono impulse response.

    :param ir: Time-major mono impulse response with shape ``(samples,)``.
    :param sample_rate: Positive sample rate in Hz.
    :returns: Float64 impulse response.
    :raises ValueError: The signal or sample rate violates the descriptor contract.
    """
    raw_response = np.asarray(ir)
    if np.iscomplexobj(raw_response):
        raise ValueError("impulse response must be real-valued")
    response = np.asarray(raw_response, dtype=np.float64)
    if response.ndim != 1 or response.size == 0:
        raise ValueError("impulse response must have shape (samples,) with at least one sample")
    if not np.isfinite(response).all():
        raise ValueError("impulse response must contain only finite values")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be finite and positive")
    return response


def _peak_normalized_impulse_response(ir: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return validated audio scaled to unit peak.

    :param ir: Time-major mono impulse response with shape ``(samples,)``.
    :param sample_rate: Positive sample rate in Hz.
    :returns: Float64 unit-peak impulse response.
    :raises ValueError: The signal is silent or otherwise invalid.
    """
    response = _validated_impulse_response(ir, sample_rate)
    peak_amplitude = float(np.max(np.abs(response)))
    if peak_amplitude == 0.0:
        raise ValueError("impulse response must have non-zero energy")
    return response / peak_amplitude


def extract_octave_rt60_seconds(
    ir: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid pyFDN T30-extrapolated RT60 values in octave bands.

    :param ir: Time-major mono impulse response with shape ``(samples,)``.
    :param sample_rate: Positive sample rate in Hz.
    :returns: Float64 RT60 seconds and corresponding octave center frequencies in Hz.
    :raises ValueError: Input is invalid or any octave band cannot be fitted.
    """
    response = _peak_normalized_impulse_response(ir, sample_rate)
    rt60, centres_hz = estimate_rt_bands(response, sample_rate)
    rt60 = np.asarray(rt60, dtype=np.float64)
    centres_hz = np.asarray(centres_hz, dtype=np.float64)
    if rt60.size == 0:
        raise ValueError("sample rate has no supported octave bands")
    valid = np.isfinite(rt60) & (rt60 > 0.0)
    if not valid.all():
        raise ValueError("every octave band must have a valid positive RT60 estimate")
    return rt60, centres_hz


def extract_octave_edc_db(ir: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return per-band normalized Schroeder energy-decay curves in dB.

    :param ir: Time-major mono impulse response with shape ``(samples,)``.
    :param sample_rate: Positive sample rate in Hz.
    :returns: Float64 EDCs with shape ``(bands, samples)`` and octave centers in Hz.
    :raises ValueError: Input is invalid or a band has no energy to normalize.
    """
    response = _peak_normalized_impulse_response(ir, sample_rate)
    bands, centres_hz = octave_bands(fs=sample_rate)
    if centres_hz.size == 0:
        raise ValueError("sample rate has no supported octave bands")
    filtered = np.stack(
        [sosfilt(sos, response) for sos in octave_band_filterbank(bands, sample_rate)]
    )
    energy_decay = edc(filtered, axis=1)
    initial_energy = energy_decay[:, :1]
    if np.any(initial_energy <= 0.0):
        raise ValueError("every octave band must have non-zero energy")

    normalized = energy_decay / initial_energy
    finite_floor = np.finfo(np.float64).tiny
    edc_db = 10.0 * np.log10(np.maximum(normalized, finite_floor))
    return edc_db, np.asarray(centres_hz, dtype=np.float64)


def extract_modal_excitation_quantiles_db(
    modal_excitation_magnitudes: np.ndarray,
) -> np.ndarray:
    """Return median-centered quantiles of modal residue magnitudes.

    The input contains one magnitude per pole, including both members of each
    conjugate pair. Excitations below :data:`MODAL_EXCITATION_FLOOR_DB` relative
    to the strongest mode are floored before median centering.

    :param modal_excitation_magnitudes: Non-negative residue magnitudes shaped
        ``(modes,)``.
    :returns: Float64 dB quantiles at
        :data:`MODAL_EXCITATION_QUANTILE_PROBABILITIES`.
    :raises ValueError: Magnitudes are invalid or every mode is unexcited.
    """
    raw_magnitudes = np.asarray(modal_excitation_magnitudes)
    if np.iscomplexobj(raw_magnitudes):
        raise ValueError("modal excitation magnitudes must be real")
    magnitudes = np.asarray(raw_magnitudes, dtype=np.float64)
    if magnitudes.ndim != 1 or magnitudes.size == 0:
        raise ValueError("modal excitation magnitudes must have shape (modes,)")
    if not np.isfinite(magnitudes).all() or np.any(magnitudes < 0.0):
        raise ValueError("modal excitation magnitudes must be finite and non-negative")

    peak_magnitude = float(np.max(magnitudes))
    if peak_magnitude == 0.0:
        raise ValueError("at least one mode must have positive excitation")
    relative_floor = 10.0 ** (MODAL_EXCITATION_FLOOR_DB / 20.0)
    excitation_db = 20.0 * np.log10(np.maximum(magnitudes / peak_magnitude, relative_floor))
    excitation_db -= np.median(excitation_db)
    return np.quantile(excitation_db, MODAL_EXCITATION_QUANTILE_PROBABILITIES)
