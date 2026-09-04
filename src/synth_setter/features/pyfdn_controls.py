"""Build normalized temporal control sketches from pyFDN impulse responses."""

import numpy as np
from pyFDN import echo_density, edc, octave_band_filterbank, octave_bands
from scipy.signal import sosfilt

_NUM_INTERVALS = 32
_NUM_OCTAVE_BANDS = 8
_ECHO_DENSITY_WINDOW = 1_024
_ECHO_DENSITY_HOP = 500
_EDC_FLOOR_DB = -60.0


def _peak_normalized_impulse_response(ir: np.ndarray, sample_rate: float) -> np.ndarray:
    """Validate mono audio and scale it to unit peak.

    :param ir: Time-major mono impulse response with shape ``(samples,)``.
    :param sample_rate: Sample rate in Hz; it must support all eight octave bands.
    :returns: Finite float64 unit-peak impulse response.
    :raises ValueError: The response or sample rate violates the sketch contract.
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
    if response.size < _ECHO_DENSITY_WINDOW:
        raise ValueError("impulse response must contain at least 1024 samples")

    peak_amplitude = float(np.max(np.abs(response)))
    if peak_amplitude == 0.0:
        raise ValueError("impulse response must have non-zero energy")
    return response / peak_amplitude


def _pool_intervals(track: np.ndarray) -> np.ndarray:
    """Average a time-major track over equal-duration intervals.

    :param track: One-dimensional sample-aligned descriptor track.
    :returns: Float64 interval means with shape ``(32,)``.
    """
    return np.asarray([interval.mean() for interval in np.array_split(track, _NUM_INTERVALS)])


def _normalize_edc_db(edc_db: np.ndarray) -> np.ndarray:
    """Map the fixed EDC decibel range to model space.

    :param edc_db: Energy decay in dB relative to initial band energy.
    :returns: Values clamped below -60 dB and mapped to ``[-1, 1]``.
    """
    return 1.0 + np.clip(edc_db, _EDC_FLOOR_DB, 0.0) / 30.0


def _normalize_echo_density(density: np.ndarray) -> np.ndarray:
    """Map non-negative Abel-Huang density to bounded model space.

    :param density: Echo density normalized to the Gaussian reference.
    :returns: Values in ``[-1, 1)`` with diffuse density one mapped to zero.
    """
    return 2.0 * density / (1.0 + density) - 1.0


def _normalize_spectral_flatness(flatness: np.ndarray) -> np.ndarray:
    """Map spectral flatness to model space.

    :param flatness: Spectral flatness values nominally in ``[0, 1]``.
    :returns: Clamped values mapped linearly to ``[-1, 1]``.
    """
    return 2.0 * np.clip(flatness, 0.0, 1.0) - 1.0


def _octave_edc_tracks(response: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return interval-pooled octave-band EDC tracks normalized to [-1, 1].

    :param response: Unit-peak mono impulse response.
    :param sample_rate: Sample rate in Hz.
    :returns: Float64 normalized EDC tracks with shape ``(8, 32)``.
    :raises ValueError: The sample rate omits a band or a band has zero energy.
    """
    bands, centres_hz = octave_bands(fs=sample_rate)
    if centres_hz.size != _NUM_OCTAVE_BANDS:
        raise ValueError("sample_rate must support all eight octave bands")

    filtered = np.stack(
        [sosfilt(sos, response) for sos in octave_band_filterbank(bands, sample_rate)]
    )
    energy_decay = edc(filtered, axis=1)
    initial_energy = energy_decay[:, :1]
    if np.any(initial_energy <= 0.0):
        raise ValueError("every octave band must have non-zero energy")

    normalized = energy_decay / initial_energy
    edc_db = 10.0 * np.log10(np.maximum(normalized, np.finfo(np.float64).tiny))
    return np.stack([_pool_intervals(track) for track in _normalize_edc_db(edc_db)])


def _echo_density_track(response: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return pyFDN Abel-Huang density with diffuse density 1 mapped to 0.

    :param response: Unit-peak mono impulse response.
    :param sample_rate: Sample rate in Hz.
    :returns: Float64 normalized density track with shape ``(32,)``.
    """
    _mixing_time, density = echo_density(
        response,
        n=_ECHO_DENSITY_WINDOW,
        fs=sample_rate,
        hop=_ECHO_DENSITY_HOP,
    )
    pooled = _pool_intervals(np.asarray(density, dtype=np.float64))
    return _normalize_echo_density(pooled)


def _spectral_flatness_track(response: np.ndarray) -> np.ndarray:
    """Return one spectral-flatness value per temporal interval in [-1, 1].

    :param response: Unit-peak mono impulse response.
    :returns: Float64 normalized flatness track with shape ``(32,)``.
    """
    flatness = []
    tiny = np.finfo(np.float64).tiny
    for interval in np.array_split(response, _NUM_INTERVALS):
        spectrum = np.fft.rfft(interval * np.hanning(interval.size))
        power = np.square(np.abs(spectrum))
        arithmetic_mean = float(np.mean(power))
        if arithmetic_mean == 0.0:
            flatness.append(0.0)
            continue
        geometric_mean = float(np.exp(np.mean(np.log(np.maximum(power, tiny)))))
        flatness.append(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))
    return _normalize_spectral_flatness(np.asarray(flatness))


def extract_reverb_sketch(ir: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return canonical temporal pyFDN controls as contiguous ``float32[10, 32]``.

    Rows 0--7 are 62.5 Hz through 8 kHz octave-band EDCs, clamped to
    ``[-60, 0]`` dB and mapped linearly to ``[-1, 1]``. Row 8 is pyFDN's
    normalized Abel-Huang echo density mapped by ``2d / (1 + d) - 1``, so the
    Gaussian diffuse-field reference ``d=1`` maps to zero without clip-specific
    normalization. Row 9 is spectral flatness mapped linearly from ``[0, 1]``.
    Every row uses the same 32 equal-duration temporal intervals. Echo density
    uses pyFDN's fixed 1024-sample Hann window and 500-sample sparse-analysis hop.

    :param ir: Time-major mono impulse response with shape ``(samples,)``.
    :param sample_rate: Sample rate in Hz; it must support all eight octave bands.
    :returns: Finite C-contiguous controls with shape ``(10, 32)`` in ``[-1, 1]``.
    :raises ValueError: The response or sample rate violates the sketch contract.
    """
    response = _peak_normalized_impulse_response(ir, sample_rate)
    sketch = np.vstack(
        (
            _octave_edc_tracks(response, sample_rate),
            _echo_density_track(response, sample_rate),
            _spectral_flatness_track(response),
        )
    )
    if not np.isfinite(sketch).all():
        raise ValueError("reverb sketch must contain only finite values")
    return np.ascontiguousarray(np.clip(sketch, -1.0, 1.0), dtype=np.float32)
