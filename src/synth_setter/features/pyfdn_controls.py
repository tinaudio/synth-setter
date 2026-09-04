"""Build normalized temporal control sketches from pyFDN impulse responses."""

import numpy as np
from pyFDN import echo_density, edc, octave_band_filterbank, octave_bands
from scipy.signal import sosfilt

_NUM_INTERVALS = 32
_NUM_OCTAVE_BANDS = 8
_ANALYSIS_WINDOW = 1_024
_ANALYSIS_HOP = 128
_EDC_FLOOR_DB = -60.0
_LOG_HEAD_FRACTION = 0.005
_LOG_RANGE_RATIO = 200.0


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
    if response.size < _ANALYSIS_WINDOW:
        raise ValueError("impulse response must contain at least 1024 samples")

    peak_amplitude = float(np.max(np.abs(response)))
    if peak_amplitude == 0.0:
        raise ValueError("impulse response must have non-zero energy")
    return response / peak_amplitude


def _log_time_edges(num_samples: int) -> np.ndarray:
    """Return the shared fractional log-time edges for one response length.

    :param num_samples: Number of analyzed response samples.
    :returns: Strictly increasing int64 sample edges with shape ``(33,)``.
    :raises ValueError: Rounded edges are not strictly increasing.
    """
    exponents = np.arange(_NUM_INTERVALS, dtype=np.float64) / (_NUM_INTERVALS - 1)
    fractions = np.concatenate(([0.0], _LOG_HEAD_FRACTION * np.power(_LOG_RANGE_RATIO, exponents)))
    fractions[-1] = 1.0
    edges = np.rint(fractions * num_samples).astype(np.int64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("log-time sample edges must be strictly increasing")
    return edges


def _pool_sample_track(track: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Average a sample-aligned track over the shared intervals.

    :param track: One-dimensional sample-aligned descriptor track.
    :param edges: Strict sample edges spanning the track.
    :returns: Float64 interval means with shape ``(32,)``.
    """
    return np.asarray(
        [track[start:end].mean() for start, end in zip(edges[:-1], edges[1:], strict=True)]
    )


def _analysis_frame_centers(num_samples: int) -> np.ndarray:
    starts = np.arange(0, num_samples - _ANALYSIS_WINDOW + 1, _ANALYSIS_HOP)
    return starts + _ANALYSIS_WINDOW // 2


def _pool_frame_track(track: np.ndarray, centers: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Average frame values by the interval containing each frame center.

    :param track: One-dimensional frame descriptor values.
    :param centers: Frame-center sample indices matching ``track``.
    :param edges: Strict sample edges spanning the analyzed response.
    :returns: Float64 interval means with shape ``(32,)``.
    :raises ValueError: A shared interval receives zero frames.
    """
    interval_indices = np.searchsorted(edges, centers, side="right") - 1
    counts = np.bincount(interval_indices, minlength=_NUM_INTERVALS)
    if counts.size != _NUM_INTERVALS or np.any(counts == 0):
        raise ValueError("every log-time interval must receive frames; found zero frames")
    totals = np.bincount(interval_indices, weights=track, minlength=_NUM_INTERVALS)
    return totals / counts


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


def _octave_edc_tracks(response: np.ndarray, sample_rate: float, edges: np.ndarray) -> np.ndarray:
    """Return interval-pooled octave-band EDC tracks normalized to [-1, 1].

    :param response: Unit-peak mono impulse response.
    :param sample_rate: Sample rate in Hz.
    :param edges: Shared log-time sample edges.
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
    return np.stack([_pool_sample_track(track, edges) for track in _normalize_edc_db(edc_db)])


def _echo_density_track(response: np.ndarray, sample_rate: float, edges: np.ndarray) -> np.ndarray:
    """Return pyFDN Abel-Huang density with diffuse density 1 mapped to 0.

    :param response: Unit-peak mono impulse response.
    :param sample_rate: Sample rate in Hz.
    :param edges: Shared log-time sample edges.
    :returns: Float64 normalized density track with shape ``(32,)``.
    """
    _mixing_time, density = echo_density(
        response,
        n=_ANALYSIS_WINDOW,
        fs=sample_rate,
        hop=_ANALYSIS_HOP,
    )
    centers = _analysis_frame_centers(response.size)
    frame_density = np.asarray(density, dtype=np.float64)[centers]
    return _normalize_echo_density(_pool_frame_track(frame_density, centers, edges))


def _spectral_flatness_track(response: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return one spectral-flatness value per temporal interval in [-1, 1].

    :param response: Unit-peak mono impulse response.
    :param edges: Shared log-time sample edges.
    :returns: Float64 normalized flatness track with shape ``(32,)``.
    """
    frames = np.lib.stride_tricks.sliding_window_view(response, _ANALYSIS_WINDOW)[::_ANALYSIS_HOP]
    spectrum = np.fft.rfft(frames * np.hanning(_ANALYSIS_WINDOW), axis=1)
    power = np.square(np.abs(spectrum))
    arithmetic_mean = np.mean(power, axis=1)
    tiny = np.finfo(np.float64).tiny
    geometric_mean = np.exp(np.mean(np.log(np.maximum(power, tiny)), axis=1))
    flatness = np.divide(
        geometric_mean,
        arithmetic_mean,
        out=np.zeros_like(geometric_mean),
        where=arithmetic_mean > 0.0,
    )
    centers = _analysis_frame_centers(response.size)
    pooled = _pool_frame_track(np.clip(flatness, 0.0, 1.0), centers, edges)
    return _normalize_spectral_flatness(pooled)


def extract_reverb_sketch(ir: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return canonical temporal pyFDN controls as contiguous ``float32[10, 32]``.

    Rows 0--7 are 62.5 Hz through 8 kHz octave-band EDCs, clamped to
    ``[-60, 0]`` dB and mapped linearly to ``[-1, 1]``. Row 8 is pyFDN's
    normalized Abel-Huang echo density mapped by ``2d / (1 + d) - 1``, so the
    Gaussian diffuse-field reference ``d=1`` maps to zero without clip-specific
    normalization. Row 9 is spectral flatness mapped linearly from ``[0, 1]``.
    Every row uses sample edges ``s_k = round(f_k N)``, where ``f_0 = 0`` and
    ``f_k = 0.005 * 200 ** ((k - 1) / 31)`` for ``k=1..32``. EDC is pooled by
    sample; echo density and fixed Hann-1024 STFT flatness tracks use hop 128 and
    are pooled by frame center ``start + 512``.

    :param ir: Time-major mono impulse response with shape ``(samples,)``.
    :param sample_rate: Sample rate in Hz; it must support all eight octave bands.
    :returns: Finite C-contiguous controls with shape ``(10, 32)`` in ``[-1, 1]``.
    :raises ValueError: The response or sample rate violates the sketch contract.
    """
    response = _peak_normalized_impulse_response(ir, sample_rate)
    edges = _log_time_edges(response.size)
    sketch = np.vstack(
        (
            _octave_edc_tracks(response, sample_rate, edges),
            _echo_density_track(response, sample_rate, edges),
            _spectral_flatness_track(response, edges),
        )
    )
    if not np.isfinite(sketch).all():
        raise ValueError("reverb sketch must contain only finite values")
    return np.ascontiguousarray(np.clip(sketch, -1.0, 1.0), dtype=np.float32)
