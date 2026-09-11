"""Behavior tests for the canonical pyFDN temporal reverb sketch."""

import numpy as np
import pytest

from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC
from synth_setter.features.pyfdn_controls import (
    _log_time_edges,
    _normalize_echo_density,
    _normalize_edc_db,
    _normalize_spectral_flatness,
    extract_reverb_sketch,
)

_SAMPLE_RATE = 44_100.0
_CANONICAL_SAMPLES = 176_400
_CANONICAL_LOG_EDGES = np.array(
    [
        0,
        882,
        1_046,
        1_241,
        1_473,
        1_747,
        2_073,
        2_459,
        2_918,
        3_462,
        4_107,
        4_872,
        5_780,
        6_858,
        8_136,
        9_653,
        11_452,
        13_586,
        16_118,
        19_123,
        22_687,
        26_916,
        31_932,
        37_884,
        44_945,
        53_323,
        63_261,
        75_052,
        89_041,
        105_638,
        125_327,
        148_687,
        176_400,
    ],
    dtype=np.int64,
)


def _broadband_decay() -> np.ndarray:
    rng = np.random.default_rng(2021)
    time = np.arange(_CANONICAL_SAMPLES, dtype=np.float64) / _SAMPLE_RATE
    return rng.standard_normal(time.size) * np.exp(-8.0 * time)


def _canonical_patch(rt_seconds: float) -> np.ndarray:
    params, _ = PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(np.random.default_rng(3002))
    params["post_delay.rt_dc_seconds"] = rt_seconds
    params["post_delay.rt_nyquist_seconds"] = rt_seconds
    return PyFDNRenderer().render(params)[0]


def test_extract_reverb_sketch_broadband_decay_returns_canonical_tensor() -> None:
    """The public transform emits the fixed model-boundary shape and dtype."""
    sketch = extract_reverb_sketch(_broadband_decay(), _SAMPLE_RATE)

    assert sketch.shape == (10, 32)
    assert sketch.dtype == np.float32
    assert sketch.flags.c_contiguous
    assert np.isfinite(sketch).all()
    assert np.all(sketch >= -1.0)
    assert np.all(sketch <= 1.0)


def test_log_time_edges_canonical_render_matches_contract_table() -> None:
    """The four-second 44.1 kHz render has the pinned fractional edges."""
    np.testing.assert_array_equal(_log_time_edges(_CANONICAL_SAMPLES), _CANONICAL_LOG_EDGES)


def test_log_time_edges_nonmonotone_rounding_raises() -> None:
    """A response too short to preserve distinct rounded edges fails loudly."""
    with pytest.raises(ValueError, match="strictly increasing"):
        _log_time_edges(100)


def test_extract_reverb_sketch_zero_frame_interval_raises() -> None:
    """A monotone grid without an STFT center in every interval fails loudly."""
    response = np.random.default_rng(31).standard_normal(100_000)

    with pytest.raises(ValueError, match="zero frames"):
        extract_reverb_sketch(response, _SAMPLE_RATE)


def test_extract_reverb_sketch_global_gain_is_invariant() -> None:
    """Finite global gain cannot alter normalized temporal controls."""
    response = _broadband_decay()

    reference = extract_reverb_sketch(response, _SAMPLE_RATE)
    louder = extract_reverb_sketch(response * 1e200, _SAMPLE_RATE)

    np.testing.assert_allclose(louder, reference, atol=1e-6)


def test_extract_reverb_sketch_decay_tracks_are_temporally_monotone() -> None:
    """Log-time pooling preserves Schroeder EDC ordering."""
    sketch = extract_reverb_sketch(_broadband_decay(), _SAMPLE_RATE)

    assert np.all(np.diff(sketch[:8], axis=1) <= 0.0)
    assert np.any(np.ptp(sketch[:8], axis=1) > 1.0)


def test_extract_reverb_sketch_frequency_dependent_decay_preserves_band_order() -> None:
    """Distinct octave-band decays remain ordered from 62.5 Hz through 8 kHz."""
    time = np.arange(_CANONICAL_SAMPLES, dtype=np.float64) / _SAMPLE_RATE
    centres_hz = [62.5, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0]
    decay_rates = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    response = np.sum(
        [
            np.sin(2.0 * np.pi * centre_hz * time) * np.exp(-decay_rate * time)
            for centre_hz, decay_rate in zip(centres_hz, decay_rates, strict=True)
        ],
        axis=0,
    )

    octave_edc = extract_reverb_sketch(response, _SAMPLE_RATE)[:8]

    assert np.all(np.diff(octave_edc[:, 8]) < 0.0)


def test_normalize_edc_db_maps_documented_landmarks() -> None:
    """Zero and minus sixty dB map to the model-space endpoints."""
    normalized = _normalize_edc_db(np.array([0.0, -30.0, -60.0, -90.0]))

    np.testing.assert_array_equal(normalized, [1.0, 0.0, -1.0, -1.0])


def test_normalize_echo_density_maps_diffuse_reference_to_zero() -> None:
    """The Abel-Huang diffuse-field reference is the model-space midpoint."""
    normalized = _normalize_echo_density(np.array([0.0, 1.0]))

    np.testing.assert_array_equal(normalized, [-1.0, 0.0])


def test_normalize_spectral_flatness_maps_unit_interval_to_model_range() -> None:
    """Spectral-flatness endpoints map linearly to model-space endpoints."""
    normalized = _normalize_spectral_flatness(np.array([0.0, 0.5, 1.0]))

    np.testing.assert_array_equal(normalized, [-1.0, 0.0, 1.0])


def test_extract_reverb_sketch_short_rt_spans_at_least_four_edc_tokens() -> None:
    """A 100 ms canonical patch retains early decay resolution."""
    edc_tracks = extract_reverb_sketch(_canonical_patch(0.1), _SAMPLE_RATE)[:8]

    changing_boundaries = np.any(np.abs(np.diff(edc_tracks, axis=1)) > 1e-4, axis=0)
    assert np.count_nonzero(changing_boundaries) >= 4


def test_extract_reverb_sketch_canonical_echo_density_builds_up_early() -> None:
    """A canonical patch preserves non-constant density in the first tokens."""
    density = extract_reverb_sketch(_canonical_patch(1.0), _SAMPLE_RATE)[8]

    assert np.ptp(density[:8]) > 0.01


def test_extract_reverb_sketch_same_response_is_bit_exact() -> None:
    """Repeated extraction from one response is bit-exact."""
    response = _canonical_patch(1.0)

    first = extract_reverb_sketch(response, _SAMPLE_RATE)
    second = extract_reverb_sketch(response, _SAMPLE_RATE)

    np.testing.assert_array_equal(second, first)


def test_extract_reverb_sketch_echo_density_tracks_diffusion_over_time() -> None:
    """The pyFDN density coordinate distinguishes sparse and diffuse intervals."""
    rng = np.random.default_rng(17)
    response = np.zeros(_CANONICAL_SAMPLES, dtype=np.float64)
    response[:11_025:701] = 1.0
    response[11_025:] = rng.standard_normal(response.size - 11_025)
    response *= np.exp(-4.0 * np.arange(response.size) / _SAMPLE_RATE)

    density = extract_reverb_sketch(response, _SAMPLE_RATE)[8]

    assert np.mean(density[:6]) < np.mean(density[12:20])
    assert np.ptp(density) > 0.1


def test_extract_reverb_sketch_flatness_preserves_interval_order() -> None:
    """Flatness remains temporal rather than collapsing intervals into a statistic."""
    rng = np.random.default_rng(23)
    half_samples = _CANONICAL_SAMPLES // 2
    noise = rng.standard_normal(half_samples)
    time = np.arange(half_samples, dtype=np.float64) / _SAMPLE_RATE
    tone = np.sin(2.0 * np.pi * 1_000.0 * time)
    response = np.concatenate((noise, tone))

    flatness = extract_reverb_sketch(response, _SAMPLE_RATE)[9]

    assert np.mean(flatness[:16]) > np.mean(flatness[16:]) + 0.2
    assert np.ptp(flatness) > 0.5


@pytest.mark.parametrize(
    ("response", "sample_rate", "message"),
    [
        (np.ones((2, 1_024)), _SAMPLE_RATE, "shape"),
        (np.array([], dtype=np.float64), _SAMPLE_RATE, "at least one sample"),
        (np.zeros(1_024), _SAMPLE_RATE, "non-zero energy"),
        (np.array([1.0, np.nan]), _SAMPLE_RATE, "finite"),
        (np.array([1.0 + 1.0j] * 1_024), _SAMPLE_RATE, "real-valued"),
        (np.ones(1_024), 0.0, "sample_rate"),
        (np.ones(1_024), 50.0, "eight octave bands"),
        (np.ones(1_023), _SAMPLE_RATE, "at least 1024 samples"),
    ],
)
def test_extract_reverb_sketch_invalid_response_raises(
    response: np.ndarray, sample_rate: float, message: str
) -> None:
    """Invalid inputs fail at the public feature boundary.

    :param response: Candidate mono impulse response.
    :param sample_rate: Candidate sample rate in Hz.
    :param message: Expected fragment of the validation error.
    """
    with pytest.raises(ValueError, match=message):
        extract_reverb_sketch(response, sample_rate)
