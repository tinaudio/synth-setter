"""Behavior tests for the canonical pyFDN temporal reverb sketch."""

import numpy as np
import pytest

from synth_setter.features.pyfdn_controls import extract_reverb_sketch

_SAMPLE_RATE = 44_100.0


def _broadband_decay() -> np.ndarray:
    rng = np.random.default_rng(2021)
    time = np.arange(int(_SAMPLE_RATE), dtype=np.float64) / _SAMPLE_RATE
    return rng.standard_normal(time.size) * np.exp(-8.0 * time)


def test_extract_reverb_sketch_broadband_decay_returns_canonical_tensor() -> None:
    """The public transform emits the fixed model-boundary shape and dtype."""
    sketch = extract_reverb_sketch(_broadband_decay(), _SAMPLE_RATE)

    assert sketch.shape == (10, 32)
    assert sketch.dtype == np.float32
    assert sketch.flags.c_contiguous
    assert np.isfinite(sketch).all()
    assert np.all(sketch >= -1.0)
    assert np.all(sketch <= 1.0)


def test_extract_reverb_sketch_global_gain_is_invariant() -> None:
    """Finite global gain cannot alter normalized temporal controls."""
    response = _broadband_decay()

    reference = extract_reverb_sketch(response, _SAMPLE_RATE)
    louder = extract_reverb_sketch(response * 1e200, _SAMPLE_RATE)

    np.testing.assert_allclose(louder, reference, atol=1e-6)


def test_extract_reverb_sketch_decay_tracks_are_temporally_monotone() -> None:
    """Equal-duration pooling preserves Schroeder EDC ordering."""
    sketch = extract_reverb_sketch(_broadband_decay(), _SAMPLE_RATE)

    assert np.all(np.diff(sketch[:8], axis=1) <= 0.0)
    assert np.any(np.ptp(sketch[:8], axis=1) > 1.0)


def test_extract_reverb_sketch_echo_density_tracks_diffusion_over_time() -> None:
    """The pyFDN density coordinate distinguishes sparse and diffuse intervals."""
    rng = np.random.default_rng(17)
    response = np.zeros(int(_SAMPLE_RATE), dtype=np.float64)
    response[:11_025:701] = 1.0
    response[11_025:] = rng.standard_normal(response.size - 11_025)
    response *= np.exp(-4.0 * np.arange(response.size) / _SAMPLE_RATE)

    density = extract_reverb_sketch(response, _SAMPLE_RATE)[8]

    assert np.mean(density[:6]) < np.mean(density[12:20])
    assert np.ptp(density) > 0.1


def test_extract_reverb_sketch_flatness_preserves_interval_order() -> None:
    """Flatness remains temporal rather than collapsing intervals into a statistic."""
    rng = np.random.default_rng(23)
    interval_samples = 1_024
    noise = rng.standard_normal(16 * interval_samples)
    time = np.arange(16 * interval_samples, dtype=np.float64) / _SAMPLE_RATE
    tone = np.sin(2.0 * np.pi * 1_000.0 * time)
    response = np.concatenate((noise, tone))

    flatness = extract_reverb_sketch(response, _SAMPLE_RATE)[9]

    assert np.mean(flatness[:16]) > np.mean(flatness[16:]) + 0.5
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
