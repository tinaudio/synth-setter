"""Behavior tests for pyFDN reverb-control descriptors."""

import numpy as np
import pytest

from synth_setter.features.pyfdn_controls import (
    MODAL_EXCITATION_QUANTILE_PROBABILITIES,
    extract_modal_excitation_quantiles_db,
    extract_octave_edc_db,
    extract_octave_rt60_seconds,
)

_SAMPLE_RATE = 44_100.0
_EXPECTED_OCTAVE_CENTRES_HZ = np.array(
    [62.5, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0]
)


def _broadband_decay() -> np.ndarray:
    rng = np.random.default_rng(2021)
    time = np.arange(int(_SAMPLE_RATE), dtype=np.float64) / _SAMPLE_RATE
    return rng.standard_normal(time.size) * np.exp(-20.0 * time)


def test_extract_octave_rt60_seconds_broadband_decay_returns_valid_bands() -> None:
    """A finite broadband decay yields one positive RT60 per standard octave band."""
    rt60, centres = extract_octave_rt60_seconds(_broadband_decay(), _SAMPLE_RATE)

    np.testing.assert_array_equal(centres, _EXPECTED_OCTAVE_CENTRES_HZ)
    assert rt60.shape == (8,)
    assert rt60.dtype == np.float64
    assert centres.dtype == np.float64
    assert np.isfinite(rt60).all()
    np.testing.assert_allclose(
        rt60,
        [0.2591, 0.3633, 0.3529, 0.3435, 0.3536, 0.3504, 0.3480, 0.3565],
        atol=1e-4,
    )
    assert (rt60 > 0.0).all()


def test_extract_octave_rt60_seconds_global_gain_is_invariant() -> None:
    """Finite high-amplitude scaling cannot alter RT60 estimates."""
    reference, _centres = extract_octave_rt60_seconds(_broadband_decay(), _SAMPLE_RATE)
    scaled, _centres = extract_octave_rt60_seconds(_broadband_decay() * 1e200, _SAMPLE_RATE)

    np.testing.assert_allclose(scaled, reference, rtol=1e-12)


def test_extract_octave_edc_db_broadband_decay_returns_normalized_curves() -> None:
    """Each octave-band EDC starts at zero dB and decreases over the IR."""
    edc_db, centres = extract_octave_edc_db(_broadband_decay(), _SAMPLE_RATE)

    np.testing.assert_array_equal(centres, _EXPECTED_OCTAVE_CENTRES_HZ)
    assert edc_db.shape == (8, int(_SAMPLE_RATE))
    assert edc_db.dtype == np.float64
    assert centres.dtype == np.float64
    np.testing.assert_allclose(edc_db[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(edc_db[3, [11_025, 22_050]], [-42.84, -87.87], atol=0.01)
    assert np.all(np.diff(edc_db, axis=1) <= 0.0)
    assert np.isfinite(edc_db).all()


def test_extract_octave_edc_db_silent_response_raises() -> None:
    """Silence has no energy-decay curve to normalize."""
    with pytest.raises(ValueError, match="non-zero energy"):
        extract_octave_edc_db(np.zeros(1024), _SAMPLE_RATE)


def test_extract_octave_edc_db_large_finite_response_remains_finite() -> None:
    """Finite high-amplitude samples cannot overflow the normalized EDC."""
    edc_db, _centres = extract_octave_edc_db(np.full(1024, 1e200), _SAMPLE_RATE)

    assert np.isfinite(edc_db).all()


def test_extract_octave_edc_db_without_supported_band_raises() -> None:
    """A sample rate below the lowest octave band fails with a domain error."""
    with pytest.raises(ValueError, match="no supported octave bands"):
        extract_octave_edc_db(np.ones(1024), 50.0)


def test_extract_modal_excitation_quantiles_db_unequal_modes_returns_expected_values() -> None:
    """Quantiles retain the spread and strong-mode tail of modal excitation."""
    quantiles = extract_modal_excitation_quantiles_db(np.array([0.1, 1.0, 10.0, 100.0]))

    assert quantiles.dtype == np.float64
    np.testing.assert_allclose(quantiles, [-27.0, -15.0, 15.0, 24.0, 27.0, 29.4])


def test_extract_modal_excitation_quantiles_db_zero_mode_uses_finite_floor() -> None:
    """An unexcited mode reaches the finite floor without contaminating quantiles."""
    quantiles = extract_modal_excitation_quantiles_db(np.array([0.0, 1.0]))

    np.testing.assert_allclose(quantiles, [-54.0, -30.0, 30.0, 48.0, 54.0, 58.8])


def test_extract_modal_excitation_quantiles_db_global_gain_is_invariant() -> None:
    """A global gain change cannot alter relative modal excitation."""
    reference = extract_modal_excitation_quantiles_db(np.array([0.1, 1.0, 10.0, 100.0]))
    louder = extract_modal_excitation_quantiles_db(np.array([0.2, 2.0, 20.0, 200.0]))

    np.testing.assert_allclose(louder, reference, atol=1e-10)


def test_extract_octave_rt60_seconds_stereo_response_raises() -> None:
    """Channel-first stereo audio is not accepted as a mono impulse response."""
    with pytest.raises(ValueError, match="shape"):
        extract_octave_rt60_seconds(np.ones((2, 1024)), _SAMPLE_RATE)


def test_extract_octave_edc_db_complex_response_raises() -> None:
    """A complex response cannot silently lose its imaginary component."""
    with pytest.raises(ValueError, match="real-valued"):
        extract_octave_edc_db(np.array([1.0 + 1.0j]), _SAMPLE_RATE)


def test_extract_octave_edc_db_nonfinite_response_raises() -> None:
    """A non-finite impulse response fails before filtering."""
    with pytest.raises(ValueError, match="finite"):
        extract_octave_edc_db(np.array([1.0, np.nan]), _SAMPLE_RATE)


def test_extract_octave_rt60_seconds_invalid_sample_rate_raises() -> None:
    """A non-positive sample rate cannot define octave bands."""
    with pytest.raises(ValueError, match="sample_rate"):
        extract_octave_rt60_seconds(np.ones(1024), 0.0)


def test_extract_modal_excitation_quantiles_db_negative_magnitude_raises() -> None:
    """Residue magnitudes cannot be negative."""
    with pytest.raises(ValueError, match="non-negative"):
        extract_modal_excitation_quantiles_db(np.array([1.0, -1.0]))


def test_modal_excitation_quantile_probabilities_pin_control_coordinates() -> None:
    """The public quantile coordinates remain ordered and omit the fixed median."""
    np.testing.assert_array_equal(
        MODAL_EXCITATION_QUANTILE_PROBABILITIES,
        np.array([0.05, 0.25, 0.75, 0.90, 0.95, 0.99]),
    )
