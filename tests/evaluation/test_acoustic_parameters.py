"""Unit tests for ``synth_setter.evaluation.acoustic_parameters``."""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.evaluation import acoustic_parameters as ap
from tests.helpers.audio_utils import noise

_SR = 44100
_PAPER_BAND_CENTRES = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)


def _decaying_noise(rt60: float, seconds: float = 1.5, seed: int = 0) -> np.ndarray:
    """Return a mono ``(1, samples)`` noise impulse response with a known RT60.

    :param rt60: Reverberation time in seconds of the exponential envelope.
    :param seconds: Length of the response in seconds.
    :param seed: RNG seed for the noise carrier.
    :return: Channel-first float32 impulse response.
    """
    samples = round(seconds * _SR)
    t = np.arange(samples) / _SR
    envelope = np.exp(-6.91 * t / rt60)
    return (noise(1, samples, seed=seed) * envelope).astype(np.float32)


def test_octave_band_t30_reports_paper_band_centres() -> None:
    """The seven octave bands of Götz et al.

    run from 125 Hz to 8 kHz.
    """
    _t30, centres = ap.octave_band_t30(_decaying_noise(0.5)[0], _SR)

    assert tuple(centres) == _PAPER_BAND_CENTRES


def test_octave_band_t30_known_decay_matches_rt60() -> None:
    """A 0.5 s exponential decay yields ~0.5 s T30 in every band."""
    t30, _centres = ap.octave_band_t30(_decaying_noise(0.5)[0], _SR)

    assert t30 == pytest.approx(np.full(7, 0.5), rel=0.15)


def test_octave_band_c50_reports_paper_band_centres() -> None:
    """C50 is evaluated over the same seven octave bands as T30."""
    _c50, centres = ap.octave_band_c50(_decaying_noise(0.5)[0], _SR)

    assert tuple(centres) == _PAPER_BAND_CENTRES


def test_octave_band_c50_known_decay_matches_analytic_clarity() -> None:
    """A 0.5 s exponential decay has C50 ≈ 10·log10(exp(0.1·a) − 1) ≈ 4.7 dB.

    The noise carrier leaves few independent samples in a low band's 50 ms early window, so per-
    band clarity scatters by a couple of dB around the analytic value.
    """
    c50, _centres = ap.octave_band_c50(_decaying_noise(0.5)[0], _SR)

    assert c50 == pytest.approx(np.full(7, 4.74), abs=3.0)


def test_t30_mape_identical_bands_returns_zero() -> None:
    """Bands scored against themselves have zero percentage error."""
    t30 = np.array([0.4, 0.5, 0.6])

    assert ap.t30_mape(t30, t30) == pytest.approx(0.0)


def test_t30_mape_handcrafted_case() -> None:
    """|0.6−0.5|/0.5 = 20 % and |0.9−1.0|/1.0 = 10 % average to 15 %."""
    assert ap.t30_mape(np.array([0.5, 1.0]), np.array([0.6, 0.9])) == pytest.approx(15.0)


def test_t30_mape_skips_bands_unfitted_on_either_side() -> None:
    """A NaN band on either side is excluded rather than propagated."""
    target = np.array([0.5, np.nan, 1.0])
    pred = np.array([0.6, 0.7, np.nan])

    assert ap.t30_mape(target, pred) == pytest.approx(20.0)


def test_t30_mape_without_valid_band_pair_raises() -> None:
    """All-NaN bands leave nothing to score."""
    with pytest.raises(ValueError, match="no valid paired octave-band T30 estimates"):
        ap.t30_mape(np.array([np.nan, 0.5]), np.array([0.5, np.nan]))


def test_c50_mae_db_identical_bands_returns_zero() -> None:
    """Bands scored against themselves have zero clarity error."""
    c50 = np.array([2.0, -1.0, 4.5])

    assert ap.c50_mae_db(c50, c50) == pytest.approx(0.0)


def test_c50_mae_db_handcrafted_case() -> None:
    """|1−3| and |−2−(−1)| average to 1.5 dB."""
    assert ap.c50_mae_db(np.array([3.0, -1.0]), np.array([1.0, -2.0])) == pytest.approx(1.5)


def test_pearson_correlation_perfectly_linear_pairs_returns_one() -> None:
    """Predictions that are an affine map of targets correlate perfectly."""
    target = np.array([0.2, 0.5, 0.9, 1.4])

    assert ap.pearson_correlation(target, 2.0 * target + 1.0) == pytest.approx(1.0)


def test_pearson_correlation_handcrafted_case() -> None:
    """R([1, 2, 3, 4], [1, 3, 2, 4]) = 0.8."""
    assert ap.pearson_correlation(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0])
    ) == pytest.approx(0.8)


def test_pearson_correlation_skips_pairs_with_nan() -> None:
    """A NaN on either side drops that pair instead of poisoning the statistic."""
    target = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 7.0])
    pred = np.array([1.0, 3.0, 2.0, 4.0, 9.0, np.nan])

    assert ap.pearson_correlation(target, pred) == pytest.approx(0.8)


def test_pearson_correlation_fewer_than_two_pairs_returns_nan() -> None:
    """Correlation is undefined for a single pair."""
    assert np.isnan(ap.pearson_correlation(np.array([1.0]), np.array([2.0])))


def test_frechet_distance_identical_sets_returns_zero() -> None:
    """A set of embeddings has zero Fréchet distance to itself."""
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [1.0, 1.0]])

    assert ap.frechet_distance(embeddings, embeddings) == pytest.approx(0.0, abs=1e-9)


def test_frechet_distance_shifted_set_returns_squared_mean_offset() -> None:
    """Equal covariances leave only ‖μ₁ − μ₂‖², so a 3-unit shift scores 9."""
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [1.0, 1.0]])

    assert ap.frechet_distance(embeddings, embeddings + [3.0, 0.0]) == pytest.approx(9.0)


def test_frechet_distance_fewer_than_two_embeddings_raises() -> None:
    """A covariance needs at least two embeddings per set."""
    with pytest.raises(ValueError, match="at least two embeddings"):
        ap.frechet_distance(np.zeros((1, 2)), np.zeros((3, 2)))


def test_frechet_distance_singular_covariances_returns_finite_nonnegative() -> None:
    """Collinear embeddings give rank-deficient covariances; the distance stays a valid one."""
    line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    distance = ap.frechet_distance(line, line + [0.5, 0.5])

    assert np.isfinite(distance)
    assert distance == pytest.approx(0.5, abs=1e-6)


def test_frechet_distance_never_negative_from_rounding() -> None:
    """Near-identical sets round to a tiny negative trace; the result is clamped to zero."""
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [1.0, 1.0]])

    assert ap.frechet_distance(embeddings, embeddings + 1e-9) >= 0.0
