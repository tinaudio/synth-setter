"""Tests for pyFDN response-loss evaluation metrics."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from pyFDN.train import losses

from synth_setter.evaluation.response_losses import (
    PYFDN_DIAGNOSTIC_LOSSES,
    PYFDN_MATCHING_LOSSES,
    compute_pyfdn_response_losses,
)

_SAMPLE_RATE = 44100
_NUM_SAMPLES = 4 * _SAMPLE_RATE
_EXPECTED_KEYS = {
    "pyfdn_asymmetric_flat_magnitude_pred",
    "pyfdn_asymmetric_flat_magnitude_target",
    "pyfdn_energy_pred",
    "pyfdn_energy_target",
    "pyfdn_flat_magnitude_pred",
    "pyfdn_flat_magnitude_target",
    "pyfdn_flat_spectrogram_pred",
    "pyfdn_flat_spectrogram_target",
    "pyfdn_match_cumulative_energy",
    "pyfdn_match_energy_decay",
    "pyfdn_match_impulse_response",
    "pyfdn_match_magnitude",
    "pyfdn_match_mel_spectrogram",
    "pyfdn_match_spectrogram",
}


def _broadband_decay(decay_rate: float = 20.0) -> np.ndarray:
    rng = np.random.default_rng(31)
    time = np.arange(_NUM_SAMPLES, dtype=np.float64) / _SAMPLE_RATE
    return (rng.standard_normal(_NUM_SAMPLES) * np.exp(-decay_rate * time))[None, :]


def test_response_loss_registries_cover_every_public_concrete_response_loss() -> None:
    """The registries cover every public concrete response-loss class."""
    public_concrete = {
        exported
        for name in losses.__all__
        if inspect.isclass(exported := getattr(losses, name))
        and issubclass(exported, losses.ResponseLoss)
        and exported is not losses.ResponseLoss
        and not inspect.isabstract(exported)
    }

    registered = set(PYFDN_MATCHING_LOSSES.values()) | set(PYFDN_DIAGNOSTIC_LOSSES.values())

    assert public_concrete == registered
    assert len(registered) == 10


def test_response_losses_return_explicit_public_metric_schema() -> None:
    """A four-second RIR produces every named metric as a finite scalar."""
    audio = _broadband_decay()

    metrics = compute_pyfdn_response_losses(audio, audio, _SAMPLE_RATE)

    assert set(metrics) == _EXPECTED_KEYS
    assert np.isfinite(list(metrics.values())).all()


def test_response_losses_constant_signals_preserve_mse_and_energy_semantics() -> None:
    """Impulse-response MSE and Energy retain pyFDN's exact scalar definitions."""
    target = np.ones((1, 8192), dtype=np.float64)
    pred = np.full_like(target, 3.0)

    metrics = compute_pyfdn_response_losses(target, pred, _SAMPLE_RATE)

    assert metrics["pyfdn_match_impulse_response"] == pytest.approx(4.0)
    assert metrics["pyfdn_energy_target"] == pytest.approx(67_092_481.0)
    assert metrics["pyfdn_energy_pred"] == pytest.approx(5_435_670_529.0)


def test_response_matching_losses_identical_decay_are_zero() -> None:
    """Every matching objective is zero for identical broadband decays."""
    audio = _broadband_decay()

    metrics = compute_pyfdn_response_losses(audio, audio, _SAMPLE_RATE)

    matching = {name: value for name, value in metrics.items() if name.startswith("pyfdn_match_")}
    assert matching == pytest.approx({name: 0.0 for name in matching}, abs=1e-12)


def test_response_matching_losses_changed_decay_are_positive() -> None:
    """Every matching objective detects a changed broadband decay."""
    target = _broadband_decay(decay_rate=25.0)
    pred = _broadband_decay(decay_rate=8.0)

    metrics = compute_pyfdn_response_losses(target, pred, _SAMPLE_RATE)

    matching = [value for name, value in metrics.items() if name.startswith("pyfdn_match_")]
    assert np.isfinite(matching).all()
    assert (np.asarray(matching) > 0.0).all()


def test_response_diagnostics_identical_decay_report_equal_sides() -> None:
    """Target-independent diagnostics report equal sides for identical RIRs."""
    audio = _broadband_decay()

    metrics = compute_pyfdn_response_losses(audio, audio, _SAMPLE_RATE)

    for prefix in PYFDN_DIAGNOSTIC_LOSSES:
        assert metrics[f"{prefix}_target"] == pytest.approx(metrics[f"{prefix}_pred"])


@pytest.mark.parametrize("sample_rate", [0.0, -1.0, np.inf, np.nan])
def test_response_losses_invalid_sample_rate_raises(sample_rate: float) -> None:
    """Non-positive and non-finite sample rates are rejected.

    :param sample_rate: Sampling frequency in Hz violating the positive-finite contract.
    """
    audio = _broadband_decay()

    with pytest.raises(ValueError, match="sample_rate"):
        compute_pyfdn_response_losses(audio, audio, sample_rate)


def test_response_losses_complex_audio_raises() -> None:
    """Complex audio is rejected before float conversion can discard its imaginary part."""
    audio = _broadband_decay().astype(np.complex128)
    audio[0, 0] += 1j

    with pytest.raises(ValueError, match="real-valued"):
        compute_pyfdn_response_losses(audio, audio, _SAMPLE_RATE)


def test_response_losses_nonfinite_audio_raises() -> None:
    """Non-finite samples are rejected before pyFDN evaluation."""
    target = _broadband_decay()
    pred = target.copy()
    pred[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_pyfdn_response_losses(target, pred, _SAMPLE_RATE)


def test_response_losses_multichannel_average_corresponding_channel_scores() -> None:
    """A wrong opposite-phase second channel contributes to each scalar reduction."""
    first = _broadband_decay(decay_rate=20.0)
    second = _broadband_decay(decay_rate=12.0)
    target = np.concatenate((first, second), axis=0)
    pred = np.concatenate((first, -second), axis=0)

    metrics = compute_pyfdn_response_losses(target, pred, _SAMPLE_RATE)
    first_metrics = compute_pyfdn_response_losses(first, first, _SAMPLE_RATE)
    second_metrics = compute_pyfdn_response_losses(second, -second, _SAMPLE_RATE)

    assert metrics == pytest.approx(
        {name: (first_metrics[name] + second_metrics[name]) / 2.0 for name in metrics}
    )
    assert metrics["pyfdn_match_impulse_response"] > 0.0


def test_response_losses_channel_count_mismatch_raises() -> None:
    """Target and prediction channel counts cannot broadcast during comparison."""
    mono = _broadband_decay()
    stereo = np.concatenate((mono, mono), axis=0)

    with pytest.raises(ValueError, match="same.*shape"):
        compute_pyfdn_response_losses(mono, stereo, _SAMPLE_RATE)


def test_response_losses_mismatched_lengths_raise() -> None:
    """Target and prediction must contain the same sample count."""
    target = _broadband_decay()
    pred = target[:, :-1]

    with pytest.raises(ValueError, match="same.*shape"):
        compute_pyfdn_response_losses(target, pred, _SAMPLE_RATE)


def test_response_losses_silent_reference_raises_instead_of_skipping_loss() -> None:
    """An unusable silent channel fails even when another channel is valid."""
    valid = _broadband_decay()
    silence = np.zeros_like(valid)
    target = np.concatenate((valid, silence), axis=0)

    with pytest.raises(ValueError, match="pyfdn_match_energy_decay"):
        compute_pyfdn_response_losses(target, target, _SAMPLE_RATE)


def test_response_losses_short_signal_raises_instead_of_skipping_loss() -> None:
    """A signal shorter than upstream analysis windows fails without dropping metrics."""
    short = _broadband_decay()[:, :512]

    with pytest.raises(ValueError, match="pyfdn_match_"):
        compute_pyfdn_response_losses(short, short, _SAMPLE_RATE)
