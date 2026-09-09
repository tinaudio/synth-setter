"""Evaluate pyFDN training response losses as audio metrics.

Usage: ``compute_pyfdn_response_losses(target[None, :], pred[None, :], 44100)``
returns a ``pyfdn_*`` scalar mapping for one-dimensional float impulse responses.
Response gain is unrestricted; PCM sample bounds do not apply to filter responses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch
from pyFDN import Response
from pyFDN.train.losses import (
    AsymmetricFlatMagnitude,
    Energy,
    FlatMagnitude,
    FlatSpectrogram,
    MatchCumulativeEnergy,
    MatchEnergyDecay,
    MatchImpulseResponse,
    MatchMagnitude,
    MatchMelSpectrogram,
    MatchSpectrogram,
    ResponseLoss,
)

PYFDN_MATCHING_LOSSES: Mapping[str, Callable[[np.ndarray], ResponseLoss]] = {
    "pyfdn_match_cumulative_energy": MatchCumulativeEnergy,
    "pyfdn_match_energy_decay": MatchEnergyDecay,
    "pyfdn_match_impulse_response": MatchImpulseResponse,
    "pyfdn_match_magnitude": MatchMagnitude,
    "pyfdn_match_mel_spectrogram": MatchMelSpectrogram,
    "pyfdn_match_spectrogram": MatchSpectrogram,
}
PYFDN_DIAGNOSTIC_LOSSES: Mapping[str, Callable[[], ResponseLoss]] = {
    "pyfdn_asymmetric_flat_magnitude": AsymmetricFlatMagnitude,
    "pyfdn_energy": Energy,
    "pyfdn_flat_magnitude": FlatMagnitude,
    "pyfdn_flat_spectrogram": FlatSpectrogram,
}


def _mono_impulse_response(audio: np.ndarray, name: str) -> np.ndarray:
    """Validate real ``(1, samples)`` audio and return a float64 time-major vector.

    :param audio: Mono impulse response, with unrestricted finite gain.
    :param name: Signal identifier included in validation errors.
    :returns: One-dimensional impulse response.
    :raises ValueError: If the input is empty, non-mono, complex, or non-finite.
    """
    raw = np.asarray(audio)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} impulse response must be real-valued")
    try:
        array = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} impulse response must be a numeric array") from error
    if array.ndim != 2 or array.shape[0] != 1 or array.shape[1] == 0:
        raise ValueError(f"{name} impulse response must be non-empty mono audio (1, samples)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} impulse response must contain only finite values")
    return array[0]


def validate_mono_impulse_response_pair(
    target: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite real mono target and prediction vectors of equal length.

    :param target: Channel-first target impulse response with shape ``(1, samples)``.
    :param pred: Channel-first prediction with the same sample count as ``target``.
    :returns: Float64 time-major target and prediction vectors.
    :raises ValueError: If either signal violates the mono impulse-response contract.
    """
    target_ir = _mono_impulse_response(target, "target")
    pred_ir = _mono_impulse_response(pred, "predicted")
    if target_ir.shape != pred_ir.shape:
        raise ValueError("target and predicted impulse responses must have the same sample count")
    return target_ir, pred_ir


def _responses(
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, Response, Response]:
    """Build pyFDN ``(samples, 1, 1)`` responses from a mono audio pair.

    :param target: Target audio with shape ``(1, samples)``.
    :param pred: Prediction with the same shape as the target.
    :param sample_rate: Shared positive sample rate in Hz.
    :returns: Target time-major vector and target/prediction Response objects.
    :raises ValueError: If the sample rate or either input is invalid.
    """
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("sample_rate must be a finite positive number") from error
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate must be a finite positive number")

    target_ir, pred_ir = validate_mono_impulse_response_pair(target, pred)
    target_response = Response(torch.from_numpy(target_ir[:, None, None]), rate)
    pred_response = Response(torch.from_numpy(pred_ir[:, None, None]), rate)
    return target_ir, target_response, pred_response


def _loss_value(metric_name: str, criterion: ResponseLoss, response: Response) -> float:
    try:
        value = float(criterion(response).detach().cpu().item())
    except Exception as error:
        raise ValueError(f"{metric_name} failed: {error}") from error
    if not np.isfinite(value):
        raise ValueError(f"{metric_name} must be finite; got {value}")
    return value


@torch.no_grad()
def compute_pyfdn_response_losses(
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: float,
) -> dict[str, float]:
    """Return all public concrete pyFDN response losses for a mono IR pair.

    Matching objectives compare prediction against target. Target-independent
    diagnostics are reported separately for each side with ``_target`` and
    ``_pred`` suffixes.

    :param target: Finite mono target impulse response with shape ``(1, samples)``.
    :param pred: Finite mono prediction with the same sample count as ``target``.
    :param sample_rate: Finite positive sample rate in Hz.
    :returns: Scalar values under explicit ``pyfdn_*`` metric names.
    :raises ValueError: If validation or any upstream loss evaluation fails.
    """
    target_ir, target_response, pred_response = _responses(target, pred, sample_rate)
    metrics: dict[str, float] = {}

    for metric_name, loss_type in PYFDN_MATCHING_LOSSES.items():
        try:
            criterion = loss_type(target_ir)
        except Exception as error:
            raise ValueError(f"{metric_name} failed: {error}") from error
        metrics[metric_name] = _loss_value(metric_name, criterion, pred_response)

    for metric_prefix, loss_type in PYFDN_DIAGNOSTIC_LOSSES.items():
        try:
            criterion = loss_type()
        except Exception as error:
            raise ValueError(f"{metric_prefix} failed: {error}") from error
        metrics[f"{metric_prefix}_target"] = _loss_value(
            f"{metric_prefix}_target", criterion, target_response
        )
        metrics[f"{metric_prefix}_pred"] = _loss_value(
            f"{metric_prefix}_pred", criterion, pred_response
        )

    return metrics


@torch.no_grad()
def compute_pyfdn_match_energy_decay(
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: float,
) -> float:
    """Return pyFDN's octave-band energy-decay matching loss for a mono IR pair.

    :param target: Finite mono target impulse response with shape ``(1, samples)``.
    :param pred: Finite mono prediction with the same sample count as ``target``.
    :param sample_rate: Finite positive sample rate in Hz.
    :returns: RMS dB difference over target-valid octave-band decay frames.
    """
    target_ir, _, pred_response = _responses(target, pred, sample_rate)
    metric_name = "pyfdn_match_energy_decay"
    return _loss_value(metric_name, MatchEnergyDecay(target_ir), pred_response)
