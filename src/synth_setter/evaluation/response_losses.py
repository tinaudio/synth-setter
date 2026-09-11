"""Evaluate pyFDN training response losses as channel-aware audio metrics.

Usage: ``compute_pyfdn_response_losses(target, pred, 44100)`` returns a
``pyfdn_*`` scalar mapping for channel-first impulse responses. Corresponding
channels are analyzed independently before their scalar results are averaged.
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


def _impulse_response_channels(audio: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite real channel-first impulse response.

    :param audio: Impulse response with unrestricted finite gain.
    :param name: Signal identifier included in validation errors.
    :returns: Float64 array with shape ``(channels, samples)``.
    :raises ValueError: If the input is empty, complex, non-numeric, or non-finite.
    """
    raw = np.asarray(audio)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} impulse response must be real-valued")
    try:
        array = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} impulse response must be a numeric array") from error
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"{name} impulse response must be non-empty channel-first audio (channels, samples)"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} impulse response must contain only finite values")
    return array


def validate_impulse_response_pair(
    target: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite real channel-first target and prediction arrays of one shape.

    :param target: Target impulse response with shape ``(channels, samples)``.
    :param pred: Prediction with exactly the same shape as ``target``.
    :returns: Float64 channel-first target and prediction arrays.
    :raises ValueError: If either signal is invalid or their geometries differ.
    """
    target_ir = _impulse_response_channels(target, "target")
    pred_ir = _impulse_response_channels(pred, "predicted")
    if target_ir.shape != pred_ir.shape:
        raise ValueError(
            "target and predicted impulse responses must have the same (channels, samples) shape"
        )
    return target_ir, pred_ir


def validate_mono_impulse_response_pair(
    target: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite real mono target and prediction vectors of equal length.

    :param target: Channel-first target impulse response with shape ``(1, samples)``.
    :param pred: Channel-first prediction with the same shape as ``target``.
    :returns: Float64 time-major target and prediction vectors.
    :raises ValueError: If either signal is invalid, non-mono, or has different geometry.
    """
    target_channels, pred_channels = validate_impulse_response_pair(target, pred)
    if target_channels.shape[0] != 1:
        raise ValueError("mono impulse responses must have shape (1, samples)")
    return target_channels[0], pred_channels[0]


def _sample_rate(sample_rate: float) -> float:
    """Return a finite positive sample rate.

    :param sample_rate: Candidate sample rate in Hz.
    :returns: Sample rate as a float.
    :raises ValueError: If the value is non-numeric, non-finite, or non-positive.
    """
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("sample_rate must be a finite positive number") from error
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate must be a finite positive number")
    return rate


def _responses(
    target_ir: np.ndarray,
    pred_ir: np.ndarray,
    sample_rate: float,
) -> tuple[Response, Response]:
    """Build pyFDN responses for one validated corresponding channel pair.

    :param target_ir: One-dimensional target impulse response.
    :param pred_ir: One-dimensional prediction.
    :param sample_rate: Validated sample rate in Hz.
    :returns: Target and prediction Response objects.
    """
    target_response = Response(torch.from_numpy(target_ir[:, None, None]), sample_rate)
    pred_response = Response(torch.from_numpy(pred_ir[:, None, None]), sample_rate)
    return target_response, pred_response


def _loss_value(metric_name: str, criterion: ResponseLoss, response: Response) -> float:
    try:
        value = float(criterion(response).detach().cpu().item())
    except Exception as error:
        raise ValueError(f"{metric_name} failed: {error}") from error
    if not np.isfinite(value):
        raise ValueError(f"{metric_name} must be finite; got {value}")
    return value


def _mono_response_losses(
    target_ir: np.ndarray, pred_ir: np.ndarray, sample_rate: float
) -> dict[str, float]:
    """Evaluate every response loss for one validated channel pair.

    :param target_ir: One-dimensional target impulse response.
    :param pred_ir: One-dimensional prediction.
    :param sample_rate: Validated sample rate in Hz.
    :returns: Scalar values under explicit ``pyfdn_*`` metric names.
    :raises ValueError: Any upstream loss construction or evaluation fails.
    """
    target_response, pred_response = _responses(target_ir, pred_ir, sample_rate)
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
def compute_pyfdn_response_losses(
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: float,
) -> dict[str, float]:
    """Return channel-mean pyFDN response losses for corresponding IR channels.

    :param target: Finite target impulse response with shape ``(channels, samples)``.
    :param pred: Finite prediction with exactly the same shape as ``target``.
    :param sample_rate: Finite positive sample rate in Hz.
    :returns: Channel-mean scalar values under explicit ``pyfdn_*`` metric names.
    """
    target_channels, pred_channels = validate_impulse_response_pair(target, pred)
    rate = _sample_rate(sample_rate)
    channel_metrics = [
        _mono_response_losses(target_ir, pred_ir, rate)
        for target_ir, pred_ir in zip(target_channels, pred_channels, strict=True)
    ]
    return {
        name: float(np.mean([metrics[name] for metrics in channel_metrics]))
        for name in channel_metrics[0]
    }


@torch.no_grad()
def compute_pyfdn_match_energy_decay(
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: float,
) -> float:
    """Return channel-mean pyFDN octave-band energy-decay matching loss.

    :param target: Finite target impulse response with shape ``(channels, samples)``.
    :param pred: Finite prediction with exactly the same shape as ``target``.
    :param sample_rate: Finite positive sample rate in Hz.
    :returns: Channel-mean RMS dB difference over target-valid decay frames.
    :raises ValueError: An upstream energy-decay loss cannot be constructed.
    """
    target_channels, pred_channels = validate_impulse_response_pair(target, pred)
    rate = _sample_rate(sample_rate)
    metric_name = "pyfdn_match_energy_decay"
    values = []
    for target_ir, pred_ir in zip(target_channels, pred_channels, strict=True):
        _, pred_response = _responses(target_ir, pred_ir, rate)
        try:
            criterion = MatchEnergyDecay(target_ir)
        except Exception as error:
            raise ValueError(f"{metric_name} failed: {error}") from error
        values.append(_loss_value(metric_name, criterion, pred_response))
    return float(np.mean(values))
