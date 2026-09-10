"""Tests for waveform-driven log-mel normalization statistics."""

import logging

import numpy as np
import pytest
import torch

from synth_setter.data.normalization_stats import estimate_log_mel_statistics
from synth_setter.models.components.spec_encoder import LogMelFrontend


class _GridFrontend(LogMelFrontend):
    """Expose waveform values as one-channel, two-position feature grids."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        """Reshape each two-sample waveform into a tiny feature grid.

        :param waveform: Two-sample waveform rows.
        :returns: One-channel feature grids.
        """
        return waveform.reshape(waveform.shape[0], 1, 1, 2)


def test_estimate_log_mel_statistics_returns_per_position_population_statistics() -> None:
    """Estimator returns population moments independently at every grid position."""
    mean, std = estimate_log_mel_statistics(
        [torch.tensor([[1.0, 2.0], [3.0, 6.0]]), torch.tensor([[5.0, 10.0]])],
        _GridFrontend(),
    )

    np.testing.assert_array_equal(mean, np.array([[[3.0, 6.0]]], dtype=np.float32))
    np.testing.assert_allclose(
        std,
        np.array([[[1.6329932, 3.2659864]]], dtype=np.float32),
        rtol=1e-6,
    )


def test_estimate_log_mel_statistics_sample_limit_truncates_final_batch() -> None:
    """Only rows within the requested cap contribute to the estimate."""
    mean, _ = estimate_log_mel_statistics(
        [torch.tensor([[1.0, 2.0], [3.0, 6.0], [101.0, 202.0]])],
        _GridFrontend(),
        sample_limit=2,
    )

    np.testing.assert_array_equal(mean, np.array([[[2.0, 4.0]]], dtype=np.float32))


def test_estimate_log_mel_statistics_fewer_than_two_rows_raises() -> None:
    """A population estimate requires at least two waveform rows."""
    with pytest.raises(ValueError, match="at least 2"):
        estimate_log_mel_statistics([torch.tensor([[1.0, 2.0]])], _GridFrontend())


def test_estimate_log_mel_statistics_masks_constant_positions_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Constant feature positions are masked with the shared warning policy.

    :param caplog: Captured warning output.
    """
    with caplog.at_level(logging.WARNING):
        mean, std = estimate_log_mel_statistics(
            [torch.tensor([[2.0, 1.0], [2.0, 3.0]])],
            _GridFrontend(),
            mask_degenerate=True,
        )

    np.testing.assert_array_equal(mean, np.array([[[2.0, 2.0]]], dtype=np.float32))
    np.testing.assert_array_equal(std, np.array([[[1.0, 1.0]]], dtype=np.float32))
    assert "std=1.0" in caplog.text


def test_estimate_log_mel_statistics_constant_position_raises_by_default() -> None:
    """Default estimation rejects constant feature positions."""
    with pytest.raises(ValueError, match="zero variance"):
        estimate_log_mel_statistics(
            [torch.tensor([[2.0, 1.0], [2.0, 3.0]])],
            _GridFrontend(),
        )


def test_estimate_log_mel_statistics_non_positive_limit_raises() -> None:
    """A non-positive sample cap is invalid."""
    with pytest.raises(ValueError, match="sample_limit must be positive"):
        estimate_log_mel_statistics([], _GridFrontend(), sample_limit=0)


def test_estimate_log_mel_statistics_non_finite_raw_features_raise() -> None:
    """Non-finite frontend output cannot become persisted statistics."""
    with pytest.raises(ValueError, match="finite"):
        estimate_log_mel_statistics(
            [torch.tensor([[1.0, 2.0], [float("nan"), 3.0]])],
            _GridFrontend(),
        )


def test_estimate_log_mel_statistics_limit_does_not_pull_an_extra_batch() -> None:
    """Reaching the cap does not render or read another batch."""

    def batches():
        yield torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        raise AssertionError("sample limit should stop iteration")

    estimate_log_mel_statistics(batches(), _GridFrontend(), sample_limit=2, mask_degenerate=True)
