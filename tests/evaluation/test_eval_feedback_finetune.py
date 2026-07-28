"""Unit tests for the paired-delta statistics of the decisive finetune eval."""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.evaluation.eval_feedback_finetune import paired_delta


def test_paired_delta_constant_shift_reports_shift_and_zero_sem() -> None:
    """A constant per-row shift yields exactly that mean delta with zero SEM."""
    base = np.array([1.0, 2.0, 3.0, 4.0])
    arm = base - 0.5

    mean, sem = paired_delta(arm, base)

    assert mean == pytest.approx(-0.5)
    assert sem == pytest.approx(0.0)


def test_paired_delta_known_values_match_hand_computation() -> None:
    """Mean and SEM match the hand-computed values for a small fixed sample."""
    arm = np.array([1.0, 3.0])
    base = np.array([0.0, 0.0])

    mean, sem = paired_delta(arm, base)

    assert mean == pytest.approx(2.0)
    # Sample std of [1, 3] is sqrt(2); SEM = sqrt(2)/sqrt(2) = 1.
    assert sem == pytest.approx(1.0)


def test_paired_delta_mismatched_lengths_raises() -> None:
    """Unequal row counts cannot be paired and must fail loudly."""
    with pytest.raises(ValueError, match="paired"):
        paired_delta(np.array([1.0]), np.array([1.0, 2.0]))
