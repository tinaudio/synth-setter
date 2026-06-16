"""Unit tests for the #489 reuse-depth reproducer's pure scoring core.

These pin the all-pairs worst-MSS reducer and the bug verdict without touching a
VST: the metric is injected so the enumeration, worst-pair selection, peak trace,
and verdict logic are exercised on plain arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.tools import repro_489_reuse_depth as repro


def _const(value: float) -> np.ndarray:
    """Return a 1-row clip whose every sample equals ``value`` (peak == ``|value|``).

    :param value: Constant amplitude filling the clip.
    :returns: A ``(1, 8)`` float32 array of ``value``.
    """
    return np.full((1, 8), value, dtype=np.float32)


def test_all_pairs_worst_mss_selects_the_most_divergent_pair() -> None:
    """The worst pair is the (i, j) maximising the injected metric, not the first tripped."""
    renders = [_const(0.0), _const(1.0), _const(5.0)]
    # L1-of-means stand-in: distance grows with the value gap, so (0, 2) is worst.
    result = repro.all_pairs_worst_mss(
        renders, metric=lambda a, b: abs(float(a.mean()) - float(b.mean()))
    )
    assert result.worst_pair == (0, 2)
    assert result.mss_max == pytest.approx(5.0)


def test_all_pairs_worst_mss_counts_every_unordered_pair() -> None:
    """``pair_count`` is n*(n-1)/2 — every unordered pair, the #489 all-pairs surface."""
    renders = [_const(float(i)) for i in range(4)]
    result = repro.all_pairs_worst_mss(renders, metric=lambda a, b: 0.0)
    assert result.pair_count == 6
    assert result.depth == 4


def test_all_pairs_worst_mss_traces_peak_amplitude_per_render() -> None:
    """Per-render peak amplitude is reported so silent (junk) renders are visible."""
    renders = [_const(0.001), _const(0.5)]
    result = repro.all_pairs_worst_mss(renders, metric=lambda a, b: 0.0)
    assert result.peaks == pytest.approx([0.001, 0.5])


def test_all_pairs_worst_mss_rejects_fewer_than_two_renders() -> None:
    """A single render has no pair to compare, so the reducer refuses it."""
    with pytest.raises(ValueError, match="at least 2"):
        repro.all_pairs_worst_mss([_const(0.1)], metric=lambda a, b: 0.0)


def test_classify_flags_bug_when_reuse_diverges_but_reload_holds() -> None:
    """Reuse over threshold while reload stays under it is the #489 signature."""
    assert repro.classify(reused_max=21.0, reloaded_max=3.0, threshold=10.0) == repro.BUG_PRESENT


def test_classify_reports_clean_when_both_arms_hold() -> None:
    """Both arms under threshold means no reproduction at this depth."""
    assert repro.classify(reused_max=2.0, reloaded_max=3.0, threshold=10.0) == repro.NO_REPRO


def test_classify_reports_inconclusive_when_reload_also_diverges() -> None:
    """Reload over threshold too means the divergence is not the reuse bug (patch/phase)."""
    assert repro.classify(reused_max=21.0, reloaded_max=15.0, threshold=10.0) == repro.INCONCLUSIVE
