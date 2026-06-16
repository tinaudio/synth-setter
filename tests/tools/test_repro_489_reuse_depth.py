"""Unit tests for the #489 reuse-depth reproducer's pure scoring core.

These pin the all-pairs worst-MSS reducer, the bug verdict, and the CLI exit path
without touching a VST: the metric is injected and the render layer is stubbed, so
enumeration, worst-pair selection, peak trace, verdict logic, and ``main``'s
non-zero exit are exercised on plain data.
"""

from __future__ import annotations

import numpy as np
import pytest

from synth_setter.tools import repro_489_reuse_depth as repro


def _const(value: float) -> np.ndarray:
    """Return a 1-row clip filled with ``value``.

    :param value: Fill value; also the clip's peak amplitude.
    :returns: A ``(1, 8)`` float32 array of ``value``.
    """
    return np.full((1, 8), value, dtype=np.float32)


def test_all_pairs_worst_mss_selects_the_first_most_divergent_pair() -> None:
    """The worst pair is the earliest (i, j) maximising the metric (``>``, not ``>=``)."""
    # means [0, 5, 0, 5]: pairs (0,1),(0,3),(1,2),(2,3) all gap 5; (0,1) is earliest.
    renders = [_const(0.0), _const(5.0), _const(0.0), _const(5.0)]
    result = repro.all_pairs_worst_mss(
        renders, metric=lambda a, b: abs(float(a.mean()) - float(b.mean()))
    )
    assert result.worst_pair == (0, 1)
    assert result.mss_max == pytest.approx(5.0)
    assert result.mss_max >= 0.0


def test_all_pairs_worst_mss_scores_and_counts_every_unordered_pair() -> None:
    """Every unordered pair is scored once and ``pair_count`` matches the calls made."""
    calls: list[tuple[int, int]] = []

    def counting_metric(a: np.ndarray, b: np.ndarray) -> float:
        calls.append((int(a[0, 0]), int(b[0, 0])))
        return 0.0

    renders = [_const(float(i)) for i in range(4)]
    result = repro.all_pairs_worst_mss(renders, metric=counting_metric)
    assert calls == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert result.pair_count == 6
    # A worst pair is recorded even when every score ties at 0.0 (sentinel is -inf).
    assert result.worst_pair == (0, 1)


def test_all_pairs_worst_mss_traces_peak_amplitude_per_render() -> None:
    """Per-render peak amplitude is reported so silent (junk) renders are visible."""
    result = repro.all_pairs_worst_mss([_const(0.001), _const(0.5)], metric=lambda a, b: 0.0)
    assert result.peaks == pytest.approx([0.001, 0.5])


def test_all_pairs_worst_mss_rejects_fewer_than_two_renders() -> None:
    """A single render has no pair to compare, so the reducer refuses it."""
    with pytest.raises(ValueError, match="at least 2"):
        repro.all_pairs_worst_mss([_const(0.1)], metric=lambda a, b: 0.0)


@pytest.mark.parametrize(
    ("reused_max", "reloaded_max", "expected"),
    [
        (21.0, 3.0, repro.Verdict.BUG_PRESENT),  # reuse diverges, reload holds
        (2.0, 3.0, repro.Verdict.NO_REPRO),  # both clean
        (21.0, 15.0, repro.Verdict.INCONCLUSIVE),  # both diverge -> not the reuse bug
        (10.0, 3.0, repro.Verdict.NO_REPRO),  # boundary: == threshold is clean
        (21.0, None, repro.Verdict.BUG_PRESENT),  # control skipped, reuse diverges
        (2.0, None, repro.Verdict.NO_REPRO),  # control skipped, reuse clean
    ],
)
def test_classify_labels_each_reuse_reload_combination(
    reused_max: float, reloaded_max: float | None, expected: repro.Verdict
) -> None:
    """``classify`` maps reuse/reload worst scores to the right verdict at every case.

    :param reused_max: Reuse-arm worst all-pairs MSS.
    :param reloaded_max: Reload-arm worst all-pairs MSS, or ``None`` when skipped.
    :param expected: The verdict ``classify`` must return.
    """
    assert repro.classify(reused_max, reloaded_max, threshold=10.0) == expected


def test_main_exits_nonzero_when_any_depth_reproduces_the_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main`` exits 1 when ``run`` reports BUG_PRESENT at any depth (CI signal).

    :param monkeypatch: Stubs ``run`` so no VST is rendered.
    """
    monkeypatch.setattr(repro, "run", lambda *a, **k: {12: repro.Verdict.BUG_PRESENT})
    with pytest.raises(SystemExit) as exc:
        repro.main(["--depths", "12", "--no-control"])
    assert exc.value.code == 1


def test_main_exits_zero_when_no_depth_reproduces_the_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main`` returns normally (exit 0) when no depth is BUG_PRESENT.

    :param monkeypatch: Stubs ``run`` and captures the parsed depths.
    """
    captured: dict[str, object] = {}

    def fake_run(depths: object, **kwargs: object) -> dict[int, repro.Verdict]:
        captured["depths"] = depths
        return {12: repro.Verdict.NO_REPRO, 40: repro.Verdict.NO_REPRO}

    monkeypatch.setattr(repro, "run", fake_run)
    repro.main(["--depths", "12", "40"])
    assert captured["depths"] == [12, 40]
