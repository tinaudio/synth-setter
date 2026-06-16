"""Unit tests for the #489 reuse-depth reproducer's pure scoring core.

These pin the all-pairs worst-MSS reducer, the loudness gate, the bug verdict, and
the CLI exit path without rendering a VST: the metric is injected and the render
layer is stubbed. One ``@slow @requires_vst`` test drives the real render chain.
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


def _sine(amplitude: float, sample_rate: int = 44100, seconds: float = 1.0) -> np.ndarray:
    """Return a 2-channel 440 Hz sine at ``amplitude``.

    :param amplitude: Peak amplitude of the tone.
    :param sample_rate: Samples per second.
    :param seconds: Tone length (≥0.4 s so the loudness meter has a full block).
    :returns: A ``(2, T)`` float32 sine.
    """
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    wave = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([wave, wave])


def test_all_pairs_worst_mss_selects_the_first_most_divergent_pair() -> None:
    """The worst pair is the earliest (i, j) maximising the metric (``>``, not ``>=``)."""
    # means [0, 5, 0, 5]: pairs (0,1),(0,3),(1,2),(2,3) all gap 5; (0,1) is earliest.
    renders = [_const(0.0), _const(5.0), _const(0.0), _const(5.0)]
    result = repro.all_pairs_worst_mss(
        renders, metric=lambda a, b: abs(float(a.mean()) - float(b.mean()))
    )
    assert result.worst_pair == (0, 1)
    assert result.mss_max == pytest.approx(5.0)


def test_all_pairs_worst_mss_scores_and_counts_every_unordered_pair() -> None:
    """Every unordered pair is scored once and ``pair_count`` matches the calls made."""
    calls: list[tuple[int, int]] = []

    def counting_metric(a: np.ndarray, b: np.ndarray) -> float:
        # Encode each render's index in its fill value so the closure can record
        # which pairs were scored without sharing mutable index state.
        calls.append((int(a[0, 0]), int(b[0, 0])))
        return 0.0

    renders = [_const(float(i)) for i in range(4)]
    result = repro.all_pairs_worst_mss(renders, metric=counting_metric)
    assert sorted(calls) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert result.pair_count == 6
    assert result.worst_pair == (0, 1)


def test_all_pairs_worst_mss_traces_peak_amplitude_per_render() -> None:
    """Per-render peak amplitude is reported so silent (junk) renders are visible."""
    result = repro.all_pairs_worst_mss([_const(0.001), _const(0.5)], metric=lambda a, b: 0.0)
    assert result.peaks == pytest.approx([0.001, 0.5])


def test_all_pairs_worst_mss_rejects_fewer_than_two_renders() -> None:
    """A single render has no pair to compare, so the reducer refuses it."""
    with pytest.raises(ValueError, match="at least 2"):
        repro.all_pairs_worst_mss([_const(0.1)], metric=lambda a, b: 0.0)


def test_integrated_loudness_rises_with_amplitude() -> None:
    """A louder tone reports higher LUFS, pinning axis/sample-rate wiring of the gate."""
    loud = repro.integrated_loudness(_sine(0.5), 44100)
    quiet = repro.integrated_loudness(_sine(0.02), 44100)
    assert np.isfinite(loud)
    assert loud > quiet


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
    assert repro.classify(reused_max, reloaded_max, repro._MSS_THRESHOLD) == expected


@pytest.mark.parametrize(
    ("verdicts", "exits"),
    [
        ({12: repro.Verdict.BUG_PRESENT}, True),
        ({12: repro.Verdict.NO_REPRO, 40: repro.Verdict.NO_REPRO}, False),
        ({12: repro.Verdict.INCONCLUSIVE}, False),
    ],
)
def test_main_exits_nonzero_only_when_a_depth_is_bug_present(
    monkeypatch: pytest.MonkeyPatch, verdicts: dict[int, repro.Verdict], exits: bool
) -> None:
    """``main`` exits 1 iff some depth is BUG_PRESENT; INCONCLUSIVE/NO_REPRO exit 0.

    :param monkeypatch: Replaces ``run`` with a stub returning ``verdicts``.
    :param verdicts: Stubbed per-depth verdicts ``run`` returns.
    :param exits: Whether ``main`` should raise ``SystemExit(1)``.
    """
    monkeypatch.setattr(repro, "run", lambda *a, **k: verdicts)
    if exits:
        with pytest.raises(SystemExit) as exc:
            repro.main(["--depths", "12", "--no-control"])
        assert exc.value.code == 1
    else:
        repro.main(["--depths", "12", "--no-control"])


def test_main_passes_parsed_depths_through_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` forwards the parsed ``--depths`` list to ``run``.

    :param monkeypatch: Captures the depths ``run`` is called with.
    """
    captured: dict[str, object] = {}

    def fake_run(depths: object, **kwargs: object) -> dict[int, repro.Verdict]:
        captured["depths"] = depths
        return {12: repro.Verdict.NO_REPRO}

    monkeypatch.setattr(repro, "run", fake_run)
    repro.main(["--depths", "12", "40"])
    assert captured["depths"] == [12, 40]


@pytest.mark.slow
@pytest.mark.requires_vst
def test_main_drives_the_real_render_chain_end_to_end() -> None:
    """``main`` renders through resolve_patch -> all_pairs_worst_mss -> classify on a real VST.

    Exercises the integration the stubbed CLI tests cannot: at depth 2 with no control
    the run completes without raising and stays NO_REPRO (no #489 SystemExit) on the
    current, flush-everything render path.
    """
    repro.main(["--depths", "2", "--no-control"])
