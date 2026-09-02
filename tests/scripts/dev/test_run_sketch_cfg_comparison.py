"""Behavior tests for the sketch CFG suite runner."""

import pytest

from scripts.dev.run_sketch_cfg_comparison import aggregate_metrics, select_vocal_rows


def test_select_vocal_rows_mixed_corpus_returns_only_included_imitations() -> None:
    """Freesound originals and excluded imitations cannot enter the sketch suite."""
    rows = [
        {"row_type": "freesound_original", "included": None, "audio_decode_status": "decoded"},
        {"row_type": "imitation", "included": False, "audio_decode_status": "decoded"},
        {"row_type": "imitation", "included": True, "audio_decode_status": "decoded"},
        {"row_type": "imitation", "included": True, "audio_decode_status": "decoded"},
    ]

    assert select_vocal_rows(rows, 2) == ((2, rows[2]), (3, rows[3]))


def test_select_vocal_rows_failed_included_imitation_raises() -> None:
    """A decode failure cannot silently substitute a later curated imitation."""
    rows = [
        {"row_type": "imitation", "included": True, "audio_decode_status": "failed"},
        {"row_type": "imitation", "included": True, "audio_decode_status": "decoded"},
    ]

    with pytest.raises(ValueError, match="row 0"):
        select_vocal_rows(rows, 1)


def test_aggregate_metrics_multiple_arms_reports_per_metric_mean() -> None:
    """Each CFG arm receives independent aggregate audio statistics."""
    rows = [
        {"arm": "cfg-c0-s0", "mss": 1.0, "wmfcc": 2.0, "sot": 3.0, "rms": 0.2},
        {"arm": "cfg-c0-s0", "mss": 3.0, "wmfcc": 4.0, "sot": 5.0, "rms": 0.4},
        {"arm": "cfg-c2-s2", "mss": 8.0, "wmfcc": 7.0, "sot": 6.0, "rms": 0.9},
    ]

    aggregates = aggregate_metrics(rows)

    assert aggregates[0]["arm"] == "cfg-c0-s0"
    assert aggregates[0]["mss_mean"] == 2.0
    assert aggregates[0]["rms_mean"] == pytest.approx(0.3)
    assert aggregates[1]["arm"] == "cfg-c2-s2"
    assert aggregates[1]["mss_mean"] == 8.0
