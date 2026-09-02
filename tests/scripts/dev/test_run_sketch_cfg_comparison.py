"""Behavior tests for the sketch CFG suite runner."""

import pytest

from scripts.dev.run_sketch_cfg_comparison import aggregate_metrics, select_vocal_rows


def test_select_vocal_rows_decoded_rows_returns_first_stored_rows() -> None:
    """Vocal sketches are the exact stored-order prefix."""
    rows = [
        {"row_id": "b", "audio_decode_status": "decoded"},
        {"row_id": "a", "audio_decode_status": "decoded"},
        {"row_id": "c", "audio_decode_status": "decoded"},
    ]

    assert select_vocal_rows(rows, 2) == ((0, rows[0]), (1, rows[1]))


def test_select_vocal_rows_failed_prefix_row_raises() -> None:
    """A decode failure cannot silently substitute a later vocal row."""
    rows = [
        {"row_id": "a", "audio_decode_status": "failed"},
        {"row_id": "b", "audio_decode_status": "decoded"},
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
