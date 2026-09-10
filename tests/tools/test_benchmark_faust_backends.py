"""Faust dataset benchmark reporting contracts."""

from __future__ import annotations

from synth_setter.tools.benchmark_faust_backends import (
    BenchmarkSample,
    balanced_backend_order,
    summarize_samples,
)


def test_balanced_backend_order_five_trials_rotates_and_reverses_hosts() -> None:
    """Five trials expose every host to changing execution positions."""
    assert [balanced_backend_order(index) for index in range(5)] == [
        ("dawdreamer", "faustwasm", "faustcpp"),
        ("faustcpp", "faustwasm", "dawdreamer"),
        ("faustwasm", "faustcpp", "dawdreamer"),
        ("dawdreamer", "faustcpp", "faustwasm"),
        ("faustcpp", "dawdreamer", "faustwasm"),
    ]


def test_summarize_samples_reports_median_and_interquartile_range() -> None:
    """Summary statistics preserve wall-time and throughput units."""
    samples = [
        BenchmarkSample(
            trial=index, backend="faustcpp", wall_seconds=value, rows_per_second=16 / value
        )
        for index, value in enumerate((4.0, 8.0, 16.0, 32.0, 64.0))
    ]

    assert summarize_samples(samples) == {
        "median_wall_seconds": 16.0,
        "wall_seconds_iqr": 24.0,
        "median_rows_per_second": 1.0,
        "rows_per_second_iqr": 1.5,
    }
