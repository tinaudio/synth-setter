"""Faust dataset benchmark reporting contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from synth_setter.tools.benchmark_faust_backends import (
    BenchmarkSample,
    balanced_backend_order,
    main,
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


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("faust") is None or shutil.which("g++") is None,
    reason="install the Faust CLI and g++",
)
def test_benchmark_cli_generates_and_consumes_each_backend_dataset(tmp_path: Path) -> None:
    """The public CLI writes a complete report from three real production datasets.

    :param tmp_path: Isolated benchmark artifact root.
    """
    output = tmp_path / "benchmark"
    main(["--output", str(output), "--rows", "1", "--trials", "1", "--block-size", "64"])

    report = json.loads((output / "results.json").read_text())
    assert report["settings"]["rows_per_trial"] == 1
    assert report["settings"]["trials_per_backend"] == 1
    assert set(report["versions"]) == {"dawdreamer", "faustcpp", "faustwasm"}
    assert len(report["source"]["commit"]) == 40
    assert isinstance(report["source"]["dirty"], bool)
    assert len(report["samples"]) == 3
    assert set(report["summary"]) == {"dawdreamer", "faustcpp", "faustwasm"}
