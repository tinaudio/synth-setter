"""Behavior tests for paired CLAP comparison reporting."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.dev.run_clap_comparison import build_paired_row, write_aggregate_comparison


def test_build_paired_row_candidate_lower_distance_marks_candidate_win() -> None:
    """A lower candidate cosine distance wins the paired prompt."""
    baseline = {
        "prompt": "frog croak",
        "cosine_similarity": "0.2",
        "cosine_distance": "0.8",
    }
    candidate = {
        "prompt": "frog croak",
        "cosine_similarity": "0.3",
        "cosine_distance": "0.7",
    }

    row = build_paired_row(1, baseline, candidate)

    assert row == {
        "index": 1,
        "prompt": "frog croak",
        "baseline_cosine_similarity": 0.2,
        "baseline_cosine_distance": 0.8,
        "candidate_cosine_similarity": 0.3,
        "candidate_cosine_distance": 0.7,
        "distance_delta_candidate_minus_baseline": -0.1,
        "winner": "candidate",
    }


def test_write_aggregate_comparison_two_arms_writes_statistics_and_wins(
    tmp_path: Path,
) -> None:
    """Aggregate output reports each arm's distances and paired wins.

    :param tmp_path: Isolates the generated aggregate CSV.
    """
    rows = [
        {
            "baseline_cosine_distance": 0.8,
            "candidate_cosine_distance": 0.7,
            "winner": "candidate",
        },
        {
            "baseline_cosine_distance": 0.6,
            "candidate_cosine_distance": 0.65,
            "winner": "baseline",
        },
    ]
    destination = tmp_path / "aggregate.csv"

    write_aggregate_comparison(destination, rows)

    with destination.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert written[0]["arm"] == "baseline"
    assert written[0]["mean"] == "0.7"
    assert written[0]["wins"] == "1"
    assert written[1]["arm"] == "candidate"
    assert written[1]["mean"] == "0.675"
    assert written[1]["wins"] == "1"
