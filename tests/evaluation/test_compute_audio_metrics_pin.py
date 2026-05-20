"""Pin tests for ``compute_audio_metrics`` CLI — wandb-metrics plan Phase 0.

These golden assertions lock down the pre-refactor CLI shape so the Phase 1 split
into ``compute_metrics()`` + ``write_metrics_csvs()`` library helpers can be
proven not to regress:

* file layout: ``metrics-{pid}.csv``, ``metrics.csv``, ``aggregated_metrics.csv``
  land at the expected paths
* schema: the four-metric column set is exactly ``{mss, wmfcc, sot, rms}``
* scalar values: the committed snapshot in ``snapshots/`` matches within a
  lenient tolerance band (per-metric ``rel=1e-2``).

The snapshot file is committed — it is **not** regenerated each run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from synth_setter.evaluation.compute_audio_metrics import main as compute_audio_metrics_main

_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "compute_audio_metrics_aggregated.csv"
_PER_SAMPLE_SNAPSHOT_PATH = (
    Path(__file__).parent / "snapshots" / "compute_audio_metrics_per_sample.csv"
)
_EXPECTED_METRIC_COLUMNS = {"mss", "wmfcc", "sot", "rms"}


@pytest.fixture(scope="module")
def cli_metrics_dir(fixture_audio_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the CLI once per module and yield the metrics directory it wrote.

    Single ``-w 1`` worker keeps the ``metrics-{pid}.csv`` filename count
    deterministic (exactly one) so file-layout assertions stay simple.

    :param fixture_audio_dir: Session-scoped audio dir from ``conftest.py``.
    :param tmp_path_factory: Pytest fixture providing session-scoped tmp paths.
    :return: Directory the CLI wrote its three CSVs into.
    """
    metrics_dir = tmp_path_factory.mktemp("pin_metrics")
    runner = CliRunner()
    result = runner.invoke(
        compute_audio_metrics_main,
        [str(fixture_audio_dir), str(metrics_dir), "-w", "1"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return metrics_dir


def test_cli_writes_expected_files(cli_metrics_dir: Path) -> None:
    """Required filenames land in the output dir: per-worker, joined, aggregated.

    :param cli_metrics_dir: Output directory from the module-scoped CLI invocation.
    """
    per_worker = list(cli_metrics_dir.glob("metrics-*.csv"))
    assert len(per_worker) == 1, f"expected one metrics-<pid>.csv, got {per_worker}"

    assert (cli_metrics_dir / "metrics.csv").is_file()
    assert (cli_metrics_dir / "aggregated_metrics.csv").is_file()


def test_aggregated_csv_columns(cli_metrics_dir: Path) -> None:
    """``aggregated_metrics.csv`` has ``mean``/``std`` columns and the four-metric row index.

    :param cli_metrics_dir: Output directory from the module-scoped CLI invocation.
    """
    agg = pd.read_csv(cli_metrics_dir / "aggregated_metrics.csv", index_col=0)
    assert list(agg.columns) == ["mean", "std"]
    assert set(agg.index) == _EXPECTED_METRIC_COLUMNS


def test_aggregated_scalar_values_within_tolerance(cli_metrics_dir: Path) -> None:
    """Mean/std scalars match the committed snapshot within ``rel=1e-2`` per metric.

    The tolerance is intentionally lenient — librosa / pedalboard / pesto have minor cross-version
    float drift on identical inputs. The point of the pin is to catch *shape* regressions (e.g. a
    missing metric, an off-by-one mean) rather than to assert bit-for-bit numerical identity.

    ``rms`` is asserted on a pure relative band: its snapshot std (~1.7e-6) sits at the same
    order of magnitude as a generic ``abs=1e-6`` floor, so a shared ``abs`` would let real
    regressions slip through silently.

    :param cli_metrics_dir: Output directory from the module-scoped CLI invocation.
    """
    agg = pd.read_csv(cli_metrics_dir / "aggregated_metrics.csv", index_col=0)
    snapshot = pd.read_csv(_SNAPSHOT_PATH, index_col=0)

    assert set(agg.index) == set(snapshot.index)
    for metric in snapshot.index:
        if metric == "rms":
            assert agg.loc[metric, "mean"] == pytest.approx(
                snapshot.loc[metric, "mean"], rel=1e-2
            ), f"{metric} mean drifted"
            assert agg.loc[metric, "std"] == pytest.approx(
                snapshot.loc[metric, "std"], rel=1e-2
            ), f"{metric} std drifted"
        else:
            assert agg.loc[metric, "mean"] == pytest.approx(
                snapshot.loc[metric, "mean"], rel=1e-2, abs=1e-6
            ), f"{metric} mean drifted"
            assert agg.loc[metric, "std"] == pytest.approx(
                snapshot.loc[metric, "std"], rel=1e-2, abs=1e-6
            ), f"{metric} std drifted"


def test_metrics_csv_per_sample_values(cli_metrics_dir: Path) -> None:
    """``metrics.csv`` row count, sample-index set, and per-cell values match the snapshot.

    Phase 1 extraction may legitimately reshape this CSV; if so, the refactor PR
    updates this test alongside the production change so the diff makes the
    behavior change explicit. Same ``rel=1e-2`` band as the aggregated snapshot
    (``rms`` is pure-rel because its magnitude sits near a generic ``abs=1e-6`` floor).

    Row ordering is filesystem-dependent (``Path.glob`` traversal order varies
    across Linux ext4 vs. conda-runner btrfs vs. macOS), so the pin asserts the
    *sorted* index and looks up cells by sample id rather than positional row.

    :param cli_metrics_dir: Output directory from the module-scoped CLI invocation.
    """
    actual = pd.read_csv(cli_metrics_dir / "metrics.csv", index_col=0).sort_index()
    snapshot = pd.read_csv(_PER_SAMPLE_SNAPSHOT_PATH, index_col=0).sort_index()

    assert len(actual) == 2, f"expected 2 rows, got {len(actual)}"
    assert list(actual.index) == list(snapshot.index), (
        f"sample-index set drifted: {list(actual.index)} vs {list(snapshot.index)}"
    )
    assert set(actual.columns) == _EXPECTED_METRIC_COLUMNS

    for sample_idx in snapshot.index:
        for metric in _EXPECTED_METRIC_COLUMNS:
            expected = snapshot.loc[sample_idx, metric]
            got = actual.loc[sample_idx, metric]
            if metric == "rms":
                assert got == pytest.approx(expected, rel=1e-2), (
                    f"sample {sample_idx} {metric} drifted: {got} vs {expected}"
                )
            else:
                assert got == pytest.approx(expected, rel=1e-2, abs=1e-6), (
                    f"sample {sample_idx} {metric} drifted: {got} vs {expected}"
                )
