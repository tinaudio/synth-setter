"""Faust dataset benchmark reporting contracts."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import ParameterValues
from synth_setter.tools import benchmark_faust_backends as benchmark
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


def test_git_provenance_reports_commit_and_dirty_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source provenance retains the measured commit and pre-run worktree state.

    :param monkeypatch: Replaces Git subprocess output.
    """
    responses = iter(
        [
            subprocess.CompletedProcess(["git"], 0, stdout=f"{'a' * 40}\n", stderr=""),
            subprocess.CompletedProcess(["git"], 0, stdout=" M tracked.py\n", stderr=""),
        ]
    )
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_args, **_kwargs: next(responses))

    assert benchmark._git_provenance() == {"commit": "a" * 40, "dirty": True}


def test_run_dataset_times_and_consumes_persisted_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One backend timing consumes the normalized rows written to Lance.

    :param tmp_path: Isolated dataset destination.
    :param monkeypatch: Replaces Lance storage while retaining runner behavior.
    """
    synth_params, note_params = benchmark._fixed_corpus(2, 1808)
    persisted = np.full((2, 13), 0.5, dtype=np.float32)

    column = MagicMock()
    column.combine_chunks.return_value.to_numpy_ndarray.return_value = persisted
    table = MagicMock(num_rows=2)
    table.column.return_value = column
    dataset = MagicMock()
    dataset.to_table.return_value = table
    monkeypatch.setattr(benchmark, "make_lance_dataset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark.lance, "dataset", lambda _path: dataset)

    sample, params = benchmark._run_dataset(
        tmp_path,
        "dawdreamer",
        backend_version="0.8.3",
        block_size=64,
        synth_params=synth_params,
        note_params=note_params,
        seed=1808,
        trial=1,
    )

    assert sample.backend == "dawdreamer"
    assert sample.trial == 1
    assert sample.wall_seconds > 0.0
    assert sample.rows_per_second > 0.0
    np.testing.assert_array_equal(params, persisted)


def test_run_benchmark_writes_balanced_report_without_real_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator writes provenance, timings, and matching row identity.

    :param tmp_path: Isolated report destination.
    :param monkeypatch: Replaces host execution while preserving orchestration.
    """
    monkeypatch.setattr(benchmark, "extract_backend_version", lambda backend: f"{backend}-1")
    monkeypatch.setattr(benchmark, "_git_provenance", lambda: {"commit": "a" * 40, "dirty": False})

    def run_dataset(
        _root: Path,
        backend: benchmark.FaustBackend,
        *,
        backend_version: str,
        block_size: int,
        synth_params: list[ParameterValues],
        note_params: list[ParameterValues],
        seed: int,
        trial: int,
    ) -> tuple[BenchmarkSample, np.ndarray]:
        del backend_version, block_size, note_params, seed, trial
        rows = len(synth_params)
        return BenchmarkSample(0, backend, 2.0, rows / 2.0), np.zeros((rows, 13), dtype=np.float32)

    monkeypatch.setattr(benchmark, "_run_dataset", run_dataset)
    report_path = benchmark.run_benchmark(
        tmp_path / "benchmark", rows=2, trials=2, seed=1808, block_size=64
    )

    report = json.loads(report_path.read_text())
    assert report["source"] == {"commit": "a" * 40, "dirty": False}
    assert report["settings"]["signal_duration_seconds"] == 4.0
    assert len(report["samples"]) == 6
    assert set(report["summary"]) == {"dawdreamer", "faustcpp", "faustwasm"}


def test_main_forwards_cli_arguments_and_prints_report_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI arguments reach the benchmark runner and expose its report path.

    :param tmp_path: Isolated requested output directory.
    :param monkeypatch: Replaces the expensive benchmark runner.
    :param capsys: Captures the operator-facing report path.
    """
    output = tmp_path / "benchmark"
    captured: dict[str, object] = {}

    def run_benchmark(path: Path, **settings: int) -> Path:
        captured.update({"output": path, **settings})
        return path / "results.json"

    monkeypatch.setattr(benchmark, "run_benchmark", run_benchmark)
    main(
        [
            "--output",
            str(output),
            "--rows",
            "2",
            "--trials",
            "3",
            "--seed",
            "5",
            "--block-size",
            "64",
        ]
    )

    assert captured == {
        "output": output,
        "rows": 2,
        "trials": 3,
        "seed": 5,
        "block_size": 64,
    }
    assert capsys.readouterr().out == f"{output / 'results.json'}\n"


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
    entrypoint = shutil.which("synth-setter-benchmark-faust-backends")
    assert entrypoint is not None
    completed = subprocess.run(  # noqa: S603 — installed entrypoint with controlled arguments
        [
            entrypoint,
            "--output",
            str(output),
            "--rows",
            "1",
            "--trials",
            "1",
            "--block-size",
            "64",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert str(output / "results.json") in completed.stdout
    report = json.loads((output / "results.json").read_text())
    assert report["settings"]["rows_per_trial"] == 1
    assert report["settings"]["trials_per_backend"] == 1
    assert set(report["versions"]) == {"dawdreamer", "faustcpp", "faustwasm"}
    assert len(report["source"]["commit"]) == 40
    assert isinstance(report["source"]["dirty"], bool)
    assert len(report["samples"]) == 3
    assert set(report["summary"]) == {"dawdreamer", "faustcpp", "faustwasm"}
