"""Tests for the local legacy-versus-map Lance loader benchmark harness."""

import json
from pathlib import Path

import pytest

from synth_setter.tools import benchmark_lance_loaders as benchmark_module
from synth_setter.tools.benchmark_lance_loaders import (
    LoaderBenchmarkResult,
    benchmark_lance_loaders,
)
from tests.helpers.lance_fixtures import write_seeded_lance_shard


@pytest.mark.slow
def test_benchmark_lance_loaders_runs_full_local_matrix(tmp_path: Path) -> None:
    """The harness records throughput, wait time, and scans for every comparison.

    :param tmp_path: Temporary root for the local Lance fixtures.
    """
    root = tmp_path / "data"
    root.mkdir()
    for seed, split in enumerate(("train", "val", "test")):
        write_seeded_lance_shard(root / f"{split}.lance", num_rows=8, seed=seed)

    results = benchmark_lance_loaders(
        root,
        batch_size=2,
        configured_num_workers=1,
        max_batches=2,
    )

    assert {(result.loader, result.conditioning, result.num_workers) for result in results} == {
        (loader, conditioning, workers)
        for loader in ("legacy", "map")
        for conditioning in ("mel", "m2l")
        for workers in (0, 1)
    }
    for result in results:
        assert result.batches == 2
        assert result.elapsed_seconds > 0
        assert 0 < result.dataloader_wait_seconds <= result.elapsed_seconds
        assert result.batches_per_second > 0
        assert result.scan_count_per_batch == (2.0 if result.loader == "legacy" else 1.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"configured_num_workers": -1}, "configured_num_workers must be nonnegative"),
        ({"max_batches": 0}, "max_batches must be positive"),
    ],
)
def test_benchmark_lance_loaders_invalid_numeric_argument_raises(
    tmp_path: Path, overrides: dict[str, int], message: str
) -> None:
    """Invalid matrix dimensions fail before opening a dataset.

    :param tmp_path: Arbitrary dataset root that must remain unopened.
    :param overrides: Invalid argument replacing one valid default.
    :param message: Expected validation error text.
    """
    arguments = {
        "batch_size": 1,
        "configured_num_workers": 1,
        "max_batches": 1,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        benchmark_lance_loaders(tmp_path, **arguments)


def test_benchmark_lance_loaders_no_full_batch_raises(tmp_path: Path) -> None:
    """A split smaller than the requested batch cannot produce a measurement.

    :param tmp_path: Temporary root for the undersized Lance fixtures.
    """
    root = tmp_path / "data"
    root.mkdir()
    for seed, split in enumerate(("train", "val", "test")):
        write_seeded_lance_shard(root / f"{split}.lance", num_rows=1, seed=seed)

    with pytest.raises(ValueError, match="produced no full training batches"):
        benchmark_lance_loaders(
            root,
            batch_size=2,
            configured_num_workers=0,
            max_batches=1,
        )


def test_main_writes_benchmark_results_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI forwards parsed dimensions and emits machine-readable records.

    :param tmp_path: Arbitrary dataset path forwarded by the CLI.
    :param monkeypatch: Fixture replacing CLI arguments and benchmark execution.
    :param capsys: Fixture capturing the JSON output.
    """
    result = LoaderBenchmarkResult("map", "mel", 0, 2, 1.0, 0.5, 2.0, 1.0)
    calls: list[tuple[Path, int, int, int]] = []

    def fake_benchmark(
        dataset_root: str | Path,
        *,
        batch_size: int,
        configured_num_workers: int,
        max_batches: int,
    ) -> list[LoaderBenchmarkResult]:
        calls.append((Path(dataset_root), batch_size, configured_num_workers, max_batches))
        return [result]

    monkeypatch.setattr(benchmark_module, "benchmark_lance_loaders", fake_benchmark)
    dataset_root = tmp_path / "lance"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_lance_loaders",
            str(dataset_root),
            "--batch-size",
            "8",
            "--num-workers",
            "2",
            "--max-batches",
            "3",
        ],
    )

    benchmark_module.main()

    assert calls == [(dataset_root, 8, 2, 3)]
    assert json.loads(capsys.readouterr().out) == [
        {
            "loader": "map",
            "conditioning": "mel",
            "num_workers": 0,
            "batches": 2,
            "elapsed_seconds": 1.0,
            "dataloader_wait_seconds": 0.5,
            "batches_per_second": 2.0,
            "scan_count_per_batch": 1.0,
        }
    ]
