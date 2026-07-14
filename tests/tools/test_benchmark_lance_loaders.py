"""Tests for the local legacy-versus-map Lance loader benchmark harness."""

import json
from collections.abc import Iterator
from itertools import product
from pathlib import Path

import pytest

from synth_setter.tools import benchmark_lance_loaders as benchmark_module
from synth_setter.tools.benchmark_lance_loaders import (
    LoaderBenchmarkResult,
    benchmark_lance_loaders,
)
from tests.helpers.lance_fixtures import write_seeded_lance_shard


@pytest.mark.slow
@pytest.mark.parametrize("configured_workers", [0, 1])
def test_benchmark_lance_loaders_runs_full_local_matrix(
    tmp_path: Path, configured_workers: int
) -> None:
    """The harness records throughput, wait time, and scans for every comparison.

    :param tmp_path: Temporary root for the local Lance fixtures.
    :param configured_workers: Configured worker count; zero must not duplicate cells.
    """
    root = tmp_path / "data"
    root.mkdir()
    for seed, split in enumerate(("train", "val", "test")):
        write_seeded_lance_shard(root / f"{split}.lance", num_rows=8, seed=seed)

    results = benchmark_lance_loaders(
        root,
        batch_size=2,
        configured_num_workers=configured_workers,
        max_batches=2,
        repetitions=2,
        random_seed=17,
    )

    expected = set(product(("legacy", "map"), ("mel", "m2l"), {0, configured_workers}))
    assert {
        (result.loader, result.conditioning, result.num_workers) for result in results
    } == expected
    for result in results:
        assert result.batches == 2
        assert result.dataset_root == str(root.resolve())
        assert result.dataset_version >= 1
        assert result.dataset_rows == 8
        assert len(result.git_revision) == 40
        assert result.batch_size == 2
        assert result.max_batches == 2
        assert result.repetitions == 2
        assert result.random_seed == 17
        assert result.persistent_workers is (result.num_workers > 0)
        assert result.median_elapsed_seconds > 0
        assert 0 < result.median_dataloader_wait_seconds <= result.median_elapsed_seconds
        assert result.batches_per_second > 0
        assert result.min_batches_per_second <= result.batches_per_second
        assert result.max_batches_per_second >= result.batches_per_second
        assert result.expected_reads_per_batch == (2.0 if result.loader == "legacy" else 1.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"configured_num_workers": -1}, "configured_num_workers must be nonnegative"),
        ({"max_batches": 0}, "max_batches must be positive"),
        ({"repetitions": 0}, "repetitions must be positive"),
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


def test_benchmark_trial_map_one_full_batch_succeeds(tmp_path: Path) -> None:
    """Map warm-up must not remove the only full batch from measurement.

    :param tmp_path: Temporary root for the one-batch Lance fixtures.
    """
    root = tmp_path / "data"
    root.mkdir()
    for seed, split in enumerate(("train", "val", "test")):
        write_seeded_lance_shard(root / f"{split}.lance", num_rows=2, seed=seed)

    result = benchmark_module._benchmark_trial(
        root,
        loader="map",
        conditioning="mel",
        num_workers=0,
        batch_size=2,
        max_batches=1,
    )

    assert result.batches == 1


def test_benchmark_trial_reuses_persistent_worker_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm-up and measurement share one loader and its worker processes.

    :param tmp_path: Arbitrary dataset root accepted by the fake module.
    :param monkeypatch: Fixture replacing the datamodule with an observable fake.
    """

    class FakeLoader:
        def __init__(self) -> None:
            self.persistent_workers = False
            self.iterations = 0

        def __iter__(self) -> Iterator[object]:
            self.iterations += 1
            return iter((object(),))

    class FakeModule:
        instance: "FakeModule | None" = None

        def __init__(self, **kwargs: object) -> None:
            self.loader = FakeLoader()
            self.loader.persistent_workers = bool(kwargs["persistent_workers"])
            self.loader_requests = 0
            FakeModule.instance = self

        def setup(self, _stage: str) -> None:
            pass

        def train_dataloader(self) -> FakeLoader:
            self.loader_requests += 1
            return self.loader

        def teardown(self, _stage: str) -> None:
            pass

    monkeypatch.setattr(benchmark_module, "LanceVSTDataModule", FakeModule)

    benchmark_module._benchmark_trial(
        tmp_path,
        loader="map",
        conditioning="mel",
        num_workers=2,
        batch_size=2,
        max_batches=1,
    )

    assert FakeModule.instance is not None
    assert FakeModule.instance.loader_requests == 1
    assert FakeModule.instance.loader.iterations == 2
    assert FakeModule.instance.loader.persistent_workers


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
    result = LoaderBenchmarkResult(
        loader="map",
        conditioning="mel",
        num_workers=0,
        dataset_root="/data",
        dataset_version=3,
        dataset_rows=16,
        git_revision="a" * 40,
        batch_size=8,
        max_batches=3,
        repetitions=2,
        random_seed=11,
        persistent_workers=False,
        batches=2,
        median_elapsed_seconds=1.0,
        median_dataloader_wait_seconds=0.5,
        batches_per_second=2.0,
        min_batches_per_second=1.5,
        max_batches_per_second=2.5,
        expected_reads_per_batch=1.0,
    )
    calls: list[tuple[Path, int, int, int, int, int]] = []

    def fake_benchmark(
        dataset_root: str | Path,
        *,
        batch_size: int,
        configured_num_workers: int,
        max_batches: int,
        repetitions: int,
        random_seed: int,
    ) -> list[LoaderBenchmarkResult]:
        calls.append(
            (
                Path(dataset_root),
                batch_size,
                configured_num_workers,
                max_batches,
                repetitions,
                random_seed,
            )
        )
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
            "--repetitions",
            "2",
            "--seed",
            "11",
        ],
    )

    benchmark_module.main()

    assert calls == [(dataset_root, 8, 2, 3, 2, 11)]
    assert json.loads(capsys.readouterr().out) == [
        {
            "loader": "map",
            "conditioning": "mel",
            "num_workers": 0,
            "dataset_root": "/data",
            "dataset_version": 3,
            "dataset_rows": 16,
            "git_revision": "a" * 40,
            "batch_size": 8,
            "max_batches": 3,
            "repetitions": 2,
            "random_seed": 11,
            "persistent_workers": False,
            "batches": 2,
            "median_elapsed_seconds": 1.0,
            "median_dataloader_wait_seconds": 0.5,
            "batches_per_second": 2.0,
            "min_batches_per_second": 1.5,
            "max_batches_per_second": 2.5,
            "expected_reads_per_batch": 1.0,
        }
    ]


def test_main_runs_real_lance_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI drives real Lance reads and emits one unique zero-worker matrix.

    :param tmp_path: Temporary root for local Lance fixtures.
    :param monkeypatch: Fixture supplying CLI arguments.
    :param capsys: Fixture capturing the JSON output.
    """
    root = tmp_path / "data"
    root.mkdir()
    for seed, split in enumerate(("train", "val", "test")):
        write_seeded_lance_shard(root / f"{split}.lance", num_rows=6, seed=seed)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_lance_loaders",
            str(root),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--max-batches",
            "1",
            "--repetitions",
            "1",
            "--seed",
            "23",
        ],
    )

    benchmark_module.main()

    records = json.loads(capsys.readouterr().out)
    assert len(records) == 4
    expected = set(product(("legacy", "map"), ("mel", "m2l"), (0,)))
    assert {
        (record["loader"], record["conditioning"], record["num_workers"]) for record in records
    } == expected
    assert all(record["dataset_rows"] == 6 for record in records)
    assert all(record["random_seed"] == 23 for record in records)
    assert not any(record["persistent_workers"] for record in records)
