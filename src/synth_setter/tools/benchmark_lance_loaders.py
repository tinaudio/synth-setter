"""Benchmark the legacy and map-style Lance dataloaders on local splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from synth_setter.conditioning import ConditioningMode
from synth_setter.data.lance_datamodule import LanceVSTDataModule

LoaderName = Literal["legacy", "map"]


@dataclass(frozen=True)
class LoaderBenchmarkResult:
    """Measurements from one loader, conditioning, and worker-count run.

    .. attribute :: loader
        Lance implementation under test.
    .. attribute :: conditioning
        Projected conditioning column.
    .. attribute :: num_workers
        Dataloader worker count.
    .. attribute :: batches
        Number of materialized batches.
    .. attribute :: elapsed_seconds
        Total run duration.
    .. attribute :: dataloader_wait_seconds
        Time spent awaiting batches.
    .. attribute :: batches_per_second
        Materialized batch throughput.
    .. attribute :: scan_count_per_batch
        Projected Lance reads per batch.
    """

    loader: LoaderName
    conditioning: ConditioningMode
    num_workers: int
    batches: int
    elapsed_seconds: float
    dataloader_wait_seconds: float
    batches_per_second: float
    scan_count_per_batch: float


def _benchmark_one(
    dataset_root: Path,
    *,
    loader: LoaderName,
    conditioning: ConditioningMode,
    num_workers: int,
    batch_size: int,
    max_batches: int,
) -> LoaderBenchmarkResult:
    """Measure one local loader configuration.

    :param dataset_root: Directory containing local ``train/val/test.lance`` splits.
    :param loader: Lance implementation under test.
    :param conditioning: Projected conditioning column.
    :param num_workers: Dataloader worker count.
    :param batch_size: Rows requested per batch.
    :param max_batches: Maximum batches to materialize.
    :return: Timing and projected-read measurements for the run.
    :raises ValueError: If the training split produces no full batch.
    """
    module = LanceVSTDataModule(
        dataset_root=dataset_root,
        use_saved_mean_and_variance=False,
        batch_size=batch_size,
        ot=False,
        num_workers=num_workers,
        conditioning=conditioning,
        pin_memory=False,
        param_spec_name="surge_xt",
        loader=loader,
    )
    module.setup("fit")
    batches = 0
    wait_seconds = 0.0
    started = time.perf_counter()
    try:
        iterator = iter(module.train_dataloader())
        for _ in range(max_batches):
            wait_started = time.perf_counter()
            try:
                next(iterator)
            except StopIteration:
                break
            wait_seconds += time.perf_counter() - wait_started
            batches += 1
    finally:
        elapsed = time.perf_counter() - started
        module.teardown("fit")
    if batches == 0:
        raise ValueError("benchmark dataset produced no full training batches")

    # The legacy adapter performs one Lance read per requested column; train
    # projects params plus one conditioning column. The map path performs one
    # projected take for the same batch.
    scan_count = 2 * batches if loader == "legacy" else batches
    return LoaderBenchmarkResult(
        loader=loader,
        conditioning=conditioning,
        num_workers=num_workers,
        batches=batches,
        elapsed_seconds=elapsed,
        dataloader_wait_seconds=wait_seconds,
        batches_per_second=batches / elapsed,
        scan_count_per_batch=scan_count / batches,
    )


def benchmark_lance_loaders(
    dataset_root: str | Path,
    *,
    batch_size: int,
    configured_num_workers: int,
    max_batches: int,
) -> list[LoaderBenchmarkResult]:
    """Run the Phase-2 comparison matrix on a local Lance fixture.

    :param dataset_root: Directory containing local ``train/val/test.lance`` splits.
    :param batch_size: Rows requested per batch.
    :param configured_num_workers: Production-like worker count compared with zero.
    :param max_batches: Maximum batches materialized per matrix cell.
    :return: Results for legacy/map, mel/m2l, and zero/configured workers.
    :raises ValueError: If numeric arguments are invalid or a run yields no batch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if configured_num_workers < 0:
        raise ValueError("configured_num_workers must be nonnegative")
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    worker_counts = (0, configured_num_workers)
    return [
        _benchmark_one(
            Path(dataset_root),
            loader=loader,
            conditioning=conditioning,
            num_workers=num_workers,
            batch_size=batch_size,
            max_batches=max_batches,
        )
        for loader in ("legacy", "map")
        for conditioning in ("mel", "m2l")
        for num_workers in worker_counts
    ]


def main() -> None:
    """Run the local comparison matrix and print JSON records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()
    results = benchmark_lance_loaders(
        args.dataset_root,
        batch_size=args.batch_size,
        configured_num_workers=args.num_workers,
        max_batches=args.max_batches,
    )
    sys.stdout.write(json.dumps([asdict(result) for result in results], indent=2) + "\n")


if __name__ == "__main__":
    main()
