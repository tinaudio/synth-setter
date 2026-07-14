"""Benchmark the legacy and map-style Lance dataloaders on local splits.

Example::

    uv run python -m synth_setter.tools.benchmark_lance_loaders ./data \
        --batch-size 128 --num-workers 4 --max-batches 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Literal, TypeAlias

import lance

from synth_setter.conditioning import ConditioningMode
from synth_setter.data.lance_datamodule import LanceVSTDataModule
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import _get_git_sha

LoaderName: TypeAlias = Literal["legacy", "map"]


@dataclass(frozen=True)
class LoaderBenchmarkResult:
    """Measurements from one loader, conditioning, and worker-count run.

    .. attribute :: loader
        Lance implementation under test.
    .. attribute :: conditioning
        Projected conditioning column.
    .. attribute :: num_workers
        Dataloader worker count.
    .. attribute :: dataset_root
        Resolved path identifying the measured dataset.
    .. attribute :: dataset_version
        Lance transaction version read by the run.
    .. attribute :: dataset_rows
        Rows in the measured training split.
    .. attribute :: git_revision
        Source revision that produced the measurement.
    .. attribute :: batch_size
        Rows requested per batch.
    .. attribute :: max_batches
        Maximum measured batches per trial.
    .. attribute :: repetitions
        Warmed trials summarized by the record.
    .. attribute :: batches
        Number of materialized batches.
    .. attribute :: median_elapsed_seconds
        Median warmed-trial duration.
    .. attribute :: median_dataloader_wait_seconds
        Median time spent awaiting batches after warm-up.
    .. attribute :: batches_per_second
        Median materialized batch throughput.
    .. attribute :: min_batches_per_second
        Minimum trial throughput.
    .. attribute :: max_batches_per_second
        Maximum trial throughput.
    .. attribute :: expected_reads_per_batch
        Static read model: projected columns for legacy, one take for map.
    """

    loader: LoaderName
    conditioning: ConditioningMode
    num_workers: int
    dataset_root: str
    dataset_version: int
    dataset_rows: int
    git_revision: str
    batch_size: int
    max_batches: int
    repetitions: int
    batches: int
    median_elapsed_seconds: float
    median_dataloader_wait_seconds: float
    batches_per_second: float
    min_batches_per_second: float
    max_batches_per_second: float
    expected_reads_per_batch: float


@dataclass(frozen=True)
class _Trial:
    """Timing values from one warmed benchmark trial.

    .. attribute :: batches
        Materialized batches.
    .. attribute :: elapsed_seconds
        Measured duration.
    .. attribute :: wait_seconds
        Time spent awaiting batches.
    """

    batches: int
    elapsed_seconds: float
    wait_seconds: float


def _benchmark_trial(
    dataset_root: Path,
    *,
    loader: LoaderName,
    conditioning: ConditioningMode,
    num_workers: int,
    batch_size: int,
    max_batches: int,
) -> _Trial:
    """Measure one local loader configuration.

    :param dataset_root: Directory containing local ``train/val/test.lance`` splits.
    :param loader: Lance implementation under test.
    :param conditioning: Projected conditioning column.
    :param num_workers: Dataloader worker count.
    :param batch_size: Rows requested per batch.
    :param max_batches: Maximum batches to materialize.
    :return: Timing measurements for one warmed run.
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
        param_spec_name=ParamSpecName("surge_xt"),
        loader=loader,
        persistent_workers=num_workers > 0,
    )
    module.setup("fit")
    try:
        data_loader = module.train_dataloader()
        warm_iterator = iter(data_loader)
        try:
            next(warm_iterator)
        except StopIteration as error:
            raise ValueError("benchmark dataset produced no full training batches") from error
        iterator = iter(data_loader)
        batches = 0
        wait_seconds = 0.0
        started = time.perf_counter()
        for _ in range(max_batches):
            wait_started = time.perf_counter()
            try:
                next(iterator)
            except StopIteration:
                break
            wait_seconds += time.perf_counter() - wait_started
            batches += 1
        elapsed = time.perf_counter() - started
    finally:
        module.teardown("fit")
    if batches == 0:
        raise ValueError("benchmark dataset produced no full training batches")
    return _Trial(batches=batches, elapsed_seconds=elapsed, wait_seconds=wait_seconds)


def benchmark_lance_loaders(
    dataset_root: str | Path,
    *,
    batch_size: int,
    configured_num_workers: int,
    max_batches: int,
    repetitions: int = 3,
) -> list[LoaderBenchmarkResult]:
    """Run the Phase-2 comparison matrix on a local Lance fixture.

    :param dataset_root: Directory containing local ``train/val/test.lance`` splits.
    :param batch_size: Rows requested per batch.
    :param configured_num_workers: Production-like worker count compared with zero.
    :param max_batches: Maximum batches materialized per matrix cell.
    :param repetitions: Warmed trials per cell used for median and spread.
    :return: Results for legacy/map, mel/m2l, and zero/configured workers.
    :raises ValueError: If numeric arguments are invalid or a run yields no batch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if configured_num_workers < 0:
        raise ValueError("configured_num_workers must be nonnegative")
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    worker_counts = tuple(dict.fromkeys((0, configured_num_workers)))
    configurations: list[tuple[LoaderName, ConditioningMode, int]] = []
    for loader in ("legacy", "map"):
        for conditioning in ("mel", "m2l"):
            for num_workers in worker_counts:
                configurations.append((loader, conditioning, num_workers))
    trials: dict[tuple[LoaderName, ConditioningMode, int], list[_Trial]] = {
        configuration: [] for configuration in configurations
    }
    rng = random.SystemRandom()
    root = Path(dataset_root).resolve()
    for _ in range(repetitions):
        round_order = configurations.copy()
        rng.shuffle(round_order)
        for loader, conditioning, num_workers in round_order:
            trials[(loader, conditioning, num_workers)].append(
                _benchmark_trial(
                    root,
                    loader=loader,
                    conditioning=conditioning,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    max_batches=max_batches,
                )
            )

    train = lance.dataset(root / "train.lance")
    revision = _get_git_sha()
    results = []
    for loader, conditioning, num_workers in configurations:
        cell = trials[(loader, conditioning, num_workers)]
        throughputs = [trial.batches / trial.elapsed_seconds for trial in cell]
        results.append(
            LoaderBenchmarkResult(
                loader=loader,
                conditioning=conditioning,
                num_workers=num_workers,
                dataset_root=str(root),
                dataset_version=train.version,
                dataset_rows=train.count_rows(),
                git_revision=revision,
                batch_size=batch_size,
                max_batches=max_batches,
                repetitions=repetitions,
                batches=cell[0].batches,
                median_elapsed_seconds=median(trial.elapsed_seconds for trial in cell),
                median_dataloader_wait_seconds=median(trial.wait_seconds for trial in cell),
                batches_per_second=median(throughputs),
                min_batches_per_second=min(throughputs),
                max_batches_per_second=max(throughputs),
                expected_reads_per_batch=2.0 if loader == "legacy" else 1.0,
            )
        )
    return results


def main() -> None:
    """Run the local comparison matrix and print JSON records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    results = benchmark_lance_loaders(
        args.dataset_root,
        batch_size=args.batch_size,
        configured_num_workers=args.num_workers,
        max_batches=args.max_batches,
        repetitions=args.repetitions,
    )
    sys.stdout.write(json.dumps([asdict(result) for result in results], indent=2) + "\n")


if __name__ == "__main__":
    main()
