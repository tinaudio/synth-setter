"""Benchmark production dataset generation across all Faust backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import lance
import numpy as np

from synth_setter.data.vst.core import extract_backend_version
from synth_setter.data.vst.param_spec import ParameterValues
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.shapes import PARAM_ARRAY_FIELD
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SYNTHS, SynthName

type FaustBackend = Literal["dawdreamer", "faustcpp", "faustwasm"]

_BACKENDS: tuple[FaustBackend, ...] = ("dawdreamer", "faustcpp", "faustwasm")
_DEFAULT_BLOCK_SIZE = 2_048
_DEFAULT_ROWS = 16
_DEFAULT_SEED = 1_808
_DEFAULT_TRIALS = 5
_VOLUME_ADDRESS = "/Sequencer/DSP1/brightOrgan/Main/volume"
_ORDER_CYCLE: tuple[tuple[FaustBackend, ...], ...] = (
    ("dawdreamer", "faustwasm", "faustcpp"),
    ("faustcpp", "faustwasm", "dawdreamer"),
    ("faustwasm", "faustcpp", "dawdreamer"),
    ("dawdreamer", "faustcpp", "faustwasm"),
    ("faustcpp", "dawdreamer", "faustwasm"),
    ("faustwasm", "dawdreamer", "faustcpp"),
)


@dataclass(frozen=True)
class BenchmarkSample:
    """One complete production dataset-generation timing.

    .. attribute :: trial

       Zero-based trial index.

    .. attribute :: backend

       Faust rendering backend.

    .. attribute :: wall_seconds

       End-to-end elapsed wall time.

    .. attribute :: rows_per_second

       Persisted dataset rows divided by elapsed wall time.
    """

    trial: int
    backend: FaustBackend
    wall_seconds: float
    rows_per_second: float


def balanced_backend_order(trial: int) -> tuple[FaustBackend, ...]:
    """Return a rotated, alternating host order for one trial.

    :param trial: Zero-based benchmark trial.
    :returns: Three backend names in execution order.
    :raises ValueError: If ``trial`` is negative.
    """
    if trial < 0:
        raise ValueError("trial must be non-negative")
    return _ORDER_CYCLE[trial % len(_ORDER_CYCLE)]


def summarize_samples(samples: Sequence[BenchmarkSample]) -> dict[str, float]:
    """Summarize one backend's timing samples.

    :param samples: Nonempty samples for one backend.
    :returns: Median and interquartile range for wall time and throughput.
    :raises ValueError: If no samples are supplied.
    """
    if not samples:
        raise ValueError("at least one benchmark sample is required")
    wall_seconds = np.asarray([sample.wall_seconds for sample in samples], dtype=np.float64)
    rows_per_second = np.asarray([sample.rows_per_second for sample in samples], dtype=np.float64)
    return {
        "median_wall_seconds": float(np.median(wall_seconds)),
        "wall_seconds_iqr": float(
            np.percentile(wall_seconds, 75) - np.percentile(wall_seconds, 25)
        ),
        "median_rows_per_second": float(np.median(rows_per_second)),
        "rows_per_second_iqr": float(
            np.percentile(rows_per_second, 75) - np.percentile(rows_per_second, 25)
        ),
    }


def _render_config(
    backend: FaustBackend,
    *,
    backend_version: str,
    block_size: int,
    rows: int,
    seed: int,
) -> RenderConfig:
    """Build the shared production benchmark contract.

    :param backend: Faust renderer under test.
    :param backend_version: Installed backend version.
    :param block_size: Offline block size for native and Wasm hosts.
    :param rows: Number of accepted rows in one dataset.
    :param seed: Shared parameter sampler seed.
    :returns: Validated render configuration.
    """
    return RenderConfig(
        synth=SYNTHS[SynthName("faust_bright_organ")],
        renderer_backend=backend,
        backend_version=backend_version,
        block_size=block_size if backend in {"faustcpp", "faustwasm"} else None,
        render_contract_version=2,
        sample_rate=44_100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=float("-inf"),
        samples_per_render_batch=1,
        samples_per_shard=rows,
        attempts_per_sample=50,
        base_seed=seed,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )


def _fixed_corpus(rows: int, seed: int) -> tuple[list[ParameterValues], list[ParameterValues]]:
    """Build unique, bounded midpoint patches shared by every host.

    :param rows: Number of corpus rows.
    :param seed: Volume-order seed.
    :returns: Synth patches and MIDI events in matching row order.
    """
    param_spec = resolve_param_spec(ParamSpecName("faust_bright_organ"))
    midpoint = np.full(param_spec.encoded_width, 0.5, dtype=np.float32)
    synth_midpoint, note_midpoint = param_spec.decode(midpoint)
    volumes = np.random.default_rng(seed).uniform(0.1, 0.4, size=rows)
    synth_params = [{**synth_midpoint, _VOLUME_ADDRESS: float(volume)} for volume in volumes]
    note_params = [
        {**note_midpoint, "pitch": 60, "note_start_and_end": (0.1, 0.35)} for _ in range(rows)
    ]
    return synth_params, note_params


def _run_dataset(
    root: Path,
    backend: FaustBackend,
    *,
    backend_version: str,
    block_size: int,
    synth_params: list[ParameterValues],
    note_params: list[ParameterValues],
    seed: int,
    trial: int,
) -> tuple[BenchmarkSample, np.ndarray]:
    """Generate and consume one production Lance dataset.

    :param root: Benchmark artifact directory.
    :param backend: Faust renderer under test.
    :param backend_version: Installed backend version.
    :param block_size: Offline block size for native and Wasm hosts.
    :param synth_params: Exact native synth patches shared by every host.
    :param note_params: Exact MIDI events shared by every host.
    :param seed: Shared parameter sampler seed recorded by the config.
    :param trial: Zero-based trial index.
    :returns: Timing sample and persisted normalized parameter rows.
    :raises RuntimeError: If production generation fails.
    """
    rows = len(synth_params)
    started = time.perf_counter()
    config = _render_config(
        backend,
        backend_version=backend_version,
        block_size=block_size,
        rows=rows,
        seed=seed,
    )
    dataset_path = root / f"trial-{trial + 1:02d}-{backend}.lance"
    make_lance_dataset(
        dataset_path,
        config,
        fixed_synth_params_list=synth_params,
        fixed_note_params_list=note_params,
    )
    elapsed = time.perf_counter() - started
    table = lance.dataset(str(dataset_path)).to_table(columns=[PARAM_ARRAY_FIELD])
    params = table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray().copy()
    if table.num_rows != rows:
        raise RuntimeError(f"{backend} persisted {table.num_rows} rows; expected {rows}")
    return (
        BenchmarkSample(
            trial=trial,
            backend=backend,
            wall_seconds=elapsed,
            rows_per_second=rows / elapsed,
        ),
        params,
    )


def _git_provenance() -> dict[str, str | bool]:
    """Resolve the measured source revision and worktree state.

    :returns: Full Git commit and whether tracked or untracked changes exist.
    :raises RuntimeError: If the source checkout revision cannot be inspected.
    """
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            ["git", "status", "--short"],  # noqa: S607
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("benchmark requires a readable Git source checkout") from error
    return {"commit": commit, "dirty": bool(status.strip())}


def run_benchmark(output: Path, *, rows: int, trials: int, seed: int, block_size: int) -> Path:
    """Run the balanced three-backend production benchmark.

    :param output: New or empty artifact directory.
    :param rows: Accepted rows per backend and trial.
    :param trials: Number of trials per backend.
    :param seed: Shared parameter sampler seed.
    :param block_size: Offline block size for native and Wasm hosts.
    :returns: Path to the JSON report.
    :raises ValueError: If counts are invalid or the output directory is nonempty.
    :raises RuntimeError: If parameter rows differ across backend runs.
    """
    if min(rows, trials, block_size) < 1:
        raise ValueError("rows, trials, and block_size must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"benchmark output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    versions = {backend: extract_backend_version(backend) for backend in _BACKENDS}
    synth_params, note_params = _fixed_corpus(rows, seed)
    samples: list[BenchmarkSample] = []
    reference_params: np.ndarray | None = None
    for trial in range(trials):
        for backend in balanced_backend_order(trial):
            sample, params = _run_dataset(
                output,
                backend,
                backend_version=versions[backend],
                block_size=block_size,
                synth_params=synth_params,
                note_params=note_params,
                seed=seed,
                trial=trial,
            )
            if reference_params is None:
                reference_params = params
            elif not np.array_equal(reference_params, params):
                raise RuntimeError(
                    f"normalized parameter rows differ for trial {trial + 1} {backend}"
                )
            samples.append(sample)
            sys.stdout.write(f"{json.dumps(asdict(sample))}\n")
            sys.stdout.flush()
    assert reference_params is not None
    report = {
        "schema_version": 1,
        "settings": {
            "rows_per_trial": rows,
            "trials_per_backend": trials,
            "seed": seed,
            "sample_rate": 44_100,
            "channels": 2,
            "signal_duration_seconds": 4.0,
            "samples_per_render_batch": 1,
            "block_size": block_size,
        },
        "versions": versions,
        "host": {"platform": platform.platform(), "python": sys.version},
        "source": _git_provenance(),
        "parameter_rows_sha256": hashlib.sha256(reference_params.tobytes()).hexdigest(),
        "samples": [asdict(sample) for sample in samples],
        "summary": {
            backend: summarize_samples([sample for sample in samples if sample.backend == backend])
            for backend in _BACKENDS
        },
    }
    report_path = output / "results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Faust backend benchmark CLI.

    :param argv: Optional command-line arguments excluding the executable name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS)
    parser.add_argument("--trials", type=int, default=_DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--block-size", type=int, default=_DEFAULT_BLOCK_SIZE)
    args = parser.parse_args(argv)
    report_path = run_benchmark(
        args.output,
        rows=args.rows,
        trials=args.trials,
        seed=args.seed,
        block_size=args.block_size,
    )
    sys.stdout.write(f"{report_path}\n")


if __name__ == "__main__":
    main()
