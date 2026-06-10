"""One-command orchestrator for the #489 surge_xt cadence investigation.

Runs the whole investigation from one script: (1) generates the fixed surge_xt
copy-source dataset, (2) derives its R2 run-root URI, and (3) drives the five
experiments — two within-run probes (render-order shuffle, reuse-depth junk) and
three paired-copy probes (reload cadence, gui cadence, reproducibility floor) —
feeding the derived URI to the copy probes so nothing hardcodes where copies
read from.

One :class:`Scale` feeds both the source generation and the copy probes, so the
copy-preflight match set (``param_spec_name`` / ``samples_per_shard`` /
``train_val_test_sizes``) cannot drift between producer and consumer.

On the wandb launcher every selected sweep is created before any agent runs, then
agents run concurrently as ``wandb agent`` subprocesses under a ``--max-parallel``
cap — so all sweeps land in the UI even if an agent stalls, and one stalled agent
never blocks the others. The orchestrator stays alive supervising the pool; run it
under ``tmux``/``nohup`` so a disconnect does not orphan in-flight agents.

Run the whole investigation::

    python -m synth_setter.tools.cadence_investigation_489               # wandb sweeps + agents
    python -m synth_setter.tools.cadence_investigation_489 --dry-run     # print the plan only
    python -m synth_setter.tools.cadence_investigation_489 --launcher local   # single-box subprocess loop
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import wandb
import wandb.env
from loguru import logger

from synth_setter.pipeline.schemas.prefix import DEFAULT_R2_PREFIX_ROOT, make_r2_prefix

# Temporary: the ``tinaudio`` W&B entity does not exist yet (sweep creation 404s),
# so runs target this working entity until #1560 stands up ``tinaudio``.
ENTITY = "khaledtinubu-n-a"
PROJECT = "synth-setter"
BUCKET = "intermediate-data"
# wandb's ``program:`` field is repo-relative; the local launcher imports by module path.
PROGRAM = "src/synth_setter/cli/generate_dataset.py"
_GENERATE_MODULE = "synth_setter.cli.generate_dataset"

PARAM_SPEC = "surge_xt"
PRESET = "presets/surge-base.vstpreset"

# Fixed reference run identity -> stable copy-source URI the copy probes replay.
REFERENCE_TASK = "ref-surge-xt-489"
REFERENCE_RUN_ID = "paired-ref-v1"

# Copy can't resample a sub-floor re-render (#724); -1000 dB lets junk/quiet
# renders reach the oracle while only true silence (-inf) trips the floor.
COPY_MIN_LOUDNESS = -1000.0

# Fresh-param experiments occasionally sample a silent (-inf) patch, which is
# skipped regardless of the floor; retries resample past it. Copy probes replay
# vetted params and can't resample, so they keep the generate default of 0.
MAX_RETRIES = 10

_GENERATE_EXPERIMENT = "generate_dataset/smoke-shard-with-oracle-eval"
# Source needs raw param shards only (copy reads same-named shards at the run
# root), so it skips the with-oracle-eval finalize/eval the probes run.
_SOURCE_EXPERIMENT = "generate_dataset/smoke-shard"


@dataclass(frozen=True)
class Scale:
    """Dataset-size knobs shared by source generation and the copy probes.

    .. attribute :: sizes

        ``train_val_test_sizes`` for the copy match set and the shuffle probe.

    .. attribute :: samples_per_shard

        Reuses per shard for the copy match set; equals the source's so filenames align.

    .. attribute :: reuse_depths

        Reuse-depth probe's swept ``samples_per_shard``; split sizes are ``max(reuse_depths)``.
    """

    sizes: tuple[int, int, int]
    samples_per_shard: int
    reuse_depths: tuple[int, ...]

    @classmethod
    def from_size(cls, size: int) -> Scale:
        """Build a cubic scale from one dataset-size int.

        :param size: Per-split sample count ``N``; yields ``sizes=(N, N, N)`` with
            one shard per split (``samples_per_shard=N``) and a no-reuse/full-reuse
            depth sweep ``(1, N)`` (deduped to ``(1,)`` at ``N == 1``).
        :returns: The scale feeding source generation and the copy probes in lockstep.
        :raises ValueError: ``size`` is below 1, so no split would hold a sample.
        """
        if size < 1:
            raise ValueError(f"dataset size must be >= 1, got {size}")
        reuse_depths = tuple(dict.fromkeys((1, size)))
        return cls(sizes=(size, size, size), samples_per_shard=size, reuse_depths=reuse_depths)


# CLI default dataset size: the full #489 run (40 samples per split). The cadence
# workflow's ``cadence_size`` input overrides it (and defaults smaller) per run.
DEFAULT_SIZE = 40

# Concurrent agents default low: each renders VST audio through one shared Xvfb
# display, so over-subscribing the box contends on CPU and the display.
DEFAULT_MAX_PARALLEL = 2


@dataclass(frozen=True)
class Experiment:
    """One investigation experiment: a fixed config plus a one- or two-knob grid.

    .. attribute :: name

        Stable identifier used for ``--only`` selection and labels.

    .. attribute :: task_name

        Distinct ``task_name`` so each experiment gets its own R2 prefix / W&B run.

    .. attribute :: fixed_overrides

        Hydra overrides shared by every cell (the experiment base + pins).

    .. attribute :: grid

        Maps a Hydra key to its swept values; the Cartesian product is the cells.

    .. attribute :: needs_copy_source

        When true, ``copy_dataset_root_uri`` is appended so the cell replays the source.
    """

    name: str
    task_name: str
    fixed_overrides: tuple[str, ...]
    grid: dict[str, list[str | int]]
    needs_copy_source: bool = False


def reference_copy_uri(prefix_root: str = DEFAULT_R2_PREFIX_ROOT) -> str:
    """Return the R2 run-root URI of the copy-source dataset.

    :param prefix_root: R2 prefix root; the default ``"data"`` yields the
        canonical run root, and a test-scoped root isolates a throwaway run.
    :returns: ``r2://<bucket>/<prefix_root>/<task>/<run>`` with no trailing slash.
    """
    prefix = make_r2_prefix(REFERENCE_TASK, REFERENCE_RUN_ID, prefix_root=prefix_root)
    return f"r2://{BUCKET}/{prefix.rstrip('/')}"


def _sizes_override(sizes: tuple[int, int, int]) -> str:
    """Build the Hydra override that pins the per-split sample counts.

    :param sizes: Sample counts in ``(train, val, test)`` order.
    :returns: A ``train_val_test_sizes=[...]`` token for the generate_dataset CLI.
    """
    return f"train_val_test_sizes=[{','.join(str(n) for n in sizes)}]"


def reference_overrides(scale: Scale, prefix_root: str = DEFAULT_R2_PREFIX_ROOT) -> list[str]:
    """Return the ``generate_dataset`` overrides that build the copy source.

    :param scale: Size knobs; ``sizes`` / ``samples_per_shard`` form the match
        set the copy probes echo.
    :param prefix_root: R2 prefix root, mirrored into the derived copy URI.
    :returns: Hydra override tokens for the source generation run.
    """
    return [
        f"experiment={_SOURCE_EXPERIMENT}",
        f"task_name={REFERENCE_TASK}",
        f"run_id={REFERENCE_RUN_ID}",
        f"r2.prefix_root={prefix_root}",
        f"render.param_spec_name={PARAM_SPEC}",
        f"render.preset_path={PRESET}",
        _sizes_override(scale.sizes),
        f"render.samples_per_shard={scale.samples_per_shard}",
        # The source only donates params for copy replay, so audio quality is
        # irrelevant; accept any non-silent render rather than fail the quiet floor.
        f"render.min_loudness={COPY_MIN_LOUDNESS}",
        f"render.max_retries={MAX_RETRIES}",
    ]


def _copy_match_overrides(scale: Scale) -> tuple[str, ...]:
    """Return the pins every copy cell shares with the source generation.

    :param scale: Size knobs supplying the shared split sizes and reuse count.
    :returns: The copy-preflight match-set overrides plus the copy loudness floor.
    """
    return (
        f"experiment={_GENERATE_EXPERIMENT}",
        f"render.param_spec_name={PARAM_SPEC}",
        f"render.preset_path={PRESET}",
        _sizes_override(scale.sizes),
        f"render.samples_per_shard={scale.samples_per_shard}",
        f"render.min_loudness={COPY_MIN_LOUDNESS}",
    )


def build_experiments(scale: Scale) -> list[Experiment]:
    """Return the five #489 experiments parameterized by ``scale``.

    :param scale: Size knobs threaded into split sizes and the reuse-depth grid.
    :returns: Ordered experiments — two within-run probes then three copy probes.
    :raises ValueError: A reuse depth does not divide the largest depth, so a reuse-depth cell
        would not split into whole shards.
    """
    # Reuse depth is the swept samples_per_shard; split sizes are the largest
    # depth, so every depth must divide it for cells to split into whole shards.
    reuse_size = max(scale.reuse_depths)
    if any(reuse_size % depth for depth in scale.reuse_depths):
        raise ValueError(
            f"reuse_depths {scale.reuse_depths} must all divide max {reuse_size} "
            "so each reuse-depth cell splits into whole shards"
        )
    return [
        Experiment(
            name="shuffle_probe",
            task_name="shuffle-probe-surge-xt",
            fixed_overrides=(
                f"experiment={_GENERATE_EXPERIMENT}",
                f"render.param_spec_name={PARAM_SPEC}",
                f"render.preset_path={PRESET}",
                "render.param_sample_cadence=shard",
                _sizes_override(scale.sizes),
                f"render.samples_per_shard={scale.samples_per_shard}",
                f"render.max_retries={MAX_RETRIES}",
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
        Experiment(
            name="reuse_depth",
            task_name="reuse-depth-surge-xt",
            fixed_overrides=(
                f"experiment={_GENERATE_EXPERIMENT}",
                f"render.param_spec_name={PARAM_SPEC}",
                f"render.preset_path={PRESET}",
                "render.param_sample_cadence=shard",
                "render.plugin_reload_cadence=once",
                "render.gui_toggle_cadence=never",
                _sizes_override((reuse_size, reuse_size, reuse_size)),
                f"render.max_retries={MAX_RETRIES}",
            ),
            grid={"render.samples_per_shard": list(scale.reuse_depths)},
        ),
        Experiment(
            name="copy_reload",
            task_name="copy-paired-reload-surge-xt",
            fixed_overrides=_copy_match_overrides(scale),
            grid={"render.plugin_reload_cadence": ["once", "render"]},
            needs_copy_source=True,
        ),
        Experiment(
            name="copy_gui",
            task_name="copy-paired-gui-surge-xt",
            fixed_overrides=_copy_match_overrides(scale),
            grid={"render.gui_toggle_cadence": ["never", "once", "render"]},
            needs_copy_source=True,
        ),
        Experiment(
            name="copy_repro",
            task_name="copy-paired-repro-surge-xt",
            fixed_overrides=_copy_match_overrides(scale),
            grid={"run_id": ["paired-repro-t1", "paired-repro-t2", "paired-repro-t3"]},
            needs_copy_source=True,
        ),
    ]


def _base_overrides(experiment: Experiment, copy_uri: str, prefix_root: str) -> list[str]:
    """Return the run-identity and pins shared by an experiment's cells and sweep.

    :param experiment: Experiment supplying the task name and fixed overrides.
    :param copy_uri: Copy-source URI appended only for copy probes.
    :param prefix_root: R2 prefix root the run writes under.
    :returns: Overrides common to every cell (the grid knobs are appended later).
    """
    overrides = [
        f"task_name={experiment.task_name}",
        f"r2.prefix_root={prefix_root}",
        *experiment.fixed_overrides,
    ]
    if experiment.needs_copy_source:
        overrides.append(f"copy_dataset_root_uri={copy_uri}")
    return overrides


def build_sweep_config(
    experiment: Experiment,
    copy_uri: str,
    prefix_root: str = DEFAULT_R2_PREFIX_ROOT,
) -> dict[str, Any]:
    """Return a wandb grid sweep config for one experiment.

    Entity/project are pinned in the config because ``wandb sweep`` ignores
    ``WANDB_ENTITY`` / ``WANDB_PROJECT`` at sweep-creation time.

    :param experiment: Experiment whose fixed overrides and grid drive the sweep.
    :param copy_uri: Copy-source URI injected into copy probes (ignored otherwise).
    :param prefix_root: R2 prefix root every cell writes under (isolates a run).
    :returns: A ``wandb.sweep``-ready config dict.
    """
    command = [
        "${interpreter}",
        "${program}",
        *_base_overrides(experiment, copy_uri, prefix_root),
        "${args_no_hyphens}",
    ]
    return {
        "program": PROGRAM,
        "entity": ENTITY,
        "project": PROJECT,
        "name": f"generate_dataset_{experiment.name}_surge_xt",
        "method": "grid",
        # grid ignores metric at scheduling time; kept for dashboard legibility.
        "metric": {"goal": "minimize", "name": "audio/mss_mean"},
        "command": command,
        "parameters": {key: {"values": values} for key, values in experiment.grid.items()},
    }


def expand_cells(
    experiment: Experiment,
    copy_uri: str,
    prefix_root: str = DEFAULT_R2_PREFIX_ROOT,
) -> list[list[str]]:
    """Expand an experiment's grid into one full override list per cell.

    :param experiment: Experiment to expand.
    :param copy_uri: Copy-source URI injected into copy probes (ignored otherwise).
    :param prefix_root: R2 prefix root every cell writes under (isolates a run).
    :returns: One Hydra override list per Cartesian-product cell.
    """
    base = _base_overrides(experiment, copy_uri, prefix_root)
    keys = list(experiment.grid)
    cells: list[list[str]] = []
    for combo in itertools.product(*(experiment.grid[key] for key in keys)):
        cells.append(
            [
                *base,
                *(f"{key}={value}" for key, value in zip(keys, combo)),
            ]
        )
    return cells


def _run_generate(overrides: list[str]) -> None:
    """Run one ``generate_dataset`` invocation as a subprocess (fail-fast).

    A non-zero exit raises ``subprocess.CalledProcessError`` (``check=True``).

    :param overrides: Hydra override tokens for the generate run.
    """
    # ``-m`` resolves the entrypoint by import, so the launcher works from any cwd.
    argv = [sys.executable, "-m", _GENERATE_MODULE, *overrides]
    # Child inherits the ambient env (R2 creds, WANDB_MODE, PYTHONPATH).
    subprocess.run(argv, check=True)  # noqa: S603 — argv built from validated literals


def generate_reference_dataset(scale: Scale, prefix_root: str, *, dry_run: bool) -> str:
    """Generate the copy-source dataset and return its run-root URI.

    :param scale: Size knobs for the source run.
    :param prefix_root: R2 prefix root for the source run and the derived URI.
    :param dry_run: When true, print the command and skip execution.
    :returns: The copy-source URI the copy probes read from.
    """
    overrides = reference_overrides(scale, prefix_root=prefix_root)
    uri = reference_copy_uri(prefix_root=prefix_root)
    if dry_run:
        logger.info(f"[dry-run] generate copy source -> {uri}")
        logger.info(f"[dry-run]   {' '.join(overrides)}")
        return uri
    logger.info(f"generating copy source -> {uri}")
    _run_generate(overrides)
    return uri


def _launch_local(
    experiment: Experiment,
    copy_uri: str,
    prefix_root: str,
    *,
    dry_run: bool,
    count: int | None,
) -> None:
    """Run an experiment's grid cells as local ``generate_dataset`` subprocesses.

    :param experiment: Experiment whose grid cells are run.
    :param copy_uri: Copy-source URI injected into copy probes.
    :param prefix_root: R2 prefix root every cell writes under.
    :param dry_run: When true, log each cell and execute nothing.
    :param count: Cell cap (``None`` runs every grid cell).
    """
    cells = expand_cells(experiment, copy_uri, prefix_root)
    for cell in cells[:count] if count is not None else cells:
        if dry_run:
            logger.info(f"[dry-run] {experiment.name} cell: {' '.join(cell)}")
            continue
        logger.info(f"{experiment.name} cell: {' '.join(cell)}")
        _run_generate(cell)


def _create_sweep(
    experiment: Experiment,
    copy_uri: str,
    prefix_root: str,
    *,
    dry_run: bool,
) -> str | None:
    """Create the wandb sweep for one experiment and return its id.

    :param experiment: Experiment whose grid drives the sweep.
    :param copy_uri: Copy-source URI injected into copy probes.
    :param prefix_root: R2 prefix root every cell writes under.
    :param dry_run: When true, log the sweep config and create nothing.
    :returns: The created sweep id, or ``None`` under ``dry_run``.
    """
    config = build_sweep_config(experiment, copy_uri, prefix_root)
    if dry_run:
        logger.info(f"[dry-run] wandb sweep {experiment.name}: {config}")
        return None
    sweep_id = wandb.sweep(config, entity=ENTITY, project=PROJECT)
    logger.info(f"created sweep {experiment.name} -> {ENTITY}/{PROJECT}/{sweep_id}")
    return sweep_id


def _run_agent(sweep_id: str, *, count: int | None) -> None:
    """Run a ``wandb agent`` for one sweep to completion as a subprocess (fail-fast).

    Running each agent as its own process lets the supervisor cap concurrency and
    keeps one stalled agent from blocking the others — unlike the in-process
    ``wandb.agent`` loop, which served sweeps strictly one at a time. A non-zero
    exit raises ``subprocess.CalledProcessError`` (``check=True``).

    :param sweep_id: Sweep the agent pulls grid cells from.
    :param count: Agent run cap forwarded to ``--count`` (``None`` runs every cell).
    """
    argv = ["wandb", "agent"]
    if count is not None:
        argv += ["--count", str(count)]
    argv.append(f"{ENTITY}/{PROJECT}/{sweep_id}")
    # wandb 0.26.1's Agent.is_flapping reads wandb.START_TIME, which only the legacy
    # wandb.old.core sets, so the agent crashes with AttributeError unless flapping
    # is disabled; the grid is already bounded by count, so flapping adds no value.
    env = {**os.environ, wandb.env.AGENT_DISABLE_FLAPPING: "true"}
    subprocess.run(argv, check=True, env=env)  # noqa: S603 — argv built from validated literals


def _supervise_agents(
    sweep_ids: list[str],
    *,
    count: int | None,
    max_parallel: int,
    run_agent: Callable[..., None] | None = None,
) -> list[str]:
    """Run an agent per sweep concurrently, capping concurrency at ``max_parallel``.

    A stalled agent occupies one pool slot only, so the remaining sweeps keep
    draining; failures are collected across all agents and raised once every
    sweep has been attempted.

    :param sweep_ids: Sweeps to run agents for.
    :param count: Per-agent run cap forwarded to each agent.
    :param max_parallel: Maximum agents running at once.
    :param run_agent: Per-agent runner; defaults to :func:`_run_agent` and is
        injectable so tests avoid real subprocesses.
    :returns: The sweep ids whose agents exited cleanly.
    :raises RuntimeError: when one or more agents fail, naming the failed sweeps.
    """
    run_agent = run_agent or _run_agent
    completed: list[str] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_agent, sid, count=count): sid for sid in sweep_ids}
        for future in as_completed(futures):
            sweep_id = futures[future]
            try:
                future.result()
                completed.append(sweep_id)
            except Exception as exc:  # noqa: BLE001 — record every failure, raise once below
                failures[sweep_id] = f"{type(exc).__name__}: {exc}"
                logger.warning(f"agent for sweep {sweep_id} failed: {failures[sweep_id]}")
    if failures:
        raise RuntimeError(f"{len(failures)} sweep agent(s) failed: {failures}")
    return completed


def run_investigation(
    *,
    scale: Scale,
    launcher: str,
    prefix_root: str,
    only: list[str] | None,
    dry_run: bool,
    count: int | None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
) -> None:
    """Generate the copy source, then dispatch each selected experiment.

    On the wandb launcher every selected sweep is created up front, then agents
    run concurrently under a ``max_parallel`` cap — so all sweeps appear in the
    UI regardless of agent fate, and no stalled agent blocks the rest.

    :param scale: Size knobs shared by the source and copy probes.
    :param launcher: ``"wandb"`` (sweeps + supervised agents) or ``"local"`` (subprocess loop).
    :param prefix_root: R2 prefix root for the source run and copy URI.
    :param only: Experiment names to run; ``None`` runs all five.
    :param dry_run: When true, print the plan and execute nothing.
    :param count: Per-experiment cell cap (``None`` runs every grid cell).
    :param max_parallel: Maximum wandb agents running at once (ignored by ``local``).
    :raises ValueError: ``launcher`` is not ``"wandb"``/``"local"``, ``only`` names
        an unknown experiment, ``count`` is set below 1, or ``max_parallel`` is below 1
        (caps that would silently run nothing).
    """
    if launcher not in ("wandb", "local"):
        raise ValueError(f"launcher must be 'wandb' or 'local', got {launcher!r}")
    if count is not None and count < 1:
        raise ValueError(f"count must be >= 1 when set, got {count}")
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    experiments = build_experiments(scale)
    by_name = {e.name: e for e in experiments}
    if only is not None:
        unknown = [name for name in only if name not in by_name]
        if unknown:
            raise ValueError(f"unknown experiment(s): {unknown}; choose from {list(by_name)}")
        experiments = [by_name[name] for name in only]

    # The copy source is only read by copy probes; skip its (real, costly)
    # generation when the selection is within-run probes alone. It must finish
    # before any copy-probe agent runs, so it stays sequential ahead of dispatch.
    if any(experiment.needs_copy_source for experiment in experiments):
        copy_uri = generate_reference_dataset(scale, prefix_root, dry_run=dry_run)
    else:
        copy_uri = reference_copy_uri(prefix_root=prefix_root)

    if launcher == "local":
        for experiment in experiments:
            _launch_local(experiment, copy_uri, prefix_root, dry_run=dry_run, count=count)
        return

    sweep_ids: list[str] = []
    for experiment in experiments:
        sweep_id = _create_sweep(experiment, copy_uri, prefix_root, dry_run=dry_run)
        if sweep_id is not None:
            sweep_ids.append(sweep_id)
    if sweep_ids:
        _supervise_agents(sweep_ids, count=count, max_parallel=max_parallel)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the orchestrator CLI arguments.

    :param argv: Argument vector; ``None`` falls back to ``sys.argv[1:]``.
    :returns: Flags for ``run_investigation`` — launcher, size, only, prefix_root, count, dry_run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launcher",
        choices=("wandb", "local"),
        default="wandb",
        help="wandb sweeps + agents (default) or a single-box subprocess loop",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"dataset size N -> [N,N,N] splits (default: {DEFAULT_SIZE}, the full #489 run)",
    )
    parser.add_argument(
        "--only",
        help="comma-separated experiment names to run (default: all five)",
    )
    parser.add_argument(
        "--prefix-root",
        default=DEFAULT_R2_PREFIX_ROOT,
        help="R2 prefix root for the source run and derived copy URI",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="per-experiment cell cap (wandb agent run cap; default: all cells)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help=(
            f"max wandb agents running at once (default: {DEFAULT_MAX_PARALLEL}); "
            "agents share one Xvfb display, so raise with care"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and execute nothing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: run (or plan) the #489 cadence investigation.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    run_investigation(
        scale=Scale.from_size(args.size),
        launcher=args.launcher,
        prefix_root=args.prefix_root,
        only=[name for name in (n.strip() for n in args.only.split(",")) if name]
        if args.only
        else None,
        dry_run=args.dry_run,
        count=args.count,
        max_parallel=args.max_parallel,
    )


if __name__ == "__main__":
    main()
