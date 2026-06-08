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

Run the whole investigation::

    python -m synth_setter.tools.cadence_investigation_489               # wandb sweeps + agents
    python -m synth_setter.tools.cadence_investigation_489 --dry-run     # print the plan only
    python -m synth_setter.tools.cadence_investigation_489 --launcher local   # single-box subprocess loop
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import wandb
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


# Full #489 run: reuse depths straddle the junk onset (~12 reuses, #706).
FULL = Scale(sizes=(40, 40, 40), samples_per_shard=20, reuse_depths=(4, 20, 40, 80))
# Smallest run that still exercises generate -> copy -> oracle for tests.
SMOKE = Scale(sizes=(2, 2, 2), samples_per_shard=2, reuse_depths=(1, 2))


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


def _launch_wandb(
    experiment: Experiment,
    copy_uri: str,
    prefix_root: str,
    *,
    dry_run: bool,
    count: int | None,
) -> None:
    """Create one wandb sweep for the experiment and run an agent over it.

    :param experiment: Experiment whose grid drives the sweep.
    :param copy_uri: Copy-source URI injected into copy probes.
    :param prefix_root: R2 prefix root every cell writes under.
    :param dry_run: When true, log the sweep config and create nothing.
    :param count: Agent run cap (``None`` runs every grid cell).
    """
    config = build_sweep_config(experiment, copy_uri, prefix_root)
    if dry_run:
        logger.info(f"[dry-run] wandb sweep {experiment.name}: {config}")
        return
    sweep_id = wandb.sweep(config, entity=ENTITY, project=PROJECT)
    logger.info(f"created sweep {experiment.name} -> {ENTITY}/{PROJECT}/{sweep_id}")
    wandb.agent(sweep_id, entity=ENTITY, project=PROJECT, count=count)


def run_investigation(
    *,
    scale: Scale,
    launcher: str,
    prefix_root: str,
    only: list[str] | None,
    dry_run: bool,
    count: int | None,
) -> None:
    """Generate the copy source, then dispatch each selected experiment.

    :param scale: Size knobs shared by the source and copy probes.
    :param launcher: ``"wandb"`` (sweeps + agents) or ``"local"`` (subprocess loop).
    :param prefix_root: R2 prefix root for the source run and copy URI.
    :param only: Experiment names to run; ``None`` runs all five.
    :param dry_run: When true, print the plan and execute nothing.
    :param count: Per-experiment cell cap (``None`` runs every grid cell).
    :raises ValueError: ``only`` names an unknown experiment, or ``count`` is set
        below 1 (a cap that would silently run no cells).
    """
    if count is not None and count < 1:
        raise ValueError(f"count must be >= 1 when set, got {count}")
    experiments = build_experiments(scale)
    by_name = {e.name: e for e in experiments}
    if only is not None:
        unknown = [name for name in only if name not in by_name]
        if unknown:
            raise ValueError(f"unknown experiment(s): {unknown}; choose from {list(by_name)}")
        experiments = [by_name[name] for name in only]

    # The copy source is only read by copy probes; skip its (real, costly)
    # generation when the selection is within-run probes alone.
    if any(experiment.needs_copy_source for experiment in experiments):
        copy_uri = generate_reference_dataset(scale, prefix_root, dry_run=dry_run)
    else:
        copy_uri = reference_copy_uri(prefix_root=prefix_root)
    for experiment in experiments:
        if launcher == "local":
            _launch_local(experiment, copy_uri, prefix_root, dry_run=dry_run, count=count)
        else:
            _launch_wandb(experiment, copy_uri, prefix_root, dry_run=dry_run, count=count)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the orchestrator CLI arguments.

    :param argv: Argument vector; ``None`` falls back to ``sys.argv[1:]``.
    :returns: Flags for ``run_investigation`` — launcher, scale, only, prefix_root, count, dry_run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launcher",
        choices=("wandb", "local"),
        default="wandb",
        help="wandb sweeps + agents (default) or a single-box subprocess loop",
    )
    parser.add_argument(
        "--scale",
        choices=("full", "smoke"),
        default="full",
        help="full #489 sizes (default) or a tiny end-to-end smoke run",
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
        scale=SMOKE if args.scale == "smoke" else FULL,
        launcher=args.launcher,
        prefix_root=args.prefix_root,
        only=[name.strip() for name in args.only.split(",")] if args.only else None,
        dry_run=args.dry_run,
        count=args.count,
    )


if __name__ == "__main__":
    main()
