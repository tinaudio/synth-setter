"""One-command runner for the #489 cadence investigation.

Generates the fixed surge_xt and surge_simple copy-source datasets, then creates
the W&B grid sweeps and runs an agent for each. Copy and shuffle probes replay
the source via its derived run-root URI; the controls omit the copy URI and
regenerate fresh. ``sweeps()`` is authoritative on the matrix; see #489.

The dataset size N is the only input; it feeds both the source and the copy
probes so their copy-preflight match set (param spec, samples-per-shard, split
sizes) cannot drift between producer and consumer.

Run it::

    python -m synth_setter.tools.cadence_sweep_489            # the full #489 run
    python -m synth_setter.tools.cadence_sweep_489 --size 5

Every sweep is created before any agent runs, so they all appear in the W&B UI
even if an agent later stalls; agents then run one at a time because they share
one Xvfb display and would otherwise contend on it. Run under tmux/nohup so a
disconnect does not orphan an in-flight agent.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any

import wandb
import wandb.env
from loguru import logger

from synth_setter.data.vst import preset_paths
from synth_setter.pipeline.schemas.prefix import DEFAULT_R2_PREFIX_ROOT, make_r2_prefix

# Temporary personal pin: the ``tinaudio`` W&B entity does not exist yet (sweep
# creation 404s), so sweeps target this working entity until #1560 stands it up.
ENTITY = "khaledtinubu-n-a"
PROJECT = "synth-setter"
BUCKET = "intermediate-data"
# R2 prefix root the whole run writes under; tests monkeypatch it to isolate a throwaway run.
PREFIX_ROOT = DEFAULT_R2_PREFIX_ROOT

# wandb's ``program:`` field is repo-relative; the source generation imports by module path.
PROGRAM = "src/synth_setter/cli/generate_dataset.py"
_GENERATE_MODULE = "synth_setter.cli.generate_dataset"

SURGE_XT = "surge_xt"
SURGE_SIMPLE = "surge_simple"
SURGE_XT_PRESET = preset_paths[SURGE_XT]
SURGE_SIMPLE_PRESET = preset_paths[SURGE_SIMPLE]

# Fixed reference run identity -> stable copy-source URI the copy probes replay.
SURGE_XT_REFERENCE_TASK = "ref-surge-xt-489"
SURGE_XT_REFERENCE_RUN_ID = "paired-surge-xt-ref-v1"
SURGE_SIMPLE_REFERENCE_TASK = "ref-surge-simple-489"
SURGE_SIMPLE_REFERENCE_RUN_ID = "paired-surge-simple-ref-v1"

# The probes run the with-oracle-eval finalize/eval; the source donates raw param
# shards only (copy reads same-named shards at the run root), so it skips the eval.
_PROBE_EXPERIMENT = "generate_dataset/smoke-shard-with-oracle-eval"
_SOURCE_EXPERIMENT = "generate_dataset/smoke-shard"

# CLI default dataset size: the full #489 run.
DEFAULT_SIZE = 40


def surge_xt_reference_copy_uri() -> str:
    """Return the R2 run-root URI of the copy-source dataset under the current ``PREFIX_ROOT``.

    :returns: ``r2://<bucket>/<prefix_root>/<task>/<run>`` with no trailing slash.
    """
    prefix = make_r2_prefix(
        SURGE_XT_REFERENCE_TASK, SURGE_XT_REFERENCE_RUN_ID, prefix_root=PREFIX_ROOT
    )
    return f"r2://{BUCKET}/{prefix.rstrip('/')}"


def surge_simple_reference_copy_uri() -> str:
    """Return the R2 run-root URI of the copy-source dataset under the current ``PREFIX_ROOT``.

    :returns: ``r2://<bucket>/<prefix_root>/<task>/<run>`` with no trailing slash.
    """
    prefix = make_r2_prefix(
        SURGE_SIMPLE_REFERENCE_TASK, SURGE_SIMPLE_REFERENCE_RUN_ID, prefix_root=PREFIX_ROOT
    )
    return f"r2://{BUCKET}/{prefix.rstrip('/')}"


def _sweep(
    name: str,
    *,
    fixed: tuple[str, ...],
    grid: dict[str, list[Any]],
) -> dict[str, Any]:
    """Assemble one ``wandb.sweep``-ready grid config from its fixed pins and swept grid.

    Entity/project are pinned in the config because ``wandb sweep`` ignores
    ``WANDB_ENTITY`` / ``WANDB_PROJECT`` at sweep-creation time.

    :param name: Stable experiment label, embedded in the sweep name.
    :param fixed: Hydra overrides shared by every grid cell.
    :param grid: Maps each swept Hydra key to its values; their product is the cells.
    :returns: A ``wandb.sweep``-ready config dict.
    """
    # ``${args_no_hyphens}`` (the swept grid cell) goes last: Hydra applies later overrides last,
    # so a fixed pin must never follow a swept key it would otherwise shadow.
    command = [
        "${interpreter}",
        "${program}",
        f"task_name={name}",
        f"r2.prefix_root={PREFIX_ROOT}",
        f"experiment={_PROBE_EXPERIMENT}",
        *fixed,
        "${args_no_hyphens}",
    ]
    return {
        "program": PROGRAM,
        "entity": ENTITY,
        "project": PROJECT,
        "name": f"generate_dataset_{name}",
        "method": "grid",
        # grid ignores metric at scheduling time; kept for dashboard legibility.
        "metric": {"goal": "minimize", "name": "audio/mss_mean"},
        "command": command,
        "parameters": {key: {"values": values} for key, values in grid.items()},
    }


def sweeps(n: int) -> list[dict[str, Any]]:
    """Return the #489 W&B grid sweep configs at dataset size ``n``.

    The copy and shuffle probes pin the derived ``copy_dataset_root_uri`` so their cells replay the
    source verbatim; the control sweeps omit it and regenerate fresh.

    :param n: Per-split sample count shared with the source generation.
    :returns: ``wandb.sweep``-ready config dicts, in run order.
    :raises ValueError: ``n`` is below 1, so no split would hold a sample.
    """
    if n < 1:
        raise ValueError(f"dataset size must be >= 1, got {n}")
    splits = f"train_val_test_sizes=[{n},{n},{n}]"
    samples_per_shard = f"render.samples_per_shard={n}"
    xt_spec = f"render.param_spec_name={SURGE_XT}"
    xt_preset = f"render.preset_path={SURGE_XT_PRESET}"
    simple_spec = f"render.param_spec_name={SURGE_SIMPLE}"
    simple_preset = f"render.preset_path={SURGE_SIMPLE_PRESET}"
    xt_copy_uri = f"copy_dataset_root_uri={surge_xt_reference_copy_uri()}"
    simple_copy_uri = f"copy_dataset_root_uri={surge_simple_reference_copy_uri()}"

    return [
        _sweep(
            "shuffle_cadence_probe_surge_xt",
            fixed=(
                "render.param_sample_cadence=shard",
                splits,
                samples_per_shard,
                xt_spec,
                xt_preset,
                xt_copy_uri,
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
        _sweep(
            "cadence_probe_surge_xt",
            fixed=(
                splits,
                samples_per_shard,
                xt_spec,
                xt_preset,
                xt_copy_uri,
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
        _sweep(
            "control_cadence_probe_surge_xt",
            fixed=(
                splits,
                samples_per_shard,
                xt_spec,
                xt_preset,
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
        _sweep(
            "shuffle_cadence_probe_surge_simple",
            fixed=(
                "render.param_sample_cadence=shard",
                splits,
                samples_per_shard,
                simple_spec,
                simple_preset,
                simple_copy_uri,
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
        _sweep(
            "cadence_probe_surge_simple",
            fixed=(
                splits,
                samples_per_shard,
                simple_spec,
                simple_preset,
                simple_copy_uri,
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
        _sweep(
            "control_cadence_probe_surge_simple",
            fixed=(
                splits,
                samples_per_shard,
                simple_spec,
                simple_preset,
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
        _sweep(
            "cadence_probe_surge_simple_xt_preset",
            fixed=(
                splits,
                samples_per_shard,
                simple_spec,
                xt_preset,
                simple_copy_uri,
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
    ]


def _run_generate(overrides: list[str]) -> None:
    """Run one ``generate_dataset`` invocation as a subprocess (fail-fast).

    A non-zero exit raises ``subprocess.CalledProcessError`` (``check=True``).

    :param overrides: Hydra override tokens for the generate run.
    """
    # ``-m`` resolves the entrypoint by import, so the launcher works from any cwd;
    # the child inherits the ambient env (R2 creds, WANDB_MODE, PYTHONPATH).
    argv = [sys.executable, "-m", _GENERATE_MODULE, *overrides]
    subprocess.run(argv, check=True)  # noqa: S603 — argv built from validated literals


def _run_agent(sweep_id: str) -> None:
    """Run a ``wandb agent`` for one sweep to completion as a subprocess (fail-fast).

    A non-zero exit raises ``subprocess.CalledProcessError`` (``check=True``).

    :param sweep_id: Sweep the agent pulls grid cells from.
    """
    # wandb 0.26.1's Agent.is_flapping reads wandb.START_TIME, which only the legacy
    # wandb.old.core sets, so the agent crashes with AttributeError unless flapping is
    # disabled; the grid is bounded, so flapping adds no value anyway.
    env = {**os.environ, wandb.env.AGENT_DISABLE_FLAPPING: "true"}
    argv = ["wandb", "agent", f"{ENTITY}/{PROJECT}/{sweep_id}"]
    subprocess.run(argv, check=True, env=env)  # noqa: S603 — fixed argv plus our sweep id


def run(n: int) -> None:
    """Generate the surge_xt and surge_simple copy sources, then create and drive every #489 sweep.

    The sources are generated first because the copy probes read them; then every sweep is created
    before any agent runs so they all land in the W&B UI; agents run one at a time to avoid
    contending on the shared Xvfb display.

    :param n: Per-split sample count shared by the sources and the copy probes.
    """
    # Build (and validate ``n`` via) the sweep configs first, so an invalid size fails fast
    # before the two expensive source-generation subprocesses run.
    sweep_configs = sweeps(n)
    copy_src_overrides = [
        f"experiment={_SOURCE_EXPERIMENT}",
        f"r2.prefix_root={PREFIX_ROOT}",
        f"train_val_test_sizes=[{n},{n},{n}]",
        f"render.samples_per_shard={n}",
    ]
    xt_overrides = copy_src_overrides + [
        f"task_name={SURGE_XT_REFERENCE_TASK}",
        f"run_id={SURGE_XT_REFERENCE_RUN_ID}",
        f"render.param_spec_name={SURGE_XT}",
        f"render.preset_path={SURGE_XT_PRESET}",
    ]
    simple_overrides = copy_src_overrides + [
        f"task_name={SURGE_SIMPLE_REFERENCE_TASK}",
        f"run_id={SURGE_SIMPLE_REFERENCE_RUN_ID}",
        f"render.param_spec_name={SURGE_SIMPLE}",
        f"render.preset_path={SURGE_SIMPLE_PRESET}",
    ]
    logger.info(f"generating copy source -> {surge_xt_reference_copy_uri()}")
    _run_generate(xt_overrides)
    logger.info(f"generating copy source -> {surge_simple_reference_copy_uri()}")
    _run_generate(simple_overrides)
    sweep_ids = [wandb.sweep(config, entity=ENTITY, project=PROJECT) for config in sweep_configs]
    for sweep_id in sweep_ids:
        logger.info(f"running agent for {ENTITY}/{PROJECT}/{sweep_id}")
        _run_agent(sweep_id)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: run the #489 cadence investigation at the chosen dataset size.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"dataset size N -> [N,N,N] splits (default: {DEFAULT_SIZE}, the full #489 run)",
    )
    args = parser.parse_args(argv)
    run(args.size)


if __name__ == "__main__":
    main()
