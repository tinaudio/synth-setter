"""One-command runner for the #489 surge_xt cadence investigation.

Generates the fixed surge_xt copy-source dataset, then creates five W&B grid
sweeps and runs an agent for each: two within-run probes (render-order shuffle,
reuse-depth) and three paired-copy probes (reload cadence, gui cadence,
reproducibility) that replay the source via its derived run-root URI.

The dataset size N is the only input; it feeds both the source and the copy
probes so their copy-preflight match set (param spec, samples-per-shard, split
sizes) cannot drift between producer and consumer.

Run it::

    python -m synth_setter.tools.cadence_sweep_489            # the full #489 run
    python -m synth_setter.tools.cadence_sweep_489 --size 5

Every sweep is created before any agent runs, so all five appear in the W&B UI
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

PARAM_SPEC = "surge_xt"
PRESET = "presets/surge-base.vstpreset"

# Fixed reference run identity -> stable copy-source URI the copy probes replay.
REFERENCE_TASK = "ref-surge-xt-489"
REFERENCE_RUN_ID = "paired-ref-v1"

# Copy can't resample a sub-floor re-render (#724); -1000 dB lets junk/quiet
# renders reach the oracle while only true silence (-inf) trips the floor.
COPY_MIN_LOUDNESS = -1000.0
# Fresh-param probes occasionally sample a silent (-inf) patch; retries resample past it.
MAX_RETRIES = 10

# The probes run the with-oracle-eval finalize/eval; the source donates raw param
# shards only (copy reads same-named shards at the run root), so it skips the eval.
_PROBE_EXPERIMENT = "generate_dataset/smoke-shard-with-oracle-eval"
_SOURCE_EXPERIMENT = "generate_dataset/smoke-shard"

# CLI default dataset size: the full #489 run.
DEFAULT_SIZE = 40


def reference_copy_uri() -> str:
    """Return the R2 run-root URI of the copy-source dataset under the current ``PREFIX_ROOT``.

    :returns: ``r2://<bucket>/<prefix_root>/<task>/<run>`` with no trailing slash.
    """
    prefix = make_r2_prefix(REFERENCE_TASK, REFERENCE_RUN_ID, prefix_root=PREFIX_ROOT)
    return f"r2://{BUCKET}/{prefix.rstrip('/')}"


def source_overrides(n: int) -> list[str]:
    """Return the ``generate_dataset`` overrides that build the copy source at size ``n``.

    :param n: Per-split sample count; sets ``[n,n,n]`` splits at one shard each.
    :returns: Hydra override tokens for the source generation run.
    :raises ValueError: ``n`` is below 1, so no split would hold a sample.
    """
    if n < 1:
        raise ValueError(f"dataset size must be >= 1, got {n}")
    return [
        f"experiment={_SOURCE_EXPERIMENT}",
        f"task_name={REFERENCE_TASK}",
        f"run_id={REFERENCE_RUN_ID}",
        f"r2.prefix_root={PREFIX_ROOT}",
        f"render.param_spec_name={PARAM_SPEC}",
        f"render.preset_path={PRESET}",
        f"train_val_test_sizes=[{n},{n},{n}]",
        f"render.samples_per_shard={n}",
        # The source only donates params for replay, so accept any non-silent render.
        f"render.min_loudness={COPY_MIN_LOUDNESS}",
        f"render.max_retries={MAX_RETRIES}",
    ]


def _sweep(
    name: str,
    task_name: str,
    *,
    fixed: tuple[str, ...],
    grid: dict[str, list[Any]],
    copy_uri: str | None = None,
) -> dict[str, Any]:
    """Assemble one ``wandb.sweep``-ready grid config from its fixed pins and swept grid.

    Entity/project are pinned in the config because ``wandb sweep`` ignores
    ``WANDB_ENTITY`` / ``WANDB_PROJECT`` at sweep-creation time.

    :param name: Stable experiment label, embedded in the sweep name.
    :param task_name: Distinct ``task_name`` giving the run its own R2 prefix / W&B run.
    :param fixed: Hydra overrides shared by every grid cell.
    :param grid: Maps each swept Hydra key to its values; their product is the cells.
    :param copy_uri: When set, appended as ``copy_dataset_root_uri`` so cells replay the source.
    :returns: A ``wandb.sweep``-ready config dict.
    """
    command = [
        "${interpreter}",
        "${program}",
        f"task_name={task_name}",
        f"r2.prefix_root={PREFIX_ROOT}",
        *fixed,
    ]
    if copy_uri is not None:
        command.append(f"copy_dataset_root_uri={copy_uri}")
    command.append("${args_no_hyphens}")
    return {
        "program": PROGRAM,
        "entity": ENTITY,
        "project": PROJECT,
        "name": f"generate_dataset_{name}_surge_xt",
        "method": "grid",
        # grid ignores metric at scheduling time; kept for dashboard legibility.
        "metric": {"goal": "minimize", "name": "audio/mss_mean"},
        "command": command,
        "parameters": {key: {"values": values} for key, values in grid.items()},
    }


def sweeps(n: int) -> list[dict[str, Any]]:
    """Return the five #489 W&B grid sweep configs at dataset size ``n``.

    Two within-run probes then three paired-copy probes; the copy probes carry the
    derived ``copy_dataset_root_uri`` so every cell replays the source verbatim. The
    copy match set (param spec / preset / split sizes / samples-per-shard) is shared
    with :func:`source_overrides` so shard filenames and the param encoding align.

    :param n: Per-split sample count shared with the source generation.
    :returns: ``wandb.sweep``-ready config dicts, in run order.
    :raises ValueError: ``n`` is below 1, so no split would hold a sample.
    """
    if n < 1:
        raise ValueError(f"dataset size must be >= 1, got {n}")
    splits = f"train_val_test_sizes=[{n},{n},{n}]"
    copy_match = (
        f"experiment={_PROBE_EXPERIMENT}",
        f"render.param_spec_name={PARAM_SPEC}",
        f"render.preset_path={PRESET}",
        splits,
        f"render.samples_per_shard={n}",
        f"render.min_loudness={COPY_MIN_LOUDNESS}",
    )
    copy_uri = reference_copy_uri()
    return [
        _sweep(
            "shuffle_probe",
            "shuffle-probe-surge-xt",
            fixed=(
                f"experiment={_PROBE_EXPERIMENT}",
                f"render.param_spec_name={PARAM_SPEC}",
                f"render.preset_path={PRESET}",
                "render.param_sample_cadence=shard",
                splits,
                f"render.samples_per_shard={n}",
                f"render.max_retries={MAX_RETRIES}",
            ),
            grid={
                "render.plugin_reload_cadence": ["once", "render"],
                "render.gui_toggle_cadence": ["never", "once", "render", "always_on"],
            },
        ),
        _sweep(
            "reuse_depth",
            "reuse-depth-surge-xt",
            fixed=(
                f"experiment={_PROBE_EXPERIMENT}",
                f"render.param_spec_name={PARAM_SPEC}",
                f"render.preset_path={PRESET}",
                "render.param_sample_cadence=shard",
                "render.plugin_reload_cadence=once",
                "render.gui_toggle_cadence=never",
                splits,
                f"render.max_retries={MAX_RETRIES}",
            ),
            # Reuse depth is the swept samples_per_shard: 1 (no reuse) and n (full
            # reuse), which collapse to a single cell at n == 1. Both divide n, so
            # every cell splits the [n,n,n] sizes into whole shards.
            grid={"render.samples_per_shard": sorted({1, n})},
        ),
        _sweep(
            "copy_reload",
            "copy-paired-reload-surge-xt",
            fixed=copy_match,
            grid={"render.plugin_reload_cadence": ["once", "render"]},
            copy_uri=copy_uri,
        ),
        _sweep(
            "copy_gui",
            "copy-paired-gui-surge-xt",
            fixed=copy_match,
            grid={"render.gui_toggle_cadence": ["never", "once", "render"]},
            copy_uri=copy_uri,
        ),
        _sweep(
            "copy_repro",
            "copy-paired-repro-surge-xt",
            fixed=copy_match,
            grid={"run_id": ["paired-repro-t1", "paired-repro-t2", "paired-repro-t3"]},
            copy_uri=copy_uri,
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
    """Generate the copy source, then create and drive all five #489 sweeps.

    The source is generated first because the copy probes read it; then every sweep
    is created before any agent runs so all five land in the W&B UI; agents run one
    at a time to avoid contending on the shared Xvfb display.

    :param n: Per-split sample count shared by the source and the copy probes;
        :func:`source_overrides` rejects a value below 1.
    """
    logger.info(f"generating copy source -> {reference_copy_uri()}")
    _run_generate(source_overrides(n))
    sweep_ids = [wandb.sweep(config, entity=ENTITY, project=PROJECT) for config in sweeps(n)]
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
