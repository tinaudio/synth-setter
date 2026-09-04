#!/usr/bin/env python
"""Repair the dataset lineage two finished 440k training runs never recorded.

One-off ops backfill for the third scope item of
https://github.com/tinaudio/synth-setter/issues/2424. Two independent repairs:

1. The dataset artifact was finalized before the ``aliases=[spec.run_id]`` write
   landed in #1881, so the immutable ``:{run_id}`` alias train and eval resolve
   against does not exist and every lookup 404s.
2. Neither 100k run recorded a ``use_artifact`` edge, so both model artifacts
   hang off their run with no upstream dataset node.

Edges attach through the public ``Api`` rather than by resuming each run, so a
finished run stays finished and a crashed one stays crashed. Both repairs are
idempotent — re-running after a partial failure repeats only what is missing.

Entity and project come from ``WANDB_ENTITY`` / ``WANDB_PROJECT``. Reports
without mutating anything unless ``--apply`` is given::

    uv run python scripts/dev/backfill_lineage_2424.py
    uv run python scripts/dev/backfill_lineage_2424.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

# The pre-#1881 dataset whose `:{run_id}` alias is missing, and the two runs
# that consumed it without recording an edge (#2424).
DATASET_ARTIFACT_NAME = "data-surge-simple-lance-440k-20k-20k"
DATASET_ARTIFACT_VERSION = "v0"
CONSUMING_RUN_IDS = (
    "flow_simple_440k_m2l_100k-20260724T090607676Z",
    "flow_simple_440k_clap_100k-20260724T160509533Z",
)


def plan_alias(aliases: Sequence[str], producing_run_id: str | None) -> str | None:
    """Return the immutable producing-run alias the artifact is still missing.

    :param aliases: Aliases the artifact version already carries.
    :param producing_run_id: The run that logged the artifact, or ``None`` when
        W&B no longer resolves one.
    :returns: The alias to add, or ``None`` when it is already present or
        cannot be derived.
    """
    if producing_run_id is None or producing_run_id in aliases:
        return None
    return producing_run_id


def plan_edges(
    consumed_artifact_names: Mapping[str, Sequence[str]], dataset_artifact_name: str
) -> tuple[str, ...]:
    """Return the runs that still lack a ``use_artifact`` edge to the dataset.

    Matching ignores the version suffix, so a run already consuming any version
    of the dataset is left alone rather than given a second edge.

    :param consumed_artifact_names: ``run_id -> ["name:version", …]`` as reported
        by each run's ``used_artifacts()``.
    :param dataset_artifact_name: Unversioned dataset artifact name to match on.
    :returns: The run ids needing an edge, in input order.
    """
    return tuple(
        run_id
        for run_id, consumed in consumed_artifact_names.items()
        if not any(name.split(":", 1)[0] == dataset_artifact_name for name in consumed)
    )


def _qualified(entity: str, project: str, path: str) -> str:
    """Return a ``entity/project/path`` reference for the W&B public API.

    :param entity: W&B entity.
    :param project: W&B project.
    :param path: Run id or ``artifact:version``.
    :returns: The fully-qualified reference.
    """
    return f"{entity}/{project}/{path}"


def _emit(message: str) -> None:
    """Write one line of the backfill report to stdout.

    :param message: Report line.
    """
    print(message)  # noqa: T201 — CLI tool: stdout is its product, not a debug print


def _report(plan_alias_value: str | None, run_ids: tuple[str, ...], applied: bool) -> None:
    """Print what was (or would be) repaired.

    :param plan_alias_value: Alias to add, or ``None`` when none is needed.
    :param run_ids: Runs needing a dataset edge.
    :param applied: Whether the mutations were performed.
    """
    verb = "added" if applied else "would add"
    if plan_alias_value is None:
        _emit(f"alias: nothing to do on {DATASET_ARTIFACT_NAME}:{DATASET_ARTIFACT_VERSION}")
    else:
        _emit(f"alias: {verb} {plan_alias_value!r} to {DATASET_ARTIFACT_NAME}")
    if not run_ids:
        _emit("edges: nothing to do; every run already consumes the dataset")
        return
    for run_id in run_ids:
        _emit(f"edges: {verb} {DATASET_ARTIFACT_NAME} -> {run_id}")


def main(argv: Sequence[str] | None = None) -> int:
    """Plan the backfill against live W&B and, with ``--apply``, perform it.

    :param argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="perform the repairs instead of reporting them"
    )
    args = parser.parse_args(argv)

    entity, project = os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        print(  # noqa: T201 — CLI tool: stderr is its product, not a debug print
            "WANDB_ENTITY and WANDB_PROJECT must be set", file=sys.stderr
        )
        return 2

    import wandb

    api = wandb.Api()
    artifact_path = f"{DATASET_ARTIFACT_NAME}:{DATASET_ARTIFACT_VERSION}"
    artifact = api.artifact(_qualified(entity, project, artifact_path))
    producer = artifact.logged_by()

    runs: dict[str, Any] = {
        run_id: api.run(_qualified(entity, project, run_id)) for run_id in CONSUMING_RUN_IDS
    }
    alias = plan_alias(artifact.aliases, producer.id if producer else None)
    run_ids = plan_edges(
        {run_id: [used.name for used in run.used_artifacts()] for run_id, run in runs.items()},
        DATASET_ARTIFACT_NAME,
    )

    if args.apply:
        if alias is not None:
            artifact.aliases.append(alias)
            artifact.save()
        for run_id in run_ids:
            runs[run_id].use_artifact(artifact)
    _report(alias, run_ids, applied=args.apply)
    if not args.apply and (alias is not None or run_ids):
        _emit("\ndry run; re-run with --apply to perform the repairs above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
