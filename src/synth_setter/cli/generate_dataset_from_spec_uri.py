"""Spec-first generate_dataset runner.

``synth-setter-generate-dataset-from-spec-uri`` → :func:`main` — operator
entry that renders an **already-materialized** ``input_spec.json`` referenced
by URI (bare local path, ``file://``, ``r2://``, or ``s3://``). The spec-first
counterpart to ``synth-setter-generate-dataset``: no Hydra compose, no spec
upload, no SkyPilot dispatch — the spec at the URI is the frozen source of
truth (typically written earlier by that launcher via ``spec_io.upload_spec``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from synth_setter.cli.generate_dataset import generate
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.spec_io import load_spec_from_uri


def run_from_spec_uri(spec_uri: str) -> None:  # noqa: DOC502 — raises propagate from callees
    """Load a materialized ``DatasetSpec`` from ``spec_uri`` and render it in-process.

    R2 credentials are validated up front because ``generate`` always uploads
    rendered shards to ``spec.r2``, wherever the spec itself came from. Runs
    with no loggers — wandb tracking belongs to the launcher that
    materialized the spec.

    Invoke from the checkout root — the same contract as the Hydra launcher
    (the render subprocess script and any relative ``render.preset_path`` /
    ``render.plugin_path`` in the spec resolve against the process CWD; the
    SkyPilot worker changes directory to the checkout before exec for the
    same reason). Shards are written under
    ``logs/generate_dataset/from_spec_uri/<run_id>/`` relative to that CWD
    (mirroring Hydra's CWD-relative ``logs/`` run-dir convention) before
    their rclone upload; re-running the same URI reuses the dir and skips
    shards already present in R2 (#750).

    :param spec_uri: Bare local path, ``file://``, ``r2://``, or ``s3://`` URI
        of an ``input_spec.json``.
    :raises RuntimeError: R2 credentials are absent/invalid (from
        ``r2_io.ensure_r2_env_loaded``), or a render subprocess exits zero
        without writing its shard (from
        :func:`~synth_setter.cli.generate_dataset.generate`).
    :raises ValueError: ``spec_uri`` carries an unsupported scheme, or the
        fetched JSON is not a valid ``DatasetSpec`` (``pydantic.ValidationError``).
    """
    # Creds gate both the r2://-spec fetch and the shard uploads generate()
    # always performs, so validate them before touching the URI.
    r2_io.ensure_r2_env_loaded()
    spec = load_spec_from_uri(spec_uri)
    logger.info(f"loaded spec {spec.run_id} from {spec_uri}")
    work_dir = Path("logs") / "generate_dataset" / "from_spec_uri" / spec.run_id
    generate(spec, work_dir, [])


def main(argv: list[str] | None = None) -> None:
    """Parse the single spec-URI positional and render that spec.

    :param argv: argv tail (without the program name); ``None`` reads
        ``sys.argv[1:]`` (the console-script path). Injectable for tests.
    """
    parser = argparse.ArgumentParser(
        prog="synth-setter-generate-dataset-from-spec-uri",
        description=(
            "Render a dataset from an already-materialized input_spec.json. "
            "Run from the checkout root."
        ),
    )
    parser.add_argument(
        "spec_uri",
        help="bare local path, file://, r2://, or s3:// URI of an input_spec.json",
    )
    args = parser.parse_args(argv)
    run_from_spec_uri(args.spec_uri)


if __name__ == "__main__":
    main()
