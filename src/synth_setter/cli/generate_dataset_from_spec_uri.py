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
import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from synth_setter.cli.generate_dataset import generate
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.spec_io import load_spec_from_uri

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger

    from synth_setter.pipeline.schemas.spec import DatasetSpec

_RESUME_WANDB_JOB_TYPE = "data-generation-resume"


def _resume_loggers(spec: DatasetSpec, work_dir: Path) -> list[Logger]:
    """Create the default W&B logger for a from-spec-URI repair attempt.

    :param spec: Frozen dataset spec being resumed.
    :param work_dir: Local shard/log directory for this resume attempt.
    :returns: A single ``WandbLogger`` grouped under the original dataset run id.
    """
    import wandb
    from lightning.pytorch.loggers.wandb import WandbLogger

    return [
        WandbLogger(
            save_dir=str(work_dir),
            name=f"resume-{spec.task_name}-{spec.run_id}",
            project=os.environ.get("WANDB_PROJECT") or "synth-setter",
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=spec.run_id,
            job_type=_RESUME_WANDB_JOB_TYPE,
            tags=["from-spec-uri", "resume", spec.task_name],
            log_model=False,
            settings=wandb.Settings(
                code_dir=".",
                console="wrap",
                console_multipart=True,
            ),
        )
    ]


def run_from_spec_uri(spec_uri: str, *, enable_wandb: bool = True) -> None:  # noqa: DOC502 — raises propagate from callees
    """Load a materialized ``DatasetSpec`` from ``spec_uri`` and render it in-process.

    R2 credentials are validated up front because ``generate`` always uploads
    rendered shards to ``spec.r2``, wherever the spec itself came from. By
    default this recovery path creates a new W&B run grouped under the original
    dataset run id, so concurrent repairs do not write into the same W&B run
    history; pass ``enable_wandb=False`` for auth-free emergency repairs or CI.

    Invoke from the checkout root — the same contract as the Hydra launcher
    (the render subprocess script and any relative ``render.preset_path`` /
    ``render.plugin_path`` in the spec resolve against the process CWD; the
    SkyPilot worker changes directory to the checkout before exec for the
    same reason). Shards are written under
    ``logs/generate_dataset/from_spec_uri/<run_id>/`` relative to that CWD
    (mirroring Hydra's CWD-relative ``logs/`` run-dir convention) before
    their rclone upload; re-running the same URI reuses the dir and skips
    shards already present in R2 (#750).

    :param spec_uri: Bare local path, ``file://``, ``r2://``, or ``s3://`` URI of
        an ``input_spec.json``.
    :param enable_wandb: Whether to create a grouped W&B repair run.
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
    loggers = _resume_loggers(spec, work_dir) if enable_wandb else []
    generate(spec, work_dir, loggers)


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
        "--no-wandb",
        dest="enable_wandb",
        action="store_false",
        help="disable the default W&B logging resume",
    )
    parser.add_argument(
        "spec_uri",
        help="bare local path, file://, r2://, or s3:// URI of an input_spec.json",
    )
    args = parser.parse_args(argv)
    run_from_spec_uri(args.spec_uri, enable_wandb=args.enable_wandb)


if __name__ == "__main__":
    main()
