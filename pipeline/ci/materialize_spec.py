#!/usr/bin/env python3
"""Compose a DatasetSpec from a Hydra experiment and write it to disk as JSON.

Used by CI smoke workflows that need an on-disk ``input_spec.json`` before running the
generator or validating spec structure. Replaces the legacy YAML-path interface with an
experiment-name interface that drives Hydra compose under the hood.

Usage::

    python -m pipeline.ci.materialize_spec <experiment> <output_dir>

The composed spec is written to ``<output_dir>/input_spec.json`` and echoed on stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import rootutils
from hydra import compose, initialize_config_dir

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pipeline.constants import INPUT_SPEC_FILENAME  # noqa: E402
from pipeline.entrypoints.generate_dataset import spec_from_cfg  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
REPO_ROOT = CONFIG_DIR.parent


def main() -> None:
    """Compose the named experiment + write the JSON spec under ``output_dir``."""
    if len(sys.argv) < 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <experiment> <output_dir>\n")
        sys.exit(1)

    experiment = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="dataset", overrides=[f"experiment={experiment}"])
    cfg.paths.root_dir = str(REPO_ROOT)
    cfg.paths.output_dir = str(REPO_ROOT)
    cfg.paths.work_dir = str(REPO_ROOT)
    spec = spec_from_cfg(cfg)

    spec_json = spec.model_dump_json(indent=2)
    output_path = output_dir / INPUT_SPEC_FILENAME
    output_path.write_text(spec_json)

    sys.stdout.write(spec_json + "\n")
    sys.stderr.write(f"\nSpec written to {output_path}\n")


if __name__ == "__main__":
    main()
