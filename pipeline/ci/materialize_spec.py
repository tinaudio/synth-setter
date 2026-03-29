#!/usr/bin/env python3
"""Smoke test: materialize a DatasetPipelineSpec and write it to disk."""

from __future__ import annotations

import sys
from pathlib import Path

from pipeline.constants import INPUT_SPEC_FILENAME
from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.spec import materialize_spec


def main() -> None:
    """Materialize a DatasetPipelineSpec from a config file and write JSON to disk."""
    if len(sys.argv) < 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <config_path> <output_dir>\n")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_dataset_config(config_path)
    cid = dataset_config_id_from_path(config_path)
    spec = materialize_spec(cfg, cid)

    spec_json = spec.model_dump_json(indent=2)
    output_path = output_dir / INPUT_SPEC_FILENAME
    output_path.write_text(spec_json)

    sys.stdout.write(spec_json + "\n")
    sys.stderr.write(f"\nSpec written to {output_path}\n")


if __name__ == "__main__":
    main()
