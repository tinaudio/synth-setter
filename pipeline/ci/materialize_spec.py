#!/usr/bin/env python3
"""Smoke test: materialize a DatasetSpec and write it to disk.

Bridge for callers that haven't migrated to ``@hydra.main``. Removed in
Phase A.3 once the entrypoint composes the spec via Hydra defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipeline.constants import INPUT_SPEC_FILENAME
from pipeline.schemas.spec import load_dataset_spec_yaml


def main() -> None:
    """Materialize a DatasetSpec from a config file and write JSON to disk."""
    if len(sys.argv) < 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <config_path> <output_dir>\n")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_dataset_spec_yaml(config_path)
    spec_json = spec.model_dump_json(indent=2)
    output_path = output_dir / INPUT_SPEC_FILENAME
    output_path.write_text(spec_json)

    sys.stdout.write(spec_json + "\n")
    sys.stderr.write(f"\nSpec written to {output_path}\n")


if __name__ == "__main__":
    main()
