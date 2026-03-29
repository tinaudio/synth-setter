#!/usr/bin/env python3
"""Validate structural correctness of a materialized DatasetPipelineSpec JSON.

Checks that required fields are present, code_version is a valid git SHA, renderer_version is non-
empty, and shards is non-empty. Does NOT assert config-specific values (shard counts, seeds, etc.)
— that's the caller's job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REQUIRED_FIELDS = [
    "base_seed",
    "code_version",
    "created_at",
    "is_repo_dirty",
    "min_loudness",
    "num_params",
    "output_format",
    "param_spec",
    "plugin_path",
    "preset_path",
    "renderer_version",
    "run_id",
    "sample_batch_size",
    "sample_rate",
    "shard_size",
    "shards",
    "signal_duration_seconds",
    "splits",
    "velocity",
]


def main() -> None:
    """Validate a spec JSON file and print a summary."""
    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <spec.json>\n")
        sys.exit(1)

    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())

    # Required fields
    missing = [f for f in _REQUIRED_FIELDS if f not in spec]
    if missing:
        sys.stderr.write(f"FAIL: missing fields: {missing}\n")
        sys.exit(1)

    # code_version must be a 40-char hex git SHA
    cv = spec["code_version"]
    if not (len(cv) == 40 and all(c in "0123456789abcdef" for c in cv)):
        sys.stderr.write(f"FAIL: code_version is not a valid SHA: {cv}\n")
        sys.exit(1)

    # renderer_version must not be empty
    if not spec["renderer_version"]:
        sys.stderr.write("FAIL: renderer_version is empty\n")
        sys.exit(1)

    # shards must not be empty
    if not spec["shards"]:
        sys.stderr.write("FAIL: shards is empty\n")
        sys.exit(1)

    sys.stdout.write("All structural checks passed:\n")
    sys.stdout.write(f"  code_version:     {cv}\n")
    sys.stdout.write(f"  renderer_version: {spec['renderer_version']}\n")
    sys.stdout.write(f"  num_params:       {spec['num_params']}\n")
    sys.stdout.write(f"  num_shards:       {len(spec['shards'])}\n")


if __name__ == "__main__":
    main()
