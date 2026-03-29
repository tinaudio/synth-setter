#!/usr/bin/env python3
"""Validate an HDF5 shard against a DatasetPipelineSpec.

Checks that the shard file is a valid HDF5 file, contains the expected
datasets (audio, mel_spec, param_array), and that each dataset's row count
matches the spec's shard_size.

CLI usage:
    python3 -m pipeline.ci.validate_shard <spec_json_path> <shard_h5_path>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import h5py

from pipeline.schemas.spec import DatasetPipelineSpec

_EXPECTED_DATASETS = ("audio", "mel_spec", "param_array")


def validate_shard(shard_path: Path, spec: DatasetPipelineSpec) -> list[str]:
    """Validate an HDF5 shard against a DatasetPipelineSpec.

    Checks:
    1. File opens as HDF5
    2. Contains expected datasets: audio, mel_spec, param_array
    3. Each dataset's row count (shape[0]) matches spec.shard_size

    Returns list of error strings (empty = valid).
    """
    if not shard_path.exists():
        return [f"shard file not found: {shard_path}"]

    try:
        f = h5py.File(shard_path, "r")
    except OSError:
        return [f"file is not valid HDF5: {shard_path}"]

    errors: list[str] = []
    with f:
        for name in _EXPECTED_DATASETS:
            if name not in f:
                errors.append(f"missing dataset: {name!r}")
                continue

            row_count = cast(h5py.Dataset, f[name]).shape[0]
            if row_count != spec.shard_size:
                errors.append(f"dataset {name!r} has {row_count} rows, expected {spec.shard_size}")

    return errors


def main() -> None:
    """CLI entry point: validate a shard HDF5 against a spec JSON."""
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <spec_json_path> <shard_h5_path>\n")
        sys.exit(1)

    spec_json_path = Path(sys.argv[1])
    shard_path = Path(sys.argv[2])

    spec = DatasetPipelineSpec.model_validate_json(spec_json_path.read_text())
    errors = validate_shard(shard_path, spec)

    if errors:
        for error in errors:
            sys.stderr.write(f"FAIL: {error}\n")
        sys.exit(1)

    sys.stdout.write(f"OK: {shard_path.name} is valid\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
