#!/usr/bin/env python3
"""Validate HDF5 shards against a DatasetSpec.

Checks that each shard file is a valid HDF5 file, contains the expected
datasets (audio, mel_spec, param_array), and that each dataset's row count
matches ``spec.render.batch_per_shard``.

CLI usage:
    python3 -m src.pipeline.ci.validate_shard <spec.json|r2://bucket/spec.json>

Iterates `spec.shards` and downloads each shard from R2 (under
`r2://{spec.r2_bucket}/{spec.r2_prefix}{shard.filename}`) before validating.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import h5py

from src.pipeline.r2_io import downloaded_to_tempfile, is_r2_uri, shard_uri
from src.pipeline.schemas.spec import DatasetSpec

_EXPECTED_DATASETS = ("audio", "mel_spec", "param_array")


def validate_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Validate one HDF5 shard against a DatasetSpec.

    Checks:
    1. File opens as HDF5
    2. Contains expected datasets: audio, mel_spec, param_array
    3. Each dataset's row count (shape[0]) matches spec.render.batch_per_shard

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
            if row_count != spec.render.batch_per_shard:
                errors.append(
                    f"dataset {name!r} has {row_count} rows, "
                    f"expected {spec.render.batch_per_shard}"
                )

    return errors


def _load_spec(spec_arg: str) -> DatasetSpec:
    """Load a spec from a local path or `r2://bucket/key` URI."""
    if is_r2_uri(spec_arg):
        with downloaded_to_tempfile(spec_arg) as local_path:
            return DatasetSpec.model_validate_json(local_path.read_text())
    return DatasetSpec.model_validate_json(Path(spec_arg).read_text())


def validate_all_shards_from_r2(spec: DatasetSpec) -> list[str]:
    """Validate every shard in `spec.shards` by downloading from R2.

    Returns aggregated error strings across all shards (empty = all valid). Each error is prefixed
    with the shard filename so the source is obvious.
    """
    errors: list[str] = []
    for shard in spec.shards:
        shard_object_uri = shard_uri(spec.r2_bucket, spec.r2_prefix, shard.filename)
        with downloaded_to_tempfile(shard_object_uri) as local_shard:
            shard_errors = validate_shard(local_shard, spec)
        for err in shard_errors:
            errors.append(f"{shard.filename}: {err}")
    return errors


def main() -> None:
    """CLI entry point: validate every shard referenced by a spec.

    The single argument is a spec JSON path or `r2://bucket/key.json` URI.
    Each shard listed in `spec.shards` is fetched from R2 and validated.
    """
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <spec.json|r2://bucket/spec.json>\n")
        sys.exit(1)

    spec_arg = sys.argv[1]
    spec = _load_spec(spec_arg)

    errors = validate_all_shards_from_r2(spec)

    if errors:
        for error in errors:
            sys.stderr.write(f"FAIL: {error}\n")
        sys.exit(1)

    sys.stdout.write(f"OK: all {len(spec.shards)} shards valid\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
