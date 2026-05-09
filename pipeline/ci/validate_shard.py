#!/usr/bin/env python3
"""Validate dataset shards against a DatasetPipelineSpec.

Checks that each shard file is a valid HDF5 (.h5) or tar (.tar) shard, contains
the expected datasets (audio, mel_spec/mel, param_array), and that each
dataset's row count matches the spec's shard_size.

CLI usage:
    python3 -m pipeline.ci.validate_shard <spec.json|r2://bucket/spec.json>

Iterates `spec.shards` and downloads each shard from R2 (under
`r2://{spec.r2_bucket}/{spec.r2_prefix}{shard.filename}`) before validating.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from pipeline.r2_io import downloaded_to_tempfile, is_r2_uri
from pipeline.schemas.spec import DatasetPipelineSpec

_EXPECTED_H5_DATASETS = ("audio", "mel_spec", "param_array")
_EXPECTED_TAR_MEMBERS = ("audio.npy", "mel.npy", "param_array.npy", "metadata.json")


def validate_shard(shard_path: Path, spec: DatasetPipelineSpec) -> list[str]:
    """Validate one shard against a DatasetPipelineSpec.

    Dispatches on suffix: ``.h5`` -> HDF5 path, ``.tar`` -> tar path. Both paths
    check that the expected per-row arrays exist and that each row count
    matches ``spec.shard_size``.

    Returns list of error strings (empty = valid).
    """
    if not shard_path.exists():
        return [f"shard file not found: {shard_path}"]

    if shard_path.suffix == ".tar":
        return _validate_tar_shard(shard_path, spec)
    return _validate_h5_shard(shard_path, spec)


def _validate_h5_shard(shard_path: Path, spec: DatasetPipelineSpec) -> list[str]:
    """Validate an HDF5 shard's datasets and row counts."""
    try:
        f = h5py.File(shard_path, "r")
    except OSError:
        return [f"file is not valid HDF5: {shard_path}"]

    errors: list[str] = []
    with f:
        for name in _EXPECTED_H5_DATASETS:
            if name not in f:
                errors.append(f"missing dataset: {name!r}")
                continue

            row_count = cast(h5py.Dataset, f[name]).shape[0]
            if row_count != spec.shard_size:
                errors.append(f"dataset {name!r} has {row_count} rows, expected {spec.shard_size}")

    return errors


def _validate_tar_shard(shard_path: Path, spec: DatasetPipelineSpec) -> list[str]:
    """Validate a tar shard's members and array row counts."""
    try:
        tar = tarfile.open(shard_path)
    except tarfile.TarError:
        return [f"file is not a valid tar archive: {shard_path}"]

    errors: list[str] = []
    with tar:
        members = {m.name for m in tar.getmembers()}
        missing = [name for name in _EXPECTED_TAR_MEMBERS if name not in members]
        if missing:
            for name in missing:
                errors.append(f"missing tar member: {name!r}")
            return errors

        for member_name, label in (
            ("audio.npy", "audio"),
            ("mel.npy", "mel_spec"),
            ("param_array.npy", "param_array"),
        ):
            extracted = tar.extractfile(member_name)
            if extracted is None:
                errors.append(f"unable to extract tar member: {member_name!r}")
                continue
            arr = np.load(io.BytesIO(extracted.read()))
            if arr.shape[0] != spec.shard_size:
                errors.append(
                    f"dataset {label!r} has {arr.shape[0]} rows, expected {spec.shard_size}"
                )

    return errors


def _load_spec(spec_arg: str) -> DatasetPipelineSpec:
    """Load a spec from a local path or `r2://bucket/key` URI."""
    if is_r2_uri(spec_arg):
        with downloaded_to_tempfile(spec_arg) as local_path:
            return DatasetPipelineSpec.model_validate_json(local_path.read_text())
    return DatasetPipelineSpec.model_validate_json(Path(spec_arg).read_text())


def _shard_uri(spec: DatasetPipelineSpec, shard_filename: str) -> str:
    """Build the R2 URI where the spec says shard `shard_filename` lives."""
    return f"r2://{spec.r2_bucket}/{spec.r2_prefix}{shard_filename}"


def validate_all_shards_from_r2(spec: DatasetPipelineSpec) -> list[str]:
    """Validate every shard in `spec.shards` by downloading from R2.

    Returns aggregated error strings across all shards (empty = all valid). Each error is prefixed
    with the shard filename so the source is obvious.
    """
    errors: list[str] = []
    for shard in spec.shards:
        shard_uri = _shard_uri(spec, shard.filename)
        with downloaded_to_tempfile(shard_uri) as local_shard:
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
