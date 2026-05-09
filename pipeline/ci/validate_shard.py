#!/usr/bin/env python3
"""Validate dataset shards against a DatasetPipelineSpec.

Performs full per-shard validation: each shard file is a valid HDF5 (.h5) or
tar (.tar) shard, contains every expected per-row array, and each row count
matches ``spec.shard_size``. The expected array names are listed in
``_EXPECTED_H5_DATASETS`` / ``_TAR_ARRAY_FIELDS``.

CLI usage:
    python3 -m pipeline.ci.validate_shard <spec.json|r2://bucket/spec.json>

Iterates ``spec.shards`` and downloads each shard from R2 (under
``r2://{spec.r2_bucket}/{spec.r2_prefix}{shard.filename}``) before validating.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from pydantic import ValidationError

from pipeline.r2_io import downloaded_to_tempfile, is_r2_uri
from pipeline.schemas.shard_metadata import ShardMetadata
from pipeline.schemas.spec import DatasetPipelineSpec

_EXPECTED_H5_DATASETS = ("audio", "mel_spec", "param_array")
_TAR_METADATA_MEMBER = "metadata.json"
_TAR_ARRAY_FIELDS = ("audio", "mel_spec", "param_array")


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


def _validate_tar_metadata(tar: tarfile.TarFile, member_name: str) -> list[str]:
    """Validate the tar's ``metadata.json`` parses as a strict ``ShardMetadata``."""
    extracted = tar.extractfile(member_name)
    if extracted is None:
        return [f"unable to extract tar member: {member_name!r}"]
    payload = extracted.read()
    try:
        ShardMetadata.model_validate_json(payload)
    except ValidationError as exc:
        return [f"{member_name}: invalid ShardMetadata: {exc}"]
    return []


def _validate_tar_shard(shard_path: Path, spec: DatasetPipelineSpec) -> list[str]:
    """Full-tier validation of a wds tar shard.

    Tar member layout: per-batch-keyed ``<batch_key>.<field>.npy`` plus a single
    ``metadata.json``. The summed row count across all batches per field must
    equal ``spec.shard_size``; ``metadata.json`` must parse as a strict
    ``ShardMetadata``.
    """
    try:
        tar = tarfile.open(shard_path)
    except tarfile.TarError:
        return [f"file is not a valid tar archive: {shard_path}"]

    errors: list[str] = []
    with tar:
        members = sorted(m.name for m in tar.getmembers())

        if _TAR_METADATA_MEMBER not in members:
            errors.append(f"missing tar member: {_TAR_METADATA_MEMBER!r}")
        else:
            errors.extend(_validate_tar_metadata(tar, _TAR_METADATA_MEMBER))

        rows_by_field: dict[str, int] = {field: 0 for field in _TAR_ARRAY_FIELDS}
        seen_by_field: dict[str, int] = {field: 0 for field in _TAR_ARRAY_FIELDS}
        for name in members:
            if not name.endswith(".npy"):
                continue
            stem, _, _ = name.rpartition(".")
            _, _, field = stem.rpartition(".")
            if field not in rows_by_field:
                continue
            extracted = tar.extractfile(name)
            if extracted is None:
                errors.append(f"unable to extract tar member: {name!r}")
                continue
            try:
                arr = np.load(io.BytesIO(extracted.read()))
            except (ValueError, EOFError, OSError) as exc:
                errors.append(f"{name}: malformed npy payload: {exc}")
                continue
            rows_by_field[field] += arr.shape[0]
            seen_by_field[field] += 1

        for field in _TAR_ARRAY_FIELDS:
            if seen_by_field[field] == 0:
                errors.append(f"missing tar member: '*.{field}.npy'")
                continue
            if rows_by_field[field] != spec.shard_size:
                errors.append(
                    f"field {field!r} summed {rows_by_field[field]} rows across "
                    f"{seen_by_field[field]} batch(es), expected {spec.shard_size}"
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
