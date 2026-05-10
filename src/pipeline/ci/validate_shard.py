#!/usr/bin/env python3
"""Validate dataset shards against a DatasetSpec.

Performs full per-shard validation: each shard file is a valid HDF5 (.h5) or
tar (.tar) shard, contains every expected per-row array, and each row count
matches ``spec.render.batch_per_shard``. The expected array names come from
``src.data.vst.generate_vst_dataset.DATASET_FIELD_NAMES`` — the writer's
own emission contract — so adding a new persisted field at the writer
auto-extends validation here.

CLI usage:
    python3 -m src.pipeline.ci.validate_shard <spec.json|r2://bucket/spec.json>

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

from src.data.vst.generate_vst_dataset import DATASET_FIELD_NAMES
from src.pipeline.r2_io import downloaded_to_tempfile, is_r2_uri
from src.pipeline.schemas.shard_metadata import ShardMetadata
from src.pipeline.schemas.spec import EXTENSION_TO_OUTPUT_FORMAT, DatasetSpec

_TAR_METADATA_MEMBER = "metadata.json"


def validate_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Validate one shard against a DatasetSpec.

    Dispatches on the shard's filename suffix via
    ``EXTENSION_TO_OUTPUT_FORMAT``: ``.h5`` -> HDF5 path, ``.tar`` -> tar path.
    Any other suffix is rejected with an error naming the registered set so a
    typo / wrong-format file does not surface as a misleading "not valid HDF5".
    Both paths check that the expected per-row arrays (``DATASET_FIELD_NAMES``)
    exist and that each row count matches ``spec.render.batch_per_shard``.

    Returns list of error strings (empty = valid).
    """
    if not shard_path.exists():
        return [f"shard file not found: {shard_path}"]

    fmt = EXTENSION_TO_OUTPUT_FORMAT.get(shard_path.suffix)
    if fmt == "hdf5":
        return _validate_h5_shard(shard_path, spec)
    if fmt == "wds":
        return _validate_tar_shard(shard_path, spec)
    return [
        f"unsupported shard suffix {shard_path.suffix!r} "
        f"(expected one of: {sorted(EXTENSION_TO_OUTPUT_FORMAT)})"
    ]


def _validate_h5_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Validate an HDF5 shard's datasets and row counts."""
    try:
        f = h5py.File(shard_path, "r")
    except OSError:
        return [f"file is not valid HDF5: {shard_path}"]

    errors: list[str] = []
    with f:
        for name in DATASET_FIELD_NAMES:
            if name not in f:
                errors.append(f"missing dataset: {name!r}")
                continue

            row_count = cast(h5py.Dataset, f[name]).shape[0]
            if row_count != spec.render.batch_per_shard:
                errors.append(
                    f"dataset {name!r} has {row_count} rows, expected {spec.render.batch_per_shard}"
                )

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


def _validate_tar_shard(shard_path: Path, spec: DatasetSpec) -> list[str]:
    """Full-tier validation of a wds tar shard.

    Tar member layout: per-batch-keyed ``<batch_key>.<field>.npy`` plus a single
    ``metadata.json``. The summed row count across all batches per field must
    equal ``spec.render.batch_per_shard``; ``metadata.json`` must parse as a strict
    ``ShardMetadata``.
    """
    try:
        tar = tarfile.open(shard_path, mode="r:")
    except tarfile.TarError:
        return [f"file is not a valid uncompressed tar archive: {shard_path}"]

    errors: list[str] = []
    with tar:
        members = sorted(m.name for m in tar.getmembers())

        if _TAR_METADATA_MEMBER not in members:
            errors.append(f"missing tar member: {_TAR_METADATA_MEMBER!r}")
        else:
            errors.extend(_validate_tar_metadata(tar, _TAR_METADATA_MEMBER))

        rows_by_field: dict[str, int] = {field: 0 for field in DATASET_FIELD_NAMES}
        seen_by_field: dict[str, int] = {field: 0 for field in DATASET_FIELD_NAMES}
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

        for field in DATASET_FIELD_NAMES:
            if seen_by_field[field] == 0:
                errors.append(f"missing tar member: '*.{field}.npy'")
                continue
            if rows_by_field[field] != spec.render.batch_per_shard:
                errors.append(
                    f"field {field!r} summed {rows_by_field[field]} rows across "
                    f"{seen_by_field[field]} batch(es), expected {spec.render.batch_per_shard}"
                )

    return errors


def _load_spec(spec_arg: str) -> DatasetSpec:
    """Load a spec from a local path or `r2://bucket/key` URI."""
    if is_r2_uri(spec_arg):
        with downloaded_to_tempfile(spec_arg) as local_path:
            return DatasetSpec.model_validate_json(local_path.read_text())
    return DatasetSpec.model_validate_json(Path(spec_arg).read_text())


def _shard_uri(spec: DatasetSpec, shard_filename: str) -> str:
    """Build the R2 URI where the spec says shard `shard_filename` lives."""
    return f"r2://{spec.r2_bucket}/{spec.r2_prefix}{shard_filename}"


def validate_all_shards_from_r2(spec: DatasetSpec) -> list[str]:
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
